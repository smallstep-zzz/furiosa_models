"""Device-resident convolution packing via public TCL fetch.pad/slide."""
from __future__ import annotations
from functools import lru_cache
import torch
import furiosa.tcl as tcl

@tcl.tensor_operation
def windows(value, unit, N, C, H, W, Hf, Wf, Hw, Ww, padding, stride):
    a = tcl.fetch.pad(value, axis=H, left=padding, right=padding, fill=0.0)
    b = tcl.fetch.pad(a, axis=W, left=padding, right=padding, fill=0.0)
    c = tcl.fetch.slide(b, axis=H, frame_axis=Hf, window_axis=Hw, stride=stride, dilation=1)
    d = tcl.fetch.slide(c, axis=W, frame_axis=Wf, window_axis=Ww, stride=stride, dilation=1)
    e = tcl.operation.contract(d, unit, to=[N, Hf, Wf, Hw, Ww, C])
    return tcl.commit.typecast(e, tcl.bf16)

@tcl.kernel
def window_kernel(value: tcl.Tensor[tcl.bf16, [N, C, H, W]],
                  unit: tcl.Tensor[tcl.bf16, [N]], Hf: tcl.Axis, Wf: tcl.Axis, Hw: tcl.Axis, Ww: tcl.Axis,
                  padding: tcl.constexpr[int], stride: tcl.constexpr[int]) -> tcl.Tensor[tcl.bf16, [N, Hf, Wf, Hw, Ww, C]]:
    return windows(value, unit, N=N, C=C, H=H, W=W, Hf=Hf, Wf=Wf, Hw=Hw, Ww=Ww, padding=padding, stride=stride)

@lru_cache(maxsize=64)
def build(shape, kernel, padding, stride):
    import time
    started = time.perf_counter()
    from furiosa.torch import CompileModule, TclModule
    n,c,h,w=shape
    hf=(h+2*padding-kernel)//stride+1; wf=(w+2*padding-kernel)//stride+1
    k=tcl.specialize_kernel(window_kernel,dict(N=n,C=c,H=h,W=w,Hf=hf,Wf=wf,Hw=kernel,Ww=kernel),constexpr=dict(padding=padding,stride=stride))
    exported=torch.export.export(TclModule(k),(torch.empty(shape,dtype=torch.bfloat16),torch.ones(n,dtype=torch.bfloat16)),strict=False)
    result = CompileModule.from_exported(exported).to('furiosa:0'), torch.ones(n,dtype=torch.bfloat16,device='furiosa:0')
    from .work_metrics import record
    record('tcl_specialize_export_compile_prepare', time.perf_counter()-started)
    return result

def pack_windows(value,kernel=3,padding=1,stride=1):
    if value.device.type!='furiosa' or value.dtype!=torch.bfloat16:
        raise ValueError('window packing requires resident BF16 NPU input')
    h,w=value.shape[-2:]
    if kernel == 1 and padding == 0 and stride == 1:
        return value.permute(0,2,3,1).contiguous().reshape(value.shape[0],h,w,1,1,value.shape[1])
    if kernel == 3 and padding == 1 and stride == 1 and h >= 32 and w >= 32 and (h > 32 or w > 32):
        rows=[]
        for hs in range(0,h,30):
            cols=[]
            for ws in range(0,w,30):
                he=min(hs+30,h); we=min(ws+30,w)
                loh=min(max(0,hs-1),h-32); low=min(max(0,ws-1),w-32)
                patch=value[:,:,loh:loh+32,low:low+32].contiguous()
                packed=_pack_direct(patch,kernel,padding,stride)
                cols.append(packed[:,hs-loh:he-loh,ws-low:we-low].contiguous())
            rows.append(torch.cat(cols,dim=2))
        return torch.cat(rows,dim=1)
    return _pack_direct(value,kernel,padding,stride)

def _pack_direct(value,kernel,padding,stride):
    c=value.shape[1]
    if c == 4 and tuple(value.shape[-2:]) == (32,32):
        value=value.repeat(1,160,1,1)
    from .work_metrics import record
    cache_before = build.cache_info()
    compiled, unit = build(tuple(value.shape),kernel,padding,stride)
    record("tcl_compiled_object_reuse" if build.cache_info().hits > cache_before.hits else "tcl_compiled_object_created")
    result=compiled(value,unit)[0]
    return result[...,:c].contiguous() if value.shape[1]!=c else result

"""Fuse strided window concatenation; keep SDK-sensitive zero padding separate."""
from functools import lru_cache
import torch
import torch.nn.functional as F
from furiosa.torch import CompileModule
from resident_model import CONFIG

class Pack(torch.nn.Module):
    def __init__(self, kernel, stride):
        super().__init__()
        self.kernel, self.stride = kernel, stride
    def forward(self, value):
        kh, kw = self.kernel
        sh, sw = self.stride
        oh, ow = (value.shape[-2]-kh)//sh+1, (value.shape[-1]-kw)//sw+1
        taps = [value[:, :, y:y+oh*sh:sh, x:x+ow*sw:sw].permute(0,2,3,1)
                for y in range(kh) for x in range(kw)]
        return torch.cat(taps, dim=-1).reshape(-1, value.shape[1]*kh*kw)

@lru_cache(maxsize=64)
def build_pack(shape, kernel, stride):
    print('compile_pack '+str((shape,kernel,stride)), flush=True)
    ep = torch.export.export(Pack(kernel, stride), (torch.empty(shape, dtype=torch.float32),))
    return CompileModule.from_exported(ep, compiler_config=CONFIG).to('furiosa:0')

def compiled_pack(self, value):
    if value.device.type != 'furiosa' or value.dtype != torch.float32:
        raise ValueError('Compiled packing requires resident FP32 NPU input')
    # SDK2026.4 fused transpose/reshape failed exactness on the DFL geometry.
    # Keep its previously validated resident packing; never route it to CPU.
    if tuple(value.shape) == (1,16,4,8400) and self.kernel_size == (1,1):
        from conv_adapter import SpatialLinearConv2d
        return SpatialLinearConv2d.pack(self, value)
    ph, pw = self.padding
    padded = value
    if pw:
        zeros = value.new_zeros(*value.shape[:-1], pw)
        padded = torch.cat((zeros, padded, zeros), dim=-1)
    if ph:
        zeros = value.new_zeros(*padded.shape[:-2], ph, padded.shape[-1])
        padded = torch.cat((zeros, padded, zeros), dim=-2)
    result = build_pack(tuple(padded.shape), self.kernel_size, self.stride)(padded)
    return F.pad(result, (0,self.inner_padding)) if self.inner_padding else result

"""FP32 fused convolution using SDK 2026.4 public convolution example."""
from __future__ import annotations
from functools import lru_cache
import torch
import furiosa.tcl as tcl
from furiosa.torch import TclModule, CompileModule
from furiosa.torch.config import CompilerConfig

@tcl.tensor_operation
def conv(value, weight, bias, N, K, H, W, Hf, Wf, Hw, Ww, padding, stride):
    padded_h = tcl.fetch.pad(value, axis=H, left=padding, right=padding, fill=0.0)
    padded = tcl.fetch.pad(padded_h, axis=W, left=padding, right=padding, fill=0.0)
    rows = tcl.fetch.slide(padded, axis=H, frame_axis=Hf, window_axis=Hw, stride=stride, dilation=1)
    windows = tcl.fetch.slide(rows, axis=W, frame_axis=Wf, window_axis=Ww, stride=stride, dilation=1)
    output = tcl.operation.contract(windows, weight, to=[N, K, Hf, Wf])
    return tcl.commit.typecast(tcl.operation.addf(output, bias), tcl.f32)

@tcl.kernel
def conv_kernel(value: tcl.Tensor[tcl.f32, [N, C, H, W]],
                weight: tcl.Tensor[tcl.f32, [K, C, Hw, Ww]],
                bias: tcl.Tensor[tcl.f32, [K]], Hf: tcl.Axis, Wf: tcl.Axis,
                padding: tcl.constexpr[int], stride: tcl.constexpr[int]) -> tcl.Tensor[tcl.f32, [N, K, Hf, Wf]]:
    return conv(value, weight, bias, N=N, K=K, H=H, W=W, Hf=Hf, Wf=Wf,
                Hw=Hw, Ww=Ww, padding=padding, stride=stride)

@lru_cache(maxsize=64)
def module(shape, weight_shape, padding=1, stride=2):
    n, c, h, w = shape
    k, wc, kh, kw = weight_shape
    if wc != c or min(n,c,h,w,k,kh,kw,stride) <= 0 or padding < 0:
        raise ValueError('Invalid dense convolution geometry')
    axes = dict(N=n,C=c,H=h,W=w,K=k,Hw=kh,Ww=kw,
                Hf=(h+2*padding-kh)//stride+1,Wf=(w+2*padding-kw)//stride+1)
    return TclModule(tcl.specialize_kernel(conv_kernel, axes, constexpr=dict(padding=padding,stride=stride)))

@lru_cache(maxsize=64)
def build(shape, weight_shape, padding=1, stride=2):
    args = (torch.empty(shape), torch.empty(weight_shape), torch.empty(weight_shape[0]))
    ep = torch.export.export(module_v1(shape,weight_shape,padding,stride), args, strict=False)
    config = CompilerConfig(implicit_type_casting=False, implicit_type_casting_all_activation=False)
    return CompileModule.from_exported(ep, compiler_config=config).to('furiosa:0')

def fused_conv(value, weight, bias, padding=1, stride=2):
    if any(x.device.type != 'furiosa' or x.dtype != torch.float32 or not x.is_contiguous()
           for x in (value,weight,bias)):
        raise ValueError('Requires contiguous resident FP32 NPU tensors')
    if tuple(bias.shape) != (weight.shape[0],):
        raise ValueError('Bias shape does not match output channels')
    result = build_v1(tuple(value.shape),tuple(weight.shape),padding,stride)(value,weight,bias)
    return result[0] if isinstance(result,(tuple,list)) else result

# Retain the direct public implementation above as v1 for reproduction.
module_v1 = module
build_v1 = build
fused_conv_v1 = fused_conv

class CompensatedConv(torch.nn.Module):
    """Three BF16 contractions, FP32 accumulation; no materialized windows."""
    def __init__(self, shape, weight_shape, padding=1, stride=2):
        super().__init__()
        self.convolution = module_v1(shape, weight_shape, padding, stride)

    def forward(self, value, weight_hi, weight_lo, bias):
        high = value.to(torch.bfloat16).to(torch.float32)
        low = (value - high).to(torch.bfloat16).to(torch.float32)
        zero = torch.zeros_like(bias)
        hh = self.convolution(high, weight_hi, zero)[0]
        hl = self.convolution(high, weight_lo, zero)[0]
        lh = self.convolution(low, weight_hi, zero)[0]
        return ((hh + hl) + lh + bias.reshape(1, -1, 1, 1),)


def prepare_weights(weight):
    """Prepare static weight components once, before resident execution."""
    if weight.device.type != 'cpu' or weight.dtype != torch.float32:
        raise ValueError('Weight preparation expects CPU FP32 weights')
    high = weight.to(torch.bfloat16).to(torch.float32)
    low = (weight - high).to(torch.bfloat16).to(torch.float32)
    return high.contiguous(), low.contiguous()

@lru_cache(maxsize=64)
def module(shape, weight_shape, padding=1, stride=2):
    return CompensatedConv(shape, weight_shape, padding, stride)

@lru_cache(maxsize=64)
def build(shape, weight_shape, padding=1, stride=2):
    args = (torch.empty(shape), torch.empty(weight_shape), torch.empty(weight_shape), torch.empty(weight_shape[0]))
    ep = torch.export.export(module(shape,weight_shape,padding,stride), args, strict=False)
    config = CompilerConfig(implicit_type_casting=False, implicit_type_casting_all_activation=False)
    return CompileModule.from_exported(ep, compiler_config=config).to('furiosa:0')

def fused_conv(value, weight_hi, weight_lo, bias, padding=1, stride=2):
    if any(x.device.type != 'furiosa' or x.dtype != torch.float32 or not x.is_contiguous()
           for x in (value,weight_hi,weight_lo,bias)):
        raise ValueError('Requires contiguous resident FP32 tensors with prepared weight components')
    result = build(tuple(value.shape),tuple(weight_hi.shape),padding,stride)(value,weight_hi,weight_lo,bias)
    return result[0] if isinstance(result,(tuple,list)) else result

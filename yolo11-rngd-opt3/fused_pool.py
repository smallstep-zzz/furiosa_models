"""Exact 5x5 max pool as separable maxima in one compiled graph."""
from functools import lru_cache
import torch
from furiosa.torch import CompileModule
from furiosa.torch.config import CompilerConfig
class Separable(torch.nn.Module):
 def forward(self,value):
  h,w=value.shape[-2:];p=torch.cat((value[...,:1],value[...,:1],value,value[...,-1:],value[...,-1:]),dim=-1)
  z=p[...,:w].contiguous()
  for i in range(1,5):z=torch.maximum(z,p[...,i:i+w].contiguous())
  p=torch.cat((z[...,:1,:],z[...,:1,:],z,z[...,-1:,:],z[...,-1:,:]),dim=-2)
  z=p[...,:h,:].contiguous()
  for i in range(1,5):z=torch.maximum(z,p[...,i:i+h,:].contiguous())
  return z
@lru_cache(maxsize=4)
def build_pool(shape):
 cfg=CompilerConfig(implicit_type_casting=False,implicit_type_casting_all_activation=False)
 return CompileModule.from_exported(torch.export.export(Separable(),(torch.empty(shape),)),compiler_config=cfg).to('furiosa:0')
class ResidentMaxPool(torch.nn.Module):
 def __init__(self,original):
  super().__init__()
  def pair(x):return (x,x) if isinstance(x,int) else tuple(x)
  if (pair(original.kernel_size),pair(original.stride),pair(original.padding),pair(original.dilation))!=((5,5),(1,1),(2,2),(1,1)) or original.ceil_mode or original.return_indices:raise ValueError('Only validated YOLO SPPF pool is supported')
 def forward(self,value):
  if tuple(value.shape)!=(1,128,20,20) or value.dtype!=torch.float32 or value.device.type!='furiosa':raise ValueError('Unvalidated pool shape/dtype/device')
  return build_pool(tuple(value.shape))(value)

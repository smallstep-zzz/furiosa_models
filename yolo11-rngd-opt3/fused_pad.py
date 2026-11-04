"""Compile zero-border concatenation; persistent zero tensors are model constants."""
from functools import lru_cache
import torch
from furiosa.torch import CompileModule
from resident_model import CONFIG
class Pad(torch.nn.Module):
 def forward(self,value,zw,zh):
  return torch.cat((zh,torch.cat((zw,value,zw),dim=-1),zh),dim=-2)
@lru_cache(maxsize=64)
def build_pad(shape,padding):
 n,c,h,w=shape;ph,pw=padding
 args=(torch.empty(shape),torch.zeros(n,c,h,pw),torch.zeros(n,c,ph,w+2*pw))
 exe=CompileModule.from_exported(torch.export.export(Pad(),args),compiler_config=CONFIG).to('furiosa:0')
 return exe,args[1].to('furiosa:0'),args[2].to('furiosa:0')
def pad(value,padding):
 assert value.device.type=='furiosa' and value.dtype==torch.float32
 exe,zw,zh=build_pad(tuple(value.shape),padding)
 return exe(value,zw,zh)

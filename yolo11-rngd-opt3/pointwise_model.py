"""Resident direct 1x1 convolution, avoiding NCHW-to-matrix round trips."""
from functools import lru_cache
import hashlib,json,os
import torch
from furiosa.torch import CompileModule
from furiosa.torch.config import CompilerConfig
from compiler_engine.renegade_diffusion_compiler import _record_graph
from compiler_engine.execution_auditor import record_function
class Pointwise(torch.nn.Module):
 def forward(self,x,w,b):return torch.nn.functional.conv2d(x,w,b)
@lru_cache(maxsize=64)
def build(shapes):
 args=tuple(torch.empty(s) for s in shapes)
 ep=torch.export.export(Pointwise(),args)
 return CompileModule.from_exported(ep,compiler_config=CompilerConfig(implicit_type_casting=False,implicit_type_casting_all_activation=False)).to('furiosa:0')
class DirectPointwise(torch.nn.Module):
 def __init__(self,m,name):
  super().__init__();self.name=name
  assert m.kernel_size==(1,1) and m.stride==(1,1) and m.padding==(0,0) and m.groups==1
  self.original_out=m.out_channels
  padded_out=max(16,1<<(m.out_channels-1).bit_length())
  self.register_buffer('weight',torch.nn.functional.pad(m.weight.detach(),(0,0,0,0,0,0,0,padded_out-m.out_channels)))
  bias=m.bias.detach().clone() if m.bias is not None else torch.zeros(m.out_channels)
  self.register_buffer('bias',torch.nn.functional.pad(bias,(0,padded_out-m.out_channels)))
 def forward(self,value):
  args=(value,self.weight,self.bias);shapes=tuple(tuple(t.shape) for t in args);exe=build(shapes)
  gid=hashlib.sha256(json.dumps(['direct_pointwise',shapes]).encode()).hexdigest()[:16]
  instance=f'{os.getpid()}-{id(exe)}';index=getattr(self,'executions',0)+1;self.executions=index;ops=['aten.conv2d']
  if index==1:_record_graph('yolo11',self.name,gid,1,ops,ops,False,'npu_compiled',phase='backend_returned',compile_instance_id=instance)
  tracing=index==1 or os.environ.get('FURIOSA_TORCH_PROFILE')=='trace'
  if tracing:_record_graph('yolo11',self.name,gid,1,ops,ops,False,'npu_execution_started',phase='first_execution_start' if index==1 else 'repeat_execution_start',compile_instance_id=instance,execution_index=index,example_inputs=args)
  with record_function(f'npu_graph::yolo11::{self.name}::{gid}::compile-{instance}::execution-{index}'):result=exe(*args)
  if tracing:_record_graph('yolo11',self.name,gid,1,ops,ops,False,'npu_compiled',phase='first_execution_completed' if index==1 else 'repeat_execution_completed',compile_instance_id=instance,execution_index=index)
  return result[:,:self.original_out]

"""Dense compensated TCL Conv2d adapter; whole-model acceptance is a separate gate."""
import torch
import os,hashlib,json
from compiler_engine.renegade_diffusion_compiler import _record_graph
from compiler_engine.execution_auditor import record_function
from tcl_conv_candidate import build,prepare_weights
class FusedConv(torch.nn.Module):
 def __init__(self,conv,name):
  super().__init__();self.name=name;self.padding=conv.padding[0];self.stride=conv.stride[0]
  assert conv.padding[0]==conv.padding[1] and conv.stride[0]==conv.stride[1] and conv.dilation==(1,1)
  w=conv.weight.detach()
  if conv.groups!=1:
   dense=w.new_zeros(conv.out_channels,conv.in_channels,*conv.kernel_size);ins=conv.in_channels//conv.groups;outs=conv.out_channels//conv.groups
   for g in range(conv.groups):dense[g*outs:(g+1)*outs,g*ins:(g+1)*ins]=w[g*outs:(g+1)*outs]
   w=dense
  self.original_out=conv.out_channels
  self.aligned = name == 'model/23/cv3/0/cv3/0/1/cv3/0/1/0/conv/Conv'
  bias=conv.bias.detach().clone() if conv.bias is not None else torch.zeros(conv.out_channels)
  if self.aligned:
   w=torch.nn.functional.pad(w,(0,0,0,0,0,48,0,48));bias=torch.nn.functional.pad(bias,(0,48));self.register_buffer('input_zeros',torch.zeros(1,48,80,80))
  hi,lo=prepare_weights(w)
  self.register_buffer('hi',hi);self.register_buffer('lo',lo);self.register_buffer('bias',bias)
 def forward(self,value):
  if self.aligned:
   assert tuple(value.shape)==(1,80,80,80)
   value=torch.cat((value,self.input_zeros),dim=1)
  args=(value.contiguous(),self.hi,self.lo,self.bias)
  exe=build(tuple(value.shape),tuple(self.hi.shape),self.padding,self.stride)
  gid=hashlib.sha256(json.dumps([list(value.shape),list(self.hi.shape),self.padding,self.stride,'compensated']).encode()).hexdigest()[:16]
  instance=f'{os.getpid()}-{id(exe)}';index=getattr(self,'executions',0)+1;self.executions=index
  ops=['public_tcl_compensated_conv']
  if index==1:_record_graph('yolo11',self.name,gid,1,ops,ops,False,'npu_compiled',phase='backend_returned',compile_instance_id=instance)
  tracing=index==1 or os.environ.get('FURIOSA_TORCH_PROFILE')=='trace'
  if tracing:_record_graph('yolo11',self.name,gid,1,ops,ops,False,'npu_execution_started',phase='first_execution_start' if index==1 else 'repeat_execution_start',compile_instance_id=instance,execution_index=index,example_inputs=args)
  with record_function(f'npu_graph::yolo11::{self.name}::{gid}::compile-{instance}::execution-{index}'):
   result=exe(*args)
  if tracing:_record_graph('yolo11',self.name,gid,1,ops,ops,False,'npu_compiled',phase='first_execution_completed' if index==1 else 'repeat_execution_completed',compile_instance_id=instance,execution_index=index)
  result=result[0] if isinstance(result,(tuple,list)) else result
  return result[:,:self.original_out].contiguous() if self.aligned else result

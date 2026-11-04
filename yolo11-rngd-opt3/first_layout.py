"""Exact compiled output layout for the first convolution only."""
import os,hashlib
from functools import lru_cache
import torch
from furiosa.torch import CompileModule
from resident_model import ResidentConv,CONFIG
from compiler_engine.renegade_diffusion_compiler import _record_graph
from compiler_engine.execution_auditor import record_function
class Layout(torch.nn.Module):
 def forward(self,y):return y.reshape(1,320,320,16).permute(0,3,1,2).contiguous()
@lru_cache(maxsize=1)
def build():return CompileModule.from_exported(torch.export.export(Layout(),(torch.empty(102400,16),)),compiler_config=CONFIG).to('furiosa:0')
class FirstConv(ResidentConv):
 def unpack(self,result,shape):
  assert tuple(shape)==(1,3,640,640) and tuple(result.shape)==(102400,16)
  exe=build();gid=hashlib.sha256(b'first-layout-v1').hexdigest()[:16];instance=f'{os.getpid()}-{id(exe)}';index=getattr(self,'layout_executions',0)+1;self.layout_executions=index
  name=self.name+'/layout';ops=['reshape','permute','contiguous']
  if index==1:_record_graph('yolo11',name,gid,3,ops,ops,False,'npu_compiled',phase='backend_returned',compile_instance_id=instance)
  tracing=index==1 or os.environ.get('FURIOSA_TORCH_PROFILE')=='trace'
  if tracing:_record_graph('yolo11',name,gid,3,ops,ops,False,'npu_execution_started',phase='first_execution_start' if index==1 else 'repeat_execution_start',compile_instance_id=instance,execution_index=index,example_inputs=(result,))
  with record_function(f'npu_graph::yolo11::{name}::{gid}::compile-{instance}::execution-{index}'):value=exe(result)
  if tracing:_record_graph('yolo11',name,gid,3,ops,ops,False,'npu_compiled',phase='first_execution_completed' if index==1 else 'repeat_execution_completed',compile_instance_id=instance,execution_index=index)
  return value

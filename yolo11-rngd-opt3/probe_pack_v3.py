import os,json,time,fcntl
from pathlib import Path
from yolo11_fxbex import configure
configure()
import torch
import torch.nn.functional as F
from furiosa.torch import CompileModule
from resident_model import CONFIG
from conv_adapter import SpatialLinearConv2d
from compiler_engine.execution_auditor import NPUExecutionAuditor
class Pack(torch.nn.Module):
 def __init__(self,c,s): super().__init__(); self.c,self.s=c,s
 def forward(self,value,zw,zh):
  padded=torch.cat((zw,value,zw),dim=-1);padded=torch.cat((zh,padded,zh),dim=-2)
  oh=(value.shape[-2]-1)//self.s+1; ow=(value.shape[-1]-1)//self.s+1
  taps=[padded[:,:,y:y+oh*self.s:self.s,x:x+ow*self.s:self.s].permute(0,2,3,1) for y in range(3) for x in range(3)]
  return torch.cat(taps,dim=-1).reshape(-1,self.c*9)
root=Path(__file__).resolve().parent;out=root/'runs'/('pack_v3_'+str(time.time_ns()));out.mkdir()
with open('/root/rngd-model-forge/reports/rngd_porting/npu.lock','a') as lock,torch.no_grad():
 fcntl.flock(lock,fcntl.LOCK_EX)
 for c,h,s in [(16,160,1),(64,160,2)]:
  torch.manual_seed(32);x=torch.randn(1,c,h,h);a=SpatialLinearConv2d(torch.nn.Conv2d(c,64,3,stride=s,padding=1));ref=a.pack(x)
  args=(x,torch.zeros(1,c,h,1),torch.zeros(1,c,1,h+2)); m=Pack(c,s)
  print('CASE '+str((c,h,s)),flush=True)
  ep=torch.export.export(m,args);exe=CompileModule.from_exported(ep,compiler_config=CONFIG).to('furiosa:0');r=tuple(t.to('furiosa:0') for t in args)
  def run():
   v=exe(*r);return F.pad(v,(0,a.inner_padding)) if a.inner_padding else v
  case=out/str(c);case.mkdir()
  with NPUExecutionAuditor(case,'pack-v3'):
   actual=run().cpu()
  torch.testing.assert_close(actual,ref,rtol=0,atol=0)
  times=[]
  for _ in range(3):
   t=time.perf_counter();run().cpu();times.append(time.perf_counter()-t)
  summary=json.loads((case/'summary.json').read_text());assert summary['profiler']['npu_task_events'] and not summary['cpu']['sdk_cpu_execution_count']
  result={'shape':list(x.shape),'exact':True,'warm_s':times,'tasks':summary['profiler']['npu_task_events']};(case/'result.json').write_text(json.dumps(result,indent=2));print(result,flush=True)

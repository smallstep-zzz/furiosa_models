import os, json, time, fcntl
from pathlib import Path
from yolo11_fxbex import configure
configure()
import torch
from conv_adapter import SpatialLinearConv2d
from fast_pack import compiled_pack
from compiler_engine.execution_auditor import NPUExecutionAuditor
root=Path(__file__).resolve().parent
out=root/'runs'/('pack_v2_'+str(time.time_ns()));out.mkdir()
results=[]
with open('/root/rngd-model-forge/reports/rngd_porting/npu.lock','a') as lock, torch.no_grad():
 fcntl.flock(lock,fcntl.LOCK_EX)
 for c,h,s in [(16,160,1),(64,160,2)]:
  torch.manual_seed(32); x=torch.randn(1,c,h,h); m=SpatialLinearConv2d(torch.nn.Conv2d(c,64,3,stride=s,padding=1)); ref=m.pack(x); resident=x.to('furiosa:0')
  case=out/str(c);case.mkdir()
  with NPUExecutionAuditor(case,'pack-v2'):
   actual=compiled_pack(m,resident).cpu()
  torch.testing.assert_close(actual,ref,rtol=0,atol=0)
  times=[]
  for _ in range(3):
   t=time.perf_counter(); z=compiled_pack(m,resident).cpu();times.append(time.perf_counter()-t)
  summary=json.loads((case/'summary.json').read_text())
  assert summary['profiler']['npu_task_events'] and not summary['cpu']['sdk_cpu_execution_count']
  results.append({'shape':list(x.shape),'warm_s':times,'exact':True,'tasks':summary['profiler']['npu_task_events']});print(results[-1],flush=True)
 (out/'result.json').write_text(json.dumps(results,indent=2))

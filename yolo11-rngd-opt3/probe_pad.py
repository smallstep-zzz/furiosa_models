import os,sys,time,json,fcntl
from pathlib import Path
os.environ.setdefault('RAYON_NUM_THREADS','16');sys.path.insert(0,'/root/rngd-model-forge')
import torch
from yolo11_fxbex import configure
configure()
from fused_pad import pad
from compiler_engine.execution_auditor import NPUExecutionAuditor
root=Path(__file__).resolve().parent;out=root/'runs'/('padding_'+str(time.time_ns()));out.mkdir(parents=True)
with open('/root/rngd-model-forge/reports/rngd_porting/npu.lock','a') as lock,torch.no_grad():
 fcntl.flock(lock,fcntl.LOCK_EX)
 results=[]
 for c,h in [(16,320),(64,160),(8,160),(3,640)]:
  torch.manual_seed(32);x=torch.randn(1,c,h,h);ref=torch.nn.functional.pad(x,(1,1,1,1));r=x.to('furiosa:0');pad(r,(1,1)).cpu();case=out/f'{c}_{h}';case.mkdir()
  with NPUExecutionAuditor(case,'padding'):
   actual=pad(r,(1,1)).cpu()
  torch.testing.assert_close(actual,ref,rtol=0,atol=0)
  ts=[]
  for _ in range(10):
   t=time.perf_counter();pad(r,(1,1)).cpu();ts.append(time.perf_counter()-t)
  summary=json.loads((case/'summary.json').read_text());assert summary['profiler']['npu_task_events'] and not summary['cpu']['sdk_cpu_execution_count'];result={'shape':list(x.shape),'times':ts,'tasks':summary['profiler']['npu_task_events'],'exact':True};results.append(result);print(result,flush=True)
 (out/'result.json').write_text(json.dumps(results,indent=2))

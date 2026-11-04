import os,sys,time,json,fcntl
from pathlib import Path
os.environ.setdefault('RAYON_NUM_THREADS','16');sys.path.insert(0,'/root/rngd-model-forge')
from yolo11_fxbex import configure
configure()
import torch
from furiosa.torch import CompileModule
from resident_model import CONFIG
from compiler_engine.execution_auditor import NPUExecutionAuditor
class Pointwise(torch.nn.Module):
 def forward(self,value,weight,bias):
  return torch.nn.functional.conv2d(value,weight,bias)
root=Path(__file__).resolve().parent;out=root/'runs'/('pointwise_'+str(time.time_ns()));out.mkdir(parents=True)
with open('/root/rngd-model-forge/reports/rngd_porting/npu.lock','a') as lock,torch.no_grad():
 fcntl.flock(lock,fcntl.LOCK_EX)
 torch.manual_seed(32);x=torch.randn(1,32,160,160);w=torch.randn(32,32,1,1)*.1;b=torch.randn(32)*.1;args=(x,w,b);ref=Pointwise()(*args)
 exe=CompileModule.from_exported(torch.export.export(Pointwise(),args),compiler_config=CONFIG).to('furiosa:0');r=tuple(t.to('furiosa:0') for t in args);exe(*r).cpu()
 with NPUExecutionAuditor(out,'pointwise'):
  actual=exe(*r).cpu()
 torch.testing.assert_close(actual,ref,rtol=1e-3,atol=1e-3)
 ts=[]
 for _ in range(10):
  t=time.perf_counter();exe(*r).cpu();ts.append(time.perf_counter()-t)
 s=json.loads((out/'summary.json').read_text());assert s['profiler']['npu_task_events'] and not s['cpu']['sdk_cpu_execution_count'];result={'warm_s':ts,'max_abs':float((actual-ref).abs().max()),'tasks':s['profiler']['npu_task_events'],'path':str(out)};(out/'result.json').write_text(json.dumps(result,indent=2));print(result,flush=True)

import time,json,fcntl
from pathlib import Path
from yolo11_optimized import configure
configure()
import torch
from activation_pointwise import DirectPointwise
from activation_fused import FusedConv
from compiler_engine.execution_auditor import NPUExecutionAuditor
root=Path(__file__).resolve().parent;out=root/'runs'/('activation_feature_'+str(time.time_ns()));out.mkdir(parents=True)
lock=open('/root/rngd-model-forge/reports/rngd_porting/npu.lock','a');fcntl.flock(lock,fcntl.LOCK_EX)
with torch.no_grad():
 torch.manual_seed(42);ready=[];rows=[]
 for k,shape in [(1,(1,32,160,160)),(3,(1,16,320,320))]:
  m=torch.nn.Conv2d(shape[1],32,k,padding=k//2).eval();x=torch.randn(shape);v=m(x);ref=v*torch.sigmoid(v)
  adapter=(DirectPointwise if k==1 else FusedConv)(m,'test'+str(k)).to('furiosa:0');r=x.to('furiosa:0');actual=adapter(r).cpu()
  torch.testing.assert_close(actual,ref,rtol=1e-3,atol=1e-3)
  ts=[]
  for _ in range(3):
   t=time.perf_counter();adapter(r).cpu();ts.append(time.perf_counter()-t)
  rows.append({'kernel':k,'max_abs':float((actual-ref).abs().max()),'warm_s':ts});ready.append((adapter,r,ref))
 anchor=torch.ones(32,32).to('furiosa:0');(anchor+1).cpu()
 with NPUExecutionAuditor(out,'conv-silu'):
  (anchor+1).cpu()
  for adapter,r,ref in ready:torch.testing.assert_close(adapter(r).cpu(),ref,rtol=1e-3,atol=1e-3)
 s=json.loads((out/'summary.json').read_text());assert s['profiler']['npu_task_events'] and not s['cpu']['sdk_cpu_execution_count']
 (out/'result.json').write_text(json.dumps(rows,indent=2));print('RESULT '+str(out));print(rows)

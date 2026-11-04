import sys, os, json, time, fcntl
from pathlib import Path
sys.path.insert(0,'/root/yolo11-rngd')
from yolo11_fxbex import configure
configure()
import torch
from conv_adapter import SpatialLinearConv2d
from resident_model import CONFIG
from furiosa.torch import CompileModule
from compiler_engine.execution_auditor import NPUExecutionAuditor
class Pack(torch.nn.Module):
 def __init__(self):
  super().__init__(); self.adapter=SpatialLinearConv2d(torch.nn.Conv2d(64,128,3,stride=2,padding=1))
 def forward(self,value): return self.adapter.pack(value)
root=Path('/root/yolo11-rngd/optimization'); root.mkdir(exist_ok=True)
out=root/('pack_'+str(time.time_ns())); out.mkdir()
with open('/root/rngd-model-forge/reports/rngd_porting/npu.lock','a') as lock, torch.no_grad():
 fcntl.flock(lock,fcntl.LOCK_EX)
 torch.manual_seed(32); x=torch.randn(1,64,160,160); m=Pack(); ref=m(x)
 ep=torch.export.export(m,(x,))
 exe=CompileModule.from_exported(ep,compiler_config=CONFIG).to('furiosa:0'); resident=x.to('furiosa:0')
 with NPUExecutionAuditor(out,'compiled-pack'):
  actual=exe(resident).cpu()
 torch.testing.assert_close(actual,ref,rtol=0,atol=0)
 timings=[]
 for _ in range(3):
  t=time.perf_counter(); z=exe(resident).cpu(); timings.append(time.perf_counter()-t)
 summary=json.loads((out/'summary.json').read_text())
 assert summary['profiler']['npu_task_events'] and not summary['cpu']['sdk_cpu_execution_count']
 print(json.dumps({'out':str(out),'warm_s':timings,'summary':summary['profiler']}),flush=True)

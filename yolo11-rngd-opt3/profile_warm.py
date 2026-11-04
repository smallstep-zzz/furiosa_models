import sys, os, time, json, fcntl
from pathlib import Path
ROOT=Path('/root/yolo11-rngd')
sys.path.insert(0,str(ROOT))
from yolo11_fxbex import configure, make_model
configure()
import torch, numpy as np
from compiler_engine.execution_auditor import NPUExecutionAuditor, record_function
out=ROOT/'runs'/('warm_profile_'+str(time.time_ns())); out.mkdir()
os.environ['FURIOSA_GRAPH_AUDIT_PATH']=str(out/'compiler_graphs.jsonl')
lock=open('/root/rngd-model-forge/reports/rngd_porting/npu.lock','a'); fcntl.flock(lock,fcntl.LOCK_EX)
with torch.no_grad():
 model=make_model(ROOT/'models/yolo11n_640.onnx',ROOT/'static_metadata.json')
 x=torch.from_numpy(np.load(ROOT/'image_reference.npz')['input']).to('furiosa:0')
 model(x).cpu()
 times=[]
 for _ in range(3):
  t=time.perf_counter(); y=model(x).cpu(); times.append(time.perf_counter()-t)
 class Timed(torch.fx.Interpreter):
  def run_node(self,node):
   t=time.perf_counter()
   with record_function('node::'+node.name): result=super().run_node(node)
   durations.append((node.name,str(node.target),time.perf_counter()-t))
   return result
 durations=[]
 with NPUExecutionAuditor(out,'warm-yolo'):
  z=Timed(model).run(x).cpu()
 torch.testing.assert_close(y,z,rtol=0,atol=0)
 report={'warm_s':times,'nodes':sorted(durations,key=lambda r:-r[2])}
 (out/'timing.json').write_text(json.dumps(report,indent=2)); print(str(out)); print(json.dumps(report),flush=True)

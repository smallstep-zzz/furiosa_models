import time,json,fcntl
from pathlib import Path
from yolo11_optimized import configure
configure()
import torch,onnx,numpy as np
from onnx2torch import convert
from resident_model import ResidentConv,CONFIG
from furiosa.torch import CompileModule
from compiler_engine.execution_auditor import NPUExecutionAuditor
root=Path(__file__).resolve().parent;out=root/'runs'/('linear_layout_'+str(time.time_ns()));out.mkdir(parents=True)
class LinearLayout(torch.nn.Module):
 def forward(self,x,w,b):
  y=torch.nn.functional.linear(x,w,b)
  return y.reshape(1,320,320,16).permute(0,3,1,2).contiguous()
lock=open('/root/rngd-model-forge/reports/rngd_porting/npu.lock','a');fcntl.flock(lock,fcntl.LOCK_EX)
with torch.no_grad():
 model=convert(onnx.load(root/'models/yolo11n_640.onnx')).eval();m=model.get_submodule('model/0/conv/Conv')
 x=torch.from_numpy(np.load(root/'image_reference.npz')['input']);ref=m(x)
 old=ResidentConv(m,'first').to('furiosa:0');r=x.to('furiosa:0');packed=old.pack(r)
 args=(packed,old.linear.weight,old.linear.bias)
 cpu_args=tuple(torch.empty(tuple(v.shape)) for v in args)
 exe=CompileModule.from_exported(torch.export.export(LinearLayout(),cpu_args),compiler_config=CONFIG).to('furiosa:0')
 actual=exe(*args).cpu();torch.testing.assert_close(actual,ref,rtol=1e-3,atol=1e-3)
 old_y=old(r).cpu();torch.testing.assert_close(actual,old_y,rtol=0,atol=0)
 ts={'old':[],'fused':[]}
 for _ in range(5):
  t=time.perf_counter();old(r).cpu();ts['old'].append(time.perf_counter()-t)
  t=time.perf_counter();exe(old.pack(r),old.linear.weight,old.linear.bias).cpu();ts['fused'].append(time.perf_counter()-t)
 anchor=torch.ones(32,32).to('furiosa:0');(anchor+1).cpu()
 with NPUExecutionAuditor(out,'linear-layout'):
  (anchor+1).cpu();actual=exe(*args).cpu()
 summary=json.loads((out/'summary.json').read_text());assert summary['profiler']['npu_task_events'] and not summary['cpu']['sdk_cpu_execution_count']
 result={'warm_s':ts,'bitwise_equal_old':True,'max_abs':float((actual-ref).abs().max()),'path':str(out)}
 (out/'result.json').write_text(json.dumps(result,indent=2));print(result)

import os,sys,time,json,fcntl
from pathlib import Path
os.environ.setdefault('RAYON_NUM_THREADS','16');sys.path.insert(0,'/root/rngd-model-forge')
from yolo11_fxbex import configure
configure()
import torch,onnx,numpy as np
from onnx2torch import convert
from tcl_conv_candidate import fused_conv,prepare_weights
from compiler_engine.execution_auditor import NPUExecutionAuditor
root=Path(__file__).resolve().parent;out=root/'runs'/('channel_pad_'+str(time.time_ns()));out.mkdir(parents=True)
with torch.no_grad():
 model=convert(onnx.load(root/'models/yolo11n_640.onnx')).eval();name='model/23/cv3/0/cv3/0/1/cv3/0/1/0/conv/Conv';m=model.get_submodule(name);captured=[];m.register_forward_pre_hook(lambda _,a:captured.append(a[0].clone()));model(torch.from_numpy(np.load(root/'image_reference.npz')['input']));x=captured[0];ref=m(x)
 w=torch.zeros(128,128,3,3)
 for i in range(80):w[i,i]=m.weight[i,0]
 bias=torch.zeros(128);bias[:80]=m.bias;hi,lo=prepare_weights(w)
 with open('/root/rngd-model-forge/reports/rngd_porting/npu.lock','a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);r=x.to('furiosa:0');zero=torch.zeros(1,48,80,80).to('furiosa:0');args=tuple(t.to('furiosa:0') for t in (hi,lo,bias))
  def run():return fused_conv(torch.cat((r,zero),dim=1),*args,padding=1,stride=1)[:,:80].contiguous()
  run().cpu()
  with NPUExecutionAuditor(out,'channel-padded-conv'):
   y=run().cpu()
  torch.testing.assert_close(y,ref,rtol=1e-3,atol=1e-3);times=[]
  for _ in range(5):
   t=time.perf_counter();run().cpu();times.append(time.perf_counter()-t)
  s=json.loads((out/'summary.json').read_text());assert s['profiler']['npu_task_events'] and not s['cpu']['sdk_cpu_execution_count'];result={'warm_s':times,'max_abs':float((y-ref).abs().max()),'tasks':s['profiler']['npu_task_events'],'path':str(out)};(out/'result.json').write_text(json.dumps(result,indent=2));print(result,flush=True)

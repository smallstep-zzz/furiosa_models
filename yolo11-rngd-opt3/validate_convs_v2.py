import os,sys,time,json,fcntl,traceback
from pathlib import Path
os.environ.setdefault('RAYON_NUM_THREADS','16');os.environ['FURIOSA_REQUIRE_NPU']='1';sys.path.insert(0,'/root/rngd-model-forge')
from yolo11_fxbex import configure
configure()
import torch,onnx,numpy as np
from onnx2torch import convert
from fused_model import FusedConv
from compiler_engine.execution_auditor import NPUExecutionAuditor
root=Path(__file__).resolve().parent;out=root/'runs'/('fused_convs_v2_'+str(time.time_ns()));out.mkdir(parents=True);fixtures={}
def hook(name):
 def capture(m,args):
  x=args[0];key=(tuple(x.shape),tuple(m.weight.shape),m.stride,m.padding,m.groups)
  if key not in fixtures:fixtures[key]=(name,m,x.detach().clone())
 return capture
with torch.no_grad():
 model=convert(onnx.load(root/'models/yolo11n_640.onnx')).eval()
 for name,m in model.named_modules():
  if isinstance(m,torch.nn.Conv2d):m.register_forward_pre_hook(hook(name))
 model(torch.from_numpy(np.load(root/'image_reference.npz')['input']))
 results=[];ready=[]
 with open('/root/rngd-model-forge/reports/rngd_porting/npu.lock','a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  for i,(key,(name,m,x)) in enumerate(fixtures.items()):
   if m.kernel_size==(1,1) or m.in_channels==3:continue
   case=out/str(i);case.mkdir();print('CASE '+str(i)+' '+name+' '+str(key),flush=True)
   row={'name':name,'signature':key}
   try:
    ref=m(x);adapter=FusedConv(m,name).to('furiosa:0');r=x.to('furiosa:0');adapter(r).cpu()
    y=adapter(r).cpu()
    row.update(max_abs=float((y-ref).abs().max()),passed=bool(torch.allclose(y,ref,rtol=1e-3,atol=1e-3)))
    torch.testing.assert_close(y,ref,rtol=1e-3,atol=1e-3)
    times=[]
    for _ in range(3):
     t=time.perf_counter();adapter(r).cpu();times.append(time.perf_counter()-t)
    row.update(warm_s=times);ready.append((name,adapter,r,ref))
   except Exception as e:
    row.update(passed=False,error=str(e)[-2000:]);(case/'error.txt').write_text(traceback.format_exc())
   results.append(row);(out/'results.json').write_text(json.dumps(results,indent=2));print(json.dumps(row),flush=True)
  with NPUExecutionAuditor(out,'all-fused-convs') as audit:
   for name,adapter,r,ref in ready:
    with audit.span('model_component::'+name):y=adapter(r).cpu()
    torch.testing.assert_close(y,ref,rtol=1e-3,atol=1e-3)
  summary=json.loads((out/'summary.json').read_text())
  assert summary['profiler']['npu_task_events'] and not summary['cpu']['sdk_cpu_execution_count']
 print('DONE '+str(out),flush=True)

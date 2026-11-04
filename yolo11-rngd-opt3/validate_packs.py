import os, json, time, fcntl
from pathlib import Path
from yolo11_fxbex import configure
configure()
import torch, onnx, numpy as np
from onnx2torch import convert
from conv_adapter import SpatialLinearConv2d
from fast_pack import compiled_pack
from compiler_engine.execution_auditor import NPUExecutionAuditor
root=Path(__file__).resolve().parent
out=root/'runs'/('all_packs_'+str(time.time_ns()));out.mkdir()
fixtures={}
def hook(m,args):
 x=args[0];key=(tuple(x.shape),m.kernel_size,m.stride,m.padding)
 if key not in fixtures: fixtures[key]=(SpatialLinearConv2d(m),x.detach().clone())
with torch.no_grad():
 model=convert(onnx.load(root/'models/yolo11n_640.onnx')).eval()
 for m in model.modules():
  if isinstance(m,torch.nn.Conv2d):m.register_forward_pre_hook(hook)
 model(torch.from_numpy(np.load(root/'image_reference.npz')['input']))
 del model
 print('unique_pack_signatures '+str(len(fixtures)),flush=True)
 results=[]
 with open('/root/rngd-model-forge/reports/rngd_porting/npu.lock','a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  with NPUExecutionAuditor(out,'all-pack-signatures') as audit:
   for i,(key,(m,x)) in enumerate(fixtures.items()):
    print('CASE '+str(i)+' '+str(key),flush=True)
    ref=m.pack(x);resident=x.to('furiosa:0')
    with audit.span('model_component::pack_'+str(i)):
     actual=compiled_pack(m,resident).cpu()
    torch.testing.assert_close(actual,ref,rtol=0,atol=0)
    results.append({'signature':key,'exact':True});(out/'results.json').write_text(json.dumps(results,indent=2))
  summary=json.loads((out/'summary.json').read_text());assert summary['profiler']['npu_task_events'] and not summary['cpu']['sdk_cpu_execution_count']
 print('PASSED '+str(len(results))+' '+str(out),flush=True)

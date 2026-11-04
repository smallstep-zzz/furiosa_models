import os,sys,time,json,fcntl
from pathlib import Path
ROOT=Path(__file__).resolve().parent
from yolo11_optimized import configure,make_model,prepare_image
configure()
import torch,numpy as np
out=ROOT/'runs'/('layout_ab_'+str(time.time_ns()));out.mkdir(parents=True)
lock=open('/root/rngd-model-forge/reports/rngd_porting/npu.lock','a');fcntl.flock(lock,fcntl.LOCK_EX)
with torch.no_grad():
 model=make_model(ROOT/'models/yolo11n_640.onnx',ROOT/'static_metadata.json')
 _,data=prepare_image(ROOT/'bus.jpg'); x=torch.from_numpy(data);xc=x.contiguous()
 values={'web_strided':x.to('furiosa:0'),'nchw_contiguous':xc.to('furiosa:0')}
 outputs={k:model(v).cpu() for k,v in values.items()}
 torch.testing.assert_close(outputs['web_strided'],outputs['nchw_contiguous'],rtol=0,atol=0)
 times={k:[] for k in values}
 for i in range(5):
  for k in (list(values) if i%2==0 else list(reversed(values))):
   t=time.perf_counter();actual=model(values[k]).cpu();times[k].append(time.perf_counter()-t)
   torch.testing.assert_close(actual,outputs[k],rtol=0,atol=0)
 report={'timing_s':times,'strides':{k:list(v.stride()) for k,v in values.items()},'output_bitwise_identical':True}
 (out/'result.json').write_text(json.dumps(report,indent=2));print('RESULT '+str(out));print(json.dumps(report))

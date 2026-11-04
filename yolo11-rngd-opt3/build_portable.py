"""Capture validated model and build a standalone FXBEX with bundled Python."""
import sys,os,json,time,fcntl,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent;SOURCE=ROOT/'portable_source'
sys.path.insert(0,str(SOURCE/'python'))
from yolo11_portable.adapter import YoloAdapter,configure,make_model,sources,sha,MODEL_FILE,REVISION
configure()
import torch,numpy as np
from fxbex.replay import capture
from fxbex.package import build
from fxbex.runtime import environment
report=ROOT/'runs'/('portable_capture_'+str(time.time_ns()));report.mkdir(parents=True)
with open('/root/rngd-model-forge/reports/rngd_porting/npu.lock','a') as lock,torch.no_grad():
 fcntl.flock(lock,fcntl.LOCK_EX)
 with capture(SOURCE) as stats:
  adapter=YoloAdapter.__new__(YoloAdapter);adapter.report=report;adapter.stats=stats
  adapter.model=make_model(ROOT/'models'/MODEL_FILE,SOURCE/'static_metadata.json')
  result=adapter.predict({'image':str(ROOT/'bus.jpg')})
 actual=np.load(Path(result['output'])/'output.npy');ref=np.load(ROOT/'image_reference.npz')['output']
 np.testing.assert_allclose(actual[:,:4],ref[:,:4],rtol=1e-3,atol=.1);np.testing.assert_allclose(actual[:,4:],ref[:,4:],rtol=1e-2,atol=.002)
 summary=json.loads((Path(result['output'])/'summary.json').read_text());assert summary['counts']['unknown_graphs']==0 and summary['counts']['npu_trace_confirmed_graphs']==89 and summary['cpu']['sdk_cpu_execution_count']==0
 def write(name,value):(SOURCE/name).write_text(json.dumps(value,indent=2))
 write('runtime_config.json',{'project_sources':sources(),'input_shape':[1,3,640,640],'task':'predict','precision':'float32 I/O; compensated BF16 products with FP32 sums for spatial conv','compiler_policy':'recorded-runnables-only'})
 write('weights_index.json',{'weights':[{'name':MODEL_FILE,'external_path':str(ROOT/'models'/MODEL_FILE),'sha256':sha(ROOT/'models'/MODEL_FILE)}]})
 write('memory_plan.json',{'allocator':'SDK','fusion':8,'mesh':'Single','weights_preparation':'FP32 ONNX -> static high/low components on CPU once; model arithmetic remains NPU'})
 versions=environment();versions={k:v for k,v in versions.items() if k in ['furiosa-torch','furiosa-tcc','torch']}
 write('manifest.json',{'model':{'name':'giangndm/yolo11-onnx','revision':REVISION,'variant':'nano640','modality':'image-to-detections','license':'AGPL-3.0'},'compatibility':{'packages':versions,'hardware':'rngd','fusion':8},'runtime':{'adapter':'yolo11n640_portable','bundled_python':{'root':'python','packages':['compiler_engine','yolo11_portable'],'entrypoint':'yolo11_portable.adapter:YoloAdapter'}},'delivery':{'verified_fxbex':'0.4.0','includes':['ONNX weights','SDK serialized Runnables','Python model adapter and required Forge runtime','class labels','static shape metadata','model card'],'requires_host':['RNGD driver','matching Furiosa SDK/PyTorch','FXBEX0.4.0','onnx1.17.0','onnx2torch1.5.15','torchvision','numpy','Pillow'],'relocation_inference':'pending independent gate'}})
 shutil.copyfile(ROOT/'UPSTREAM_MODEL_CARD.md',SOURCE/'UPSTREAM_MODEL_CARD.md')
 (SOURCE/'USAGE.txt').write_text('Use a compatible RNGD SDK host with FXBEX 0.4.0. No original model project or external ONNX required.\nfrom fxbex import YOLO\nwith YOLO("model.fxbex") as model:\n    result=model.predict("image.jpg")\n\nInput is resized/letterboxed to640. Only nano640/batch1 verified. Original model AGPL-3.0; see UPSTREAM_MODEL_CARD.md.\n')
 package=ROOT/'fxbex/yolo11n-fp32comp-opt3-640-fusion8-sdk2026.4.0-portable.fxbex';build(SOURCE,package,standalone=True)
 result.update(package=str(package),sha256=sha(package),bytes=package.stat().st_size,standalone=True,bundled_runtime=True,saved_runnables=len(json.loads((SOURCE/'execution_plan.json').read_text())['steps']))
 (report/'result.json').write_text(json.dumps(result,indent=2));(ROOT/'portable_capture_result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

import os,sys,time,json,fcntl
from pathlib import Path
os.environ.setdefault('RAYON_NUM_THREADS','16');os.environ['FURIOSA_REQUIRE_NPU']='1';sys.path.insert(0,'/root/rngd-model-forge')
import torch
from furiosa.torch import CompileModule
from furiosa.torch.config import CompilerConfig,set_compiler_config,set_cpu_fallback
from compiler_engine.renegade_diffusion_compiler import _configure_fused_pes
from compiler_engine.execution_auditor import NPUExecutionAuditor
from pool_adapter import ResidentMaxPool
class Separable(torch.nn.Module):
 def forward(self,value):
  h,w=value.shape[-2:];p=torch.cat((value[...,:1],value[...,:1],value,value[...,-1:],value[...,-1:]),dim=-1)
  z=p[...,:w].contiguous()
  for i in range(1,5):z=torch.maximum(z,p[...,i:i+w].contiguous())
  p=torch.cat((z[...,:1,:],z[...,:1,:],z,z[...,-1:,:],z[...,-1:,:]),dim=-2)
  z=p[...,:h,:].contiguous()
  for i in range(1,5):z=torch.maximum(z,p[...,i:i+h,:].contiguous())
  return z
root=Path(__file__).resolve().parent;out=root/'runs'/('pool_'+str(time.time_ns()));out.mkdir(parents=True);torch.set_num_threads(8)
with open('/root/rngd-model-forge/reports/rngd_porting/npu.lock','a') as lock,torch.no_grad():
 fcntl.flock(lock,fcntl.LOCK_EX);_configure_fused_pes();set_cpu_fallback(False);cfg=CompilerConfig(implicit_type_casting=False,implicit_type_casting_all_activation=False);set_compiler_config(cfg)
 torch.manual_seed(32);x=-torch.rand(1,128,20,20);ref=torch.nn.functional.max_pool2d(x,5,1,2);m=Separable();torch.testing.assert_close(m(x),ref,rtol=0,atol=0)
 ep=torch.export.export(m,(x,));exe=CompileModule.from_exported(ep,compiler_config=cfg).to('furiosa:0');r=x.to('furiosa:0');exe(r).cpu()
 with NPUExecutionAuditor(out,'pool-fused'):
  actual=exe(r).cpu()
 torch.testing.assert_close(actual,ref,rtol=0,atol=0)
 old=ResidentMaxPool(torch.nn.MaxPool2d(5,1,2));times={}
 for name,fn in [('old',old),('separable',m),('compiled',exe)]:
  fn(r).cpu();t=[]
  for _ in range(10):
   start=time.perf_counter();y=fn(r).cpu();t.append(time.perf_counter()-start);torch.testing.assert_close(y,ref,rtol=0,atol=0)
  times[name]=t
 summary=json.loads((out/'summary.json').read_text());assert summary['profiler']['npu_task_events'] and not summary['cpu']['sdk_cpu_execution_count'];result={'times':times,'exact':True,'tasks':summary['profiler']['npu_task_events'],'path':str(out)};(out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

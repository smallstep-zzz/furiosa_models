import os,sys,time,json,fcntl
from pathlib import Path
os.environ.setdefault('RAYON_NUM_THREADS','16');os.environ['FURIOSA_REQUIRE_NPU']='1';sys.path.insert(0,'/root/rngd-model-forge')
import torch
from furiosa.torch.config import CompilerConfig,set_compiler_config,set_cpu_fallback
from compiler_engine.renegade_diffusion_compiler import _configure_fused_pes
from compiler_engine.execution_auditor import NPUExecutionAuditor
from tcl_conv_candidate import fused_conv,prepare_weights
root=Path(__file__).resolve().parent;out=root/'runs'/('tcl_compensated_'+str(time.time_ns()));out.mkdir(parents=True);torch.set_num_threads(8)
with open('/root/rngd-model-forge/reports/rngd_porting/npu.lock','a') as lock,torch.no_grad():
 fcntl.flock(lock,fcntl.LOCK_EX);_configure_fused_pes();set_cpu_fallback(False);set_compiler_config(CompilerConfig(implicit_type_casting=False,implicit_type_casting_all_activation=False))
 torch.manual_seed(97);x=torch.randn(1,16,320,320);w=torch.randn(32,16,3,3)*.1;b=torch.randn(32)*.1;ref=torch.nn.functional.conv2d(x,w,b,stride=2,padding=1);hi,lo=prepare_weights(w);args=tuple(t.to('furiosa:0') for t in (x,hi,lo,b))
 fused_conv(*args).cpu()
 with NPUExecutionAuditor(out,'tcl-conv-hardware'):
  actual=fused_conv(*args).cpu()
 times=[]
 for _ in range(10):
  t=time.perf_counter();fused_conv(*args).cpu();times.append(time.perf_counter()-t)
 result={'warm_s':times,'max_abs':float((actual-ref).abs().max()),'allclose':bool(torch.allclose(actual,ref,rtol=1e-3,atol=1e-3))};(out/'result.json').write_text(json.dumps(result));print(result,flush=True);torch.testing.assert_close(actual,ref,rtol=1e-3,atol=1e-3)

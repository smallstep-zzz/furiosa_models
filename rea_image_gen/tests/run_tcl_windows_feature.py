import os,json,time
from pathlib import Path
import torch
from compiler_engine.renegade_diffusion_compiler import _configure_fused_pes
_configure_fused_pes()
from compiler_engine.tcl_windows import pack_windows
from compiler_engine.execution_auditor import NPUExecutionAuditor
out=Path(os.environ.get('FURIOSA_REPRO_DIR','reports/rngd_porting/runs/tcl_windows_tiny'))
out.mkdir(parents=True,exist_ok=True)
n,c,h,w=map(int,os.environ.get('WINDOW_SHAPE','1,4,4,4').split(','))
k=int(os.environ.get('WINDOW_KERNEL','3'));pad=k//2
torch.manual_seed(20261102)
x=torch.randn(n,c,h,w,dtype=torch.bfloat16)
p=torch.nn.functional.pad(x,(pad,pad,pad,pad))
ref=torch.stack([p[:,:,kh:kh+h,kw:kw+w].permute(0,2,3,1) for kh in range(k) for kw in range(k)],dim=-2).reshape(n,h,w,k,k,c)
a=x.to('furiosa:0');start=time.perf_counter()
with NPUExecutionAuditor(out,'tcl-windows') as audit:
    with audit.span('model_component::tcl_windows'):
        y=pack_windows(a,k,pad)
actual=y.cpu();torch.testing.assert_close(actual,ref,rtol=0,atol=0)
s=json.loads((out/'summary.json').read_text());r={'exact_match':True,'device':str(y.device),'tasks':s['profiler']['npu_task_events'],'elapsed_s':time.perf_counter()-start}
(out/'result.json').write_text(json.dumps(r));print(json.dumps(r));assert r['tasks']>0

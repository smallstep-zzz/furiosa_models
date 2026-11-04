import json, os
from pathlib import Path
os.environ["FURIOSA_SDPA_FP32"] = "1"
import torch
from compiler_engine.renegade_diffusion_compiler import _configure_fused_pes
from compiler_engine.graph_lowerings import manual_scaled_dot_product_attention
from compiler_engine.execution_auditor import NPUExecutionAuditor
_configure_fused_pes()
torch.manual_seed(20261102)
out=Path(os.environ['FURIOSA_REPRO_DIR']);out.mkdir(parents=True,exist_ok=True)
q=torch.randn(1,10,256,64,dtype=torch.bfloat16)
k=torch.randn(1,10,77,64,dtype=torch.bfloat16)
v=torch.randn_like(k)
expected=torch.nn.functional.scaled_dot_product_attention(q,k,v)
cpu=manual_scaled_dot_product_attention(q,k,v)
with NPUExecutionAuditor(out,'sdpa-fp32-intermediates') as audit:
    with audit.span('model_component::sdpa_precision'):
        got=manual_scaled_dot_product_attention(q.to('furiosa:0'),k.to('furiosa:0'),v.to('furiosa:0'))
got=got.cpu()
d=(got.float()-expected.float()).abs()
result={'cpu_decomposition_max_abs':float((cpu.float()-expected.float()).abs().max()),'npu_max_abs':float(d.max()),'mismatches':int((d>0.005+0.005*expected.float().abs()).sum()),'dtype':str(got.dtype)}
(out/'precision.json').write_text(json.dumps(result,indent=2));print(result)
assert result['mismatches']==0

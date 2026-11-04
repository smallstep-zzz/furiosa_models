"""Extract only FLUX block0 AdaLN weights using shard index; no full model load."""
import os,json,time
from pathlib import Path
import torch
from safetensors import safe_open
from diffusers.models.normalization import AdaLayerNormZero
from compiler_engine.renegade_diffusion_compiler import _NPUBlock,_configure_fused_pes
from compiler_engine.execution_auditor import NPUExecutionAuditor
root=Path('/root/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21/transformer')
out=Path(os.environ.get('FURIOSA_REPRO_DIR','reports/rngd_porting/runs/flux_norm0'));out.mkdir(parents=True,exist_ok=True)
os.environ['FURIOSA_GRAPH_AUDIT_PATH']=str(out/'compiler_graphs.jsonl')
index=json.loads((root/'diffusion_pytorch_model.safetensors.index.json').read_text())['weight_map']
prefix='transformer_blocks.0.norm1.'
keys=[k for k in index if k.startswith(prefix)]
module=AdaLayerNormZero(3072).to(dtype=torch.bfloat16).eval()
state={}
for key in keys:
 with safe_open(root/index[key],framework='pt',device='cpu') as shard:state[key[len(prefix):]]=shard.get_tensor(key)
module.load_state_dict(state,strict=True)
torch.manual_seed(20261102)
hidden=torch.randn(1,1024,3072,dtype=torch.bfloat16);temb=torch.randn(1,3072,dtype=torch.bfloat16)
with torch.inference_mode(): expected=module(hidden,emb=temb)
(out/'extraction.json').write_text(json.dumps({'source_revision':root.parent.name,'prefix':prefix,'weight_tensors':len(state),'weight_bytes':sum(t.numel()*t.element_size() for t in state.values()),'shape':list(hidden.shape),'dtype':str(hidden.dtype)},indent=2))
_configure_fused_pes();module.to('furiosa:0')
if os.environ.get('FURIOSA_FLUX_SEGMENT_NORM') == '1':
 from compiler_engine.flux_porting import segment_adaptive_norm
 compiled=segment_adaptive_norm(module,'transformer.block0.norm1')
else: compiled=_NPUBlock(module,'flux','transformer.block0.norm1')
with NPUExecutionAuditor(out,'flux-norm-feature') as auditor:
 with auditor.span('model_component::flux.norm1'):
  with torch.inference_mode():got=compiled(hidden.to('furiosa:0'),emb=temb.to('furiosa:0'))
checks=[]
for a,b in zip(got,expected):
 a=a.cpu().float();b=b.float();d=(a-b).abs();checks.append({'shape':list(a.shape),'max_abs':float(d.max()),'mismatches':int((d>0.05+0.05*b.abs()).sum())})
summary=json.loads((out/'summary.json').read_text())
result={'checks':checks,'counts':summary['counts'],'tasks':summary['profiler']['npu_task_events'],'status':'pass' if all(c['mismatches']==0 for c in checks) and summary['profiler']['npu_task_events']>0 else 'failed'}
(out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result));assert result['status']=='pass'

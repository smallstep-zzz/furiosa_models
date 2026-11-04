"""Download component-format official Klein4B weights; never load/scan model binaries."""
from pathlib import Path
import json,time
from huggingface_hub import snapshot_download
out=Path('reports/rngd_porting/flux_klein_preparation.json');out.parent.mkdir(parents=True,exist_ok=True)
repo='black-forest-labs/FLUX.2-klein-4B';start=time.perf_counter()
p=Path('/root/.local/share/huggingface/token');token=p.read_text().strip() if p.exists() else None
out.write_text(json.dumps({'model':repo,'status':'downloading_components','npu_verified':False}))
try:
 path=snapshot_download(repo,token=token,allow_patterns=['model_index.json','scheduler/*','tokenizer/*','text_encoder/*','transformer/*','vae/*'],ignore_patterns=['*.bin','*.msgpack','*.onnx','*.h5'],max_workers=8)
 result={'model':repo,'status':'cached_pending_feature_tests','snapshot':path,'elapsed_s':time.perf_counter()-start,'npu_verified':False}
except Exception as e:
 result={'model':repo,'status':'download_failed','error_type':type(e).__name__,'elapsed_s':time.perf_counter()-start,'npu_verified':False}
out.write_text(json.dumps(result,indent=2));print(json.dumps(result));raise SystemExit(0 if result['status']=='cached_pending_feature_tests' else 1)

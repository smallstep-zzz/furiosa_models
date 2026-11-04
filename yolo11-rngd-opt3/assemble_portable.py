"""Prepare a self-contained Python runtime tree; execute only after feature gates."""
from pathlib import Path
import re,shutil
ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'portable_source'
PYROOT=SOURCE/'python';PKG=PYROOT/'yolo11_portable';PKG.mkdir(parents=True,exist_ok=False)
(PKG/'__init__.py').write_text('"""Pinned portable YOLO11n RNGD runtime."""\n')
modules=['conv_adapter','pool_adapter','resident_model','fast_pack','static_adapter','fused_pool','fused_model','tcl_conv_candidate','pointwise_model','first_layout','direct_route','activation_pointwise','activation_tcl','activation_fused','activation_route']
for name in modules:
 s=(ROOT/(name+'.py')).read_text()
 for other in modules+['image_demo']:
  s=s.replace('from '+other+' import','from .'+other+' import')
 (PKG/(name+'.py')).write_text(s)
s=(ROOT/'image_demo.py').read_text();start=s.index('def detections(');end=s.index("if __name__",start)
(PKG/'image_demo.py').write_text('import torch\nfrom torchvision.ops import batched_nms\n\n'+s[start:end])
shutil.copyfile(ROOT/'models/classes.txt',PKG/'classes.txt')
engine=PYROOT/'compiler_engine';engine.mkdir()
for p in Path('/root/rngd-model-forge/compiler_engine').glob('*.py'):
 if not p.name.startswith('.'):
  shutil.copyfile(p,engine/p.name)
s=(ROOT/'yolo11_optimized.py').read_text().replace('sys.path.insert(0, str(ROOT))','').replace("    sys.path.insert(0, '/root/rngd-model-forge')",'')
start=s.index('def sources():');end=s.index('\n\ndef configure():',start)
s=s[:start]+'''def sources():
    base=ROOT.parent
    files=list(ROOT.glob('*.py'))+list((base/'compiler_engine').glob('*.py'))
    return {str(p.relative_to(base)):sha(p) for p in sorted(files)}
'''+s[end:]
for name in modules+['image_demo']:
 s=s.replace('from '+name+' import','from .'+name+' import')
s=s.replace("weight = Path(weights_dir)/MODEL_FILE if weights_dir else Path(index[0]['external_path'])", "weight = root/index[0]['path']")
s=s.replace("weights = Path(weights_dir)/MODEL_FILE if weights_dir else Path(index['external_path'])", "weights = Path(root)/index['path']")
s=s.replace("'yolo11n640_optimized'", "'yolo11n640_portable'")
s=s.replace('from .resident_model import adapt', 'from .activation_route import adapt')
s=s.replace('return canvas, np.asarray(canvas, dtype=np.float32).transpose(2,0,1)[None]/255', 'return canvas, np.ascontiguousarray(np.asarray(canvas, dtype=np.float32).transpose(2,0,1)[None]/255)')
s=s.replace("f'{label}: {score:.3f}'", "f'{(ROOT / \"classes.txt\").read_text().splitlines()[label]}: {score:.3f}'")
s=s.replace("canvas.save(output/'detections.png')", "canvas.save(output/'detections.png', compress_level=1)")
s=s.replace('preparation_s = time.perf_counter()-preparing', 'preparation_s = time.perf_counter()-preparing\n                anchor = torch.ones((32,32)).to("furiosa:0")\n                (anchor+1).cpu()')
s=s.replace("with (audit.span('model_component::yolo11n_full') if audit else nullcontext()):", "if audit:\n                    with audit.span('diagnostic::trace_anchor'): (anchor+1).cpu()\n                with (audit.span('model_component::yolo11n_full') if audit else nullcontext()):")
s=s.replace("if not tasks or summary['cpu']['sdk_cpu_execution_count']:", "if (not tasks or summary['cpu']['sdk_cpu_execution_count'] or summary['counts']['unknown_graphs'] or summary['counts']['npu_trace_confirmed_graphs'] != 89):")
(PKG/'adapter.py').write_text(s)
shutil.copyfile(ROOT/'static_metadata.json',SOURCE/'static_metadata.json')
print(SOURCE)

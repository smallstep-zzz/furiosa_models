"""Strict full-model inference through native FP32 resident adapters."""
import argparse
import hashlib
import fcntl
import json
import os
from pathlib import Path
import sys
import time
os.environ.setdefault('RAYON_NUM_THREADS', '16')
os.environ.setdefault('OMP_NUM_THREADS', '8')
os.environ['FURIOSA_REQUIRE_NPU'] = '1'
sys.path.insert(0, '/root/rngd-model-forge')
import numpy as np
import onnx
import torch
from onnx2torch import convert
from static_adapter import specialize
from compiler_engine.execution_auditor import NPUExecutionAuditor, record_function
from compiler_engine.renegade_diffusion_compiler import _configure_fused_pes
from furiosa.torch.config import set_cpu_fallback, get_cpu_fallback, set_compiler_config
from resident_model import adapt, CONFIG
from compiler_engine import porting_progress

ROOT = Path(__file__).resolve().parent
p = argparse.ArgumentParser()
p.add_argument('--output', type=Path, required=True)
p.add_argument('--input', type=Path, default=ROOT / 'image_reference.npz')
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
sources = a.output / 'source'
sources.mkdir()
hashes = {}
for path in ROOT.glob('*.py'):
    content = path.read_bytes()
    (sources / path.name).write_bytes(content)
    hashes[path.name] = hashlib.sha256(content).hexdigest()
(a.output / 'source_hashes.json').write_text(json.dumps(hashes, indent=2))
os.environ['FURIOSA_GRAPH_AUDIT_PATH'] = str(a.output / 'compiler_graphs.jsonl')
porting_progress.PATH = ROOT / 'progress.json'
porting_progress.update('integration', 'running', 'First spatial-linear FP32 convolution passed; whole-model resident execution')
torch.set_num_threads(8)
data = np.load(a.input)
x, ref = torch.from_numpy(data['input']), torch.from_numpy(data['output'])
model = specialize(convert(onnx.load(ROOT / 'models/yolo11n_640.onnx')).eval(), x)
model = adapt(model)
with torch.no_grad():
    lock = open('/root/rngd-model-forge/reports/rngd_porting/npu.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX)
    _configure_fused_pes()
    set_cpu_fallback(False)
    set_compiler_config(CONFIG)
    model = model.to('furiosa:0')
    resident = x.to('furiosa:0')
    class Traced(torch.fx.Interpreter):
        def run_node(self, node):
            print('node '+node.name, flush=True)
            with record_function('model_component::'+node.name):
                result = super().run_node(node)
                tensors = [t for t in torch.utils._pytree.tree_leaves(result) if isinstance(t, torch.Tensor)]
                assert all(t.device.type == 'furiosa' for t in tensors), f'CPU model tensor at {node.name}'
                return result
    start = time.perf_counter()
    with NPUExecutionAuditor(a.output, 'yolo11n-resident'):
        actual = Traced(model).run(resident).cpu()
        first_s = time.perf_counter()-start
    np.save(a.output / 'output.npy', actual.numpy())
    torch.testing.assert_close(actual[:, :4], ref[:, :4], rtol=1e-3, atol=0.1)
    torch.testing.assert_close(actual[:, 4:], ref[:, 4:], rtol=1e-2, atol=0.002)
    summary = json.loads((a.output / 'summary.json').read_text())
    assert summary['profiler']['npu_task_events'] > 0
    assert summary['cpu']['sdk_cpu_execution_count'] == 0
    timings = []
    for _ in range(3):
        start = time.perf_counter()
        y = model(resident).cpu()
        timings.append(time.perf_counter()-start)
        torch.testing.assert_close(y, actual, rtol=0, atol=0)
    report = {'status': 'passed', 'file': 'yolo11n_640.onnx', 'precision': 'float32',
              'first_with_compile_profile_s': first_s, 'warm_s': timings,
              'npu_tasks': summary['profiler']['npu_task_events'], 'sdk_cpu_fallback': get_cpu_fallback(),
              'max_abs_coordinates': float((actual[:, :4]-ref[:, :4]).abs().max()),
              'max_abs_scores': float((actual[:, 4:]-ref[:, 4:]).abs().max())}
    (a.output / 'result.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    porting_progress.update('integration', 'passed', f"{report['npu_tasks']} Tasks; ONNX reference passed")
    porting_progress.update('repeat', 'passed', f'Three identical warm outputs: {timings}')

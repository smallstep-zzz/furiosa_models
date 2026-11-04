"""Whole ONNX graph on RNGD, strict fallback and official execution evidence."""
import argparse
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
from furiosa.torch import CompileModule
from furiosa.torch.config import get_cpu_fallback
from compiler_engine.renegade_diffusion_compiler import _configure_fused_pes, _COMPILER_CONFIG
from compiler_engine.execution_auditor import NPUExecutionAuditor
from compiler_engine import porting_progress

ROOT = Path(__file__).resolve().parent
porting_progress.PATH = ROOT / 'progress.json'
p = argparse.ArgumentParser()
p.add_argument('--output', type=Path, required=True)
p.add_argument('--input', type=Path, default=ROOT / 'cpu_reference.npz')
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
torch.set_num_threads(8)
porting_progress.update('prepare', 'passed', 'Forge 871ea7f; ONNX CPU conversion passed; torch 2.10.0 / SDK 2026.4.0', model='giangndm/yolo11-onnx (n640)')
porting_progress.update('features', 'running', 'Full static ONNX graph, native SDK compile; strict CPU fallback disabled')
data = np.load(a.input)
x = torch.from_numpy(data['input'])
ref = torch.from_numpy(data['output'])
model = convert(onnx.load(ROOT / 'models/yolo11n_640.onnx')).eval()
model = specialize(model, x)
lock = open('/root/rngd-model-forge/reports/rngd_porting/npu.lock', 'a')
fcntl.flock(lock, fcntl.LOCK_EX)
_configure_fused_pes()
assert get_cpu_fallback() is False
with torch.no_grad():
    start = time.perf_counter()
    print('export_start', flush=True)
    ep = torch.export.export(model, (x,))
    (a.output / 'export_graph.txt').write_text(str(ep.graph))
    print('compile_start', flush=True)
    compiled = CompileModule.from_module(ep.module(), (x,), compiler_config=_COMPILER_CONFIG, task_name='yolo11n_640_full')
    compile_s = time.perf_counter()-start
    print(json.dumps({'compile_s': compile_s, 'phase': 'prepare'}), flush=True)
    start = time.perf_counter()
    compiled = compiled.to('furiosa:0')
    resident_x = x.to('furiosa:0')
    prepare_s = time.perf_counter()-start
    with NPUExecutionAuditor(a.output / 'audit', 'yolo11n-640') as audit:
        with audit.span('model_component::yolo11n_640_full'):
            start = time.perf_counter()
            y = compiled(resident_x)
            if isinstance(y, (tuple, list)):
                y = y[0]
            assert y.device.type == 'furiosa'
            actual = y.cpu()
            first_s = time.perf_counter()-start
    summary = json.loads((a.output / 'audit/summary.json').read_text())
    torch.testing.assert_close(actual[:, :4], ref[:, :4], rtol=1e-3, atol=0.1)
    torch.testing.assert_close(actual[:, 4:], ref[:, 4:], rtol=1e-2, atol=0.002)
    tasks = summary['profiler']['npu_task_events']
    assert tasks > 0, 'Missing official NPU Tasks'
    assert summary['cpu'].get('sdk_cpu_execution_count', 0) == 0, 'SDK CPU execution observed'
    timings = []
    for _ in range(3):
        start = time.perf_counter()
        repeat = compiled(resident_x)
        if isinstance(repeat, (tuple, list)):
            repeat = repeat[0]
        repeat = repeat.cpu()
        timings.append(time.perf_counter()-start)
        torch.testing.assert_close(repeat, actual, rtol=0, atol=0)
    np.save(a.output / 'output.npy', actual.numpy())
    result = {'status': 'passed', 'model': 'giangndm/yolo11-onnx', 'file': 'yolo11n_640.onnx',
              'precision': 'float32', 'input': list(x.shape), 'output': list(actual.shape),
              'compile_s': compile_s, 'prepare_s': prepare_s, 'first_profiled_s': first_s,
              'warm_s': timings, 'npu_tasks': tasks, 'sdk_cpu_fallback': get_cpu_fallback(),
              'max_abs_coordinates': float((actual[:, :4]-ref[:, :4]).abs().max()),
              'max_abs_scores': float((actual[:, 4:]-ref[:, 4:]).abs().max()),
              'tolerances': {'coordinates': {'rtol': 0.001, 'atol': 0.1}, 'scores': {'rtol': 0.01, 'atol': 0.002}}}
    (a.output / 'result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    porting_progress.update('features', 'passed', f'{tasks} official NPU Tasks; ONNX Runtime comparison passed')
    porting_progress.update('repeat', 'passed', f'Three deterministic warm forwards: {timings}')

"""Real first convolution, isolated native-SDK route and channel-padding probe."""
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
from furiosa.torch import CompileModule
from furiosa.torch.config import CompilerConfig, TacticHintConfig
from compiler_engine.renegade_diffusion_compiler import _configure_fused_pes
from compiler_engine.execution_auditor import NPUExecutionAuditor

ROOT = Path(__file__).resolve().parent
p = argparse.ArgumentParser()
p.add_argument('--route', choices=['exact-fp32', 'padded-fp32', 'padded-default', 'native-autocast'], required=True)
a = p.parse_args()
out = ROOT / 'runs' / ('conv_'+a.route+'_'+str(time.time_ns()))
out.mkdir(parents=True, exist_ok=False)
torch.set_num_threads(8)
model = convert(onnx.load(ROOT / 'models/yolo11n_640.onnx')).eval()
conv = next(m for m in model.modules() if isinstance(m, torch.nn.Conv2d))
x = torch.from_numpy(np.load(ROOT / 'cpu_reference.npz')['input'])
with torch.no_grad():
    ref = conv(x)
if a.route.startswith('padded'):
    class PaddedConv(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(16, conv.out_channels, 3, stride=2, padding=1)
            with torch.no_grad():
                self.conv.weight.zero_()
                self.conv.weight[:, :3].copy_(conv.weight)
                self.conv.bias.copy_(conv.bias)
        def forward(self, value):
            return self.conv(torch.nn.functional.pad(value, (0, 0, 0, 0, 0, 13)))
    test = PaddedConv().eval()
else:
    class ExactConv(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = conv
        def forward(self, value):
            return self.conv(value)
    test = ExactConv().eval()
config = CompilerConfig(tactic_hint=TacticHintConfig.ForVisionModel,
                        implicit_type_casting=(a.route in ('padded-default', 'native-autocast')),
                        implicit_type_casting_all_activation=(a.route == 'native-autocast'))
with torch.no_grad():
    torch.testing.assert_close(test(x), ref, rtol=1e-5, atol=1e-5)
    lock = open('/root/rngd-model-forge/reports/rngd_porting/npu.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX)
    _configure_fused_pes()
    start = time.perf_counter()
    print('compile_start '+a.route, flush=True)
    compiled = CompileModule.from_module(test, (x,), compiler_config=config).to('furiosa:0')
    print(json.dumps({'compile_prepare_s': time.perf_counter()-start}), flush=True)
    resident_x = x.to('furiosa:0')
    with NPUExecutionAuditor(out / 'audit', 'yolo11-first-conv') as audit:
        with audit.span('model_component::first_conv'):
            result = compiled(resident_x).cpu()
    torch.testing.assert_close(result, ref, rtol=1e-3, atol=1e-3)
    summary = json.loads((out / 'audit/summary.json').read_text())
    tasks = summary['profiler']['npu_task_events']
    assert tasks > 0
    report = {'route': a.route, 'status': 'passed', 'npu_tasks': tasks,
              'max_abs': float((result-ref).abs().max())}
    (out / 'result.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)

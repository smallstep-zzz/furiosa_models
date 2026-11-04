"""Exact failing second-convolution FP32 layout, no full-model retry."""
import os
os.environ['FURIOSA_REQUIRE_NPU'] = '1'
os.environ.setdefault('RAYON_NUM_THREADS', '8')
import sys
sys.path.insert(0, '/root/rngd-model-forge')
import fcntl
import json
from pathlib import Path
import time
import torch
from conv_adapter import SpatialLinearConv2d
from compiler_engine.renegade_diffusion_compiler import _configure_fused_pes
from compiler_engine.execution_auditor import NPUExecutionAuditor

torch.set_num_threads(4)
root = Path(__file__).resolve().parent
out = root / 'runs' / ('pack_'+str(time.time_ns()))
out.mkdir(parents=True)
with open('/root/rngd-model-forge/reports/rngd_porting/npu.lock', 'a') as lock, torch.no_grad():
    fcntl.flock(lock, fcntl.LOCK_EX)
    _configure_fused_pes()
    torch.manual_seed(12)
    x = torch.randn(1,16,320,320)
    adapter = SpatialLinearConv2d(torch.nn.Conv2d(16,32,3,stride=2,padding=1))
    reference = adapter.pack(x)
    resident = x.to('furiosa:0')
    with NPUExecutionAuditor(out, 'second-conv-pack') as audit:
        with audit.span('model_component::pack'):
            actual = adapter.pack(resident).cpu()
    torch.testing.assert_close(actual,reference,rtol=0,atol=0)
    summary = json.loads((out/'summary.json').read_text())
    result = {'status':'passed','exact':True,'npu_tasks':summary['profiler']['npu_task_events']}
    assert result['npu_tasks']>0 and summary['cpu']['sdk_cpu_execution_count']==0
    (out/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result))

"""Exact first 80-channel detection head, padded native contraction on RNGD."""
import os
os.environ['FURIOSA_REQUIRE_NPU'] = '1'
os.environ.setdefault('RAYON_NUM_THREADS', '8')
import sys
sys.path.insert(0, '/root/rngd-model-forge')
import fcntl
import json
from pathlib import Path
import time
import onnx
from onnx2torch import convert
import torch
from resident_model import ResidentConv, CONFIG
from compiler_engine.renegade_diffusion_compiler import _configure_fused_pes
from compiler_engine.execution_auditor import NPUExecutionAuditor
from furiosa.torch.config import set_compiler_config

torch.set_num_threads(4)
root = Path(__file__).resolve().parent
out = root / 'runs' / ('head_'+str(time.time_ns()))
out.mkdir(parents=True)
with open('/root/rngd-model-forge/reports/rngd_porting/npu.lock', 'a') as lock, torch.no_grad():
    fcntl.flock(lock, fcntl.LOCK_EX)
    _configure_fused_pes()
    set_compiler_config(CONFIG)
    model = convert(onnx.load(root/'models/yolo11n_640.onnx')).eval()
    conv = next(c for c in model.modules() if isinstance(c, torch.nn.Conv2d) and c.in_channels==64 and c.out_channels==80 and c.kernel_size==(1,1))
    torch.manual_seed(12)
    x = torch.randn(1,64,80,80)
    reference = conv(x)
    adapter = ResidentConv(conv, 'head80').to('furiosa:0')
    resident = x.to('furiosa:0')
    with NPUExecutionAuditor(out, 'yolo-head80'):
        actual = adapter(resident).cpu()
    torch.testing.assert_close(actual,reference,rtol=1e-3,atol=1e-3)
    summary = json.loads((out/'summary.json').read_text())
    result = {'status':'passed','max_abs':float((actual-reference).abs().max()),'npu_tasks':summary['profiler']['npu_task_events']}
    assert result['npu_tasks']>0 and summary['cpu']['sdk_cpu_execution_count']==0
    (out/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result))

"""Real SPPF MaxPool shape, negative-border self-check and NPU trace."""
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
from pool_adapter import ResidentMaxPool
from compiler_engine.renegade_diffusion_compiler import _configure_fused_pes
from compiler_engine.execution_auditor import NPUExecutionAuditor
from furiosa.torch.config import CompilerConfig, set_compiler_config

torch.set_num_threads(4)
root = Path(__file__).resolve().parent
out = root / 'runs' / ('pool_'+str(time.time_ns()))
out.mkdir(parents=True)
with open('/root/rngd-model-forge/reports/rngd_porting/npu.lock', 'a') as lock, torch.no_grad():
    fcntl.flock(lock, fcntl.LOCK_EX)
    _configure_fused_pes()
    set_compiler_config(CompilerConfig(implicit_type_casting=False, implicit_type_casting_all_activation=False))
    torch.manual_seed(12)
    x = -torch.rand(1,128,20,20)
    original = torch.nn.MaxPool2d(5, stride=1, padding=2)
    adapter = ResidentMaxPool(original)
    reference = original(x)
    torch.testing.assert_close(adapter(x), reference, rtol=0, atol=0)
    resident = x.to('furiosa:0')
    with NPUExecutionAuditor(out, 'yolo-sppf-maxpool') as audit:
        with audit.span('model_component::maxpool'):
            actual = adapter(resident).cpu()
    torch.save({'actual': actual, 'reference': reference}, out / 'comparison.pt')
    torch.testing.assert_close(actual,reference,rtol=0,atol=0)
    summary = json.loads((out/'summary.json').read_text())
    result = {'status':'passed','exact':True,'npu_tasks':summary['profiler']['npu_task_events']}
    assert result['npu_tasks']>0 and summary['cpu']['sdk_cpu_execution_count']==0
    (out/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result))

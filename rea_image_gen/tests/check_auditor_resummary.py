import json,time,shutil
from pathlib import Path
from compiler_engine.execution_auditor import NPUExecutionAuditor
p=Path('reports/rngd_porting/runs/sdxl_turbo_image_r11')
old=json.loads((p/'summary.json').read_text());shutil.copy2(p/'summary.json',p/'summary_before_sweep.json')
t=time.perf_counter();NPUExecutionAuditor.summarize_existing(p,'sdxl');elapsed=time.perf_counter()-t
new=json.loads((p/'summary.json').read_text())
assert new['counts']==old['counts']
assert new['cpu']['attributed_aten_cpu_ops']==old['cpu']['attributed_aten_cpu_ops']
r={'elapsed_s':elapsed,'counts_equal':True,'cpu_attribution_equal':True}
(p/'resummary_regression.json').write_text(json.dumps(r));print(json.dumps(r))

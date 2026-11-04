"""Run a porting task and append measured work history without invented counts."""
import argparse,json,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--kind',choices=['test','revise','create','reuse'],required=True);p.add_argument('--feature',required=True);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
cmd=a.command[1:] if a.command[:1]==['--'] else a.command
if not cmd:p.error('a command is required')
out=Path('reports/rngd_porting/work_history');out.mkdir(parents=True,exist_ok=True)
stamp=time.time_ns();log=out/f'{stamp}.log';started=time.perf_counter()
with log.open('w') as f:run=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
row={'timestamp':datetime.now(timezone.utc).isoformat(),'kind':a.kind,'feature':a.feature,'elapsed_s':time.perf_counter()-started,'returncode':run.returncode,'log_path':str(log.resolve())}
with (out/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
print(json.dumps(row,ensure_ascii=False));raise SystemExit(run.returncode)

"""Same-window interval unions; never add nested/overlapping categories."""
import json,sys,collections
from pathlib import Path

def union(spans):
 merged=[]
 for start,end in sorted(spans):
  if merged and start<=merged[-1][1]:merged[-1][1]=max(merged[-1][1],end)
  else:merged.append([start,end])
 return sum(b-a for a,b in merged)

p=Path(sys.argv[1]);events=json.loads((p/'furiosa_trace.json').read_text())['traceEvents']
spans=[e for e in events if e.get('ph')=='X' and e.get('name','').startswith('pipeline_inference::')]
# Integration trace contains a diagnostic anchor plus the model; prefer explicit model span.
if not spans:raise SystemExit('Explicit pipeline_inference scope required; do not infer an unrelated time window')
span=max(spans,key=lambda e:e['dur']);start=span['ts'];end=start+span['dur'];groups=collections.defaultdict(list)
for e in events:
 if e.get('ph')!='X' or not e.get('dur',0):continue
 a,b=max(start,e['ts']),min(end,e['ts']+e['dur'])
 if a>=b:continue
 name=e.get('name','');groups[name].append((a,b))
 for category,trigger in [('all_tasks','::Task::'),('all_d2d','::d2d::'),('all_d2h','::d2h::')]:
  if trigger in name:groups[category].append((a,b))
result={'window_ms':span['dur']/1000,'scope':'profiled; not unprofiled latency','categories_overlap':True,'rows':sorted([{'name':name,'union_ms':union(v)/1000,'count':len(v)} for name,v in groups.items()],key=lambda r:-r['union_ms'])}
(p/'interval_unions.json').write_text(json.dumps(result,indent=2));print(json.dumps({'window_ms':result['window_ms'],'top':result['rows'][:15]}))

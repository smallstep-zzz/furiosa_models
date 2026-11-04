import json,collections
from pathlib import Path
p=Path('reports/rngd_porting/runs/sdxl_turbo_image_r11')
d=json.loads((p/'torch_cpu_trace.json').read_text());events=d['traceEvents'] if isinstance(d,dict) else d
cost=collections.defaultdict(lambda:[0,0.0]);large=[]
for e in events:
 if e.get('ph')!='X':continue
 name=e.get('name','');dur=e.get('dur',0)/1e6
 cost[name][0]+=1;cost[name][1]+=dur
 if dur>2:large.append({'name':name,'seconds':dur,'ts':e.get('ts'),'tid':e.get('tid')})
r={'inclusive_times_not_additive':True,'top_names':sorted(cost.items(),key=lambda x:x[1][1],reverse=True)[:35],'long_spans':sorted(large,key=lambda x:x['seconds'],reverse=True)[:30]}
(p/'phase_costs.json').write_text(json.dumps(r,indent=2));print(json.dumps(r)[:12000])

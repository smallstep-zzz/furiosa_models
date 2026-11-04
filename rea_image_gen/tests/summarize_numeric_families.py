"""Group observations by implementation family; never count elements as bugs."""
import argparse,json
from collections import defaultdict
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('input',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
groups=defaultdict(list)
for row in json.loads(a.input.read_text()):
    groups[row['stage'].split('.')[0]].append(row)
result={'confirmed_root_cause_count':None,'note':'Families are investigation groups, not proven independent bugs. Element violations and repeated observations must not be summed as bug counts.','families':[]}
for name,rows in groups.items():
    failed=[r for r in rows if r['npu_vs_cpu_bf16']['mismatches']]
    result['families'].append({'family':name,'stage_observations':len(rows),'failing_stage_observations':len(failed),'first_failure':failed[0] if failed else None,'status':'needs_operator_bisection' if failed else 'passed_observed_stages','observations':rows})
a.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({r['family']:r['failing_stage_observations'] for r in result['families']}))

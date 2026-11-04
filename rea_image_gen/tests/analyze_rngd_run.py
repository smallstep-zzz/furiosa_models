"""Summarize existing traces without rerunning a model or summing nested durations."""
import argparse, json, re
from pathlib import Path

def union_us(events):
    intervals=sorted((e['ts'], e['ts']+e.get('dur',0)) for e in events if 'ts' in e)
    total=0; end=float('-inf')
    for start,stop in intervals:
        total+=max(0,stop-max(start,end)); end=max(end,stop)
    return total

def analyze(run):
    trace=json.loads((run/'furiosa_trace.json').read_text())
    events=trace.get('traceEvents',[]) if isinstance(trace,dict) else trace
    tasks=[e for e in events if '::Task' in str(e.get('name','')) and e.get('ph')=='X']
    cpu=json.loads((run/'torch_cpu_trace.json').read_text())
    events=cpu.get('traceEvents',[]) if isinstance(cpu,dict) else cpu
    windows=[e for e in events if e.get('ph')=='X' and 'window_preparation' in str(e.get('name',''))]
    log=(run/'run.log').read_text(errors='replace')
    time=re.search(r'추론 완료 소요 시간:\s*([0-9.]+)',log)
    wall=float(time.group(1)) if time else None
    active=union_us(tasks)/1e6
    result={'run':run.name,'verdict':'mixed_cpu_model_operations' if windows else 'requires_device_boundary_audit',
      'inference_wall_s':wall,'npu_task_count':len(tasks),'npu_task_interval_union_s':active,
      'npu_task_union_over_wall_pct':100*active/wall if wall else None,
      'cpu_model_window_scope_count':len(windows),'cpu_window_interval_union_s':union_us(windows)/1e6,
      'note':'Task duration is profiler evidence, not calibrated hardware utilization. Window time excludes transfers outside its span. Cold/warm split is unavailable in single-shot runs.',
      'next':'Replace CPU convolution packing with resident NPU/TCL feature; measure same-process warm inference.'}
    (run/'bottleneck_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);analyze(p.parse_args().run)

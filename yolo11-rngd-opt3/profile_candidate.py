import os,sys,time,json,fcntl,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from yolo11_optimized import configure,make_model
configure()
import resident_model
from direct_route import adapt
resident_model.adapt=adapt
import torch,numpy as np
from tools.optimization_packet import PhaseTimings,union_duration
from compiler_engine.execution_auditor import NPUExecutionAuditor,record_function
out=ROOT/'runs'/('remaining_'+str(time.time_ns()));out.mkdir(parents=True)
os.environ['FURIOSA_GRAPH_AUDIT_PATH']=str(out/'compiler_graphs.jsonl')
lock=open('/root/rngd-model-forge/reports/rngd_porting/npu.lock','a');fcntl.flock(lock,fcntl.LOCK_EX)
with torch.no_grad():
 model=make_model(ROOT/'models/yolo11n_640.onnx',ROOT/'static_metadata.json')
 x=torch.from_numpy(np.load(ROOT/'image_reference.npz')['input']).to('furiosa:0')
 expected=model(x).cpu()
 phases=PhaseTimings()
 for _ in range(5):
  with phases.phase('total'): y=model(x).cpu()
  torch.testing.assert_close(y,expected,rtol=0,atol=0)
 class Timed(torch.fx.Interpreter):
  def run_node(self,node):
   with phases.phase(str(node.op)+'::'+str(node.target)):
    return super().run_node(node)
 for _ in range(3):
  with phases.phase('instrumented_total'): y=Timed(model).run(x).cpu()
 phases.save(out/'timing.json')
 anchor=torch.ones(32,32).to('furiosa:0');(anchor+1).cpu()
 with NPUExecutionAuditor(out,'yolo11-remaining'):
  with record_function('diagnostic::anchor'): (anchor+1).cpu()
  with record_function('pipeline_inference::yolo11'): z=Timed(model).run(x).cpu()
 torch.testing.assert_close(z,expected,rtol=0,atol=0)
 events=json.loads((out/'furiosa_trace.json').read_text())['traceEvents']
 spans=[e for e in events if e.get('name')=='pipeline_inference::yolo11' and e.get('ph')=='X']
 span=spans[0];a,b=span['ts'],span['ts']+span['dur'];groups=collections.defaultdict(list)
 for e in events:
  if e.get('ph')=='X' and e.get('dur',0)>0:
   lo,hi=max(a,e['ts']),min(b,e['ts']+e['dur'])
   if hi>lo:groups[e['name']].append((lo,hi))
 rows=sorted([{'name':k,'union_ms':union_duration(v)/1000,'count':len(v)} for k,v in groups.items()],key=lambda r:-r['union_ms'])
 (out/'trace_breakdown.json').write_text(json.dumps({'window_ms':span['dur']/1000,'rows':rows},indent=2))
 (out/'optimization_features.json').write_text(json.dumps({'observations':{'many_small_npu_tasks':{'value':True,'evidence':str(out/'summary.json')}}}))
 print('RESULT '+str(out),flush=True)
 print(json.dumps({'warm':phases.samples['total'],'trace':rows[:40]}),flush=True)

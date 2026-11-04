"""Capture genuine SDK Runnables, or verify zero-compiler FXBEX replay."""
import argparse
import fcntl
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT/'fxbex/yolo11n-fp32-optimized-fusion8-640-sdk2026.4.0.fxbex'
p = argparse.ArgumentParser()
p.add_argument('command',choices=['capture','replay'])
p.add_argument('--reference',type=Path,default=Path('/root/yolo11-rngd/runs/resident_image_v4/output.npy'))
a = p.parse_args()
from yolo11_optimized import YoloAdapter, configure, make_model, sha, sources, MODEL_FILE, REVISION
configure()
import numpy as np
import torch
from fxbex.settings import lock_path
report = ROOT/'runs'/('fxbex_'+a.command+'_'+str(time.time_ns()))
report.mkdir(parents=True,exist_ok=False)
if a.command == 'capture':
    from fxbex.replay import capture
    from fxbex.package import build
    from fxbex.runtime import environment
    from onnx2torch import convert
    from static_adapter import specialize, specialize_saved
    import onnx
    source = ROOT/'compiled_yolo11n640'
    source.mkdir(exist_ok=False)
    metadata = source/'static_metadata.json'
    weights = ROOT/'models'/MODEL_FILE
    example = torch.from_numpy(np.load(ROOT/'image_reference.npz')['input'])
    with torch.no_grad():
        first = specialize(convert(onnx.load(weights)).eval(), example, metadata)
        restored = specialize_saved(convert(onnx.load(weights)).eval(), example.shape, metadata)
        torch.testing.assert_close(first(example),restored(example),rtol=0,atol=0)
        del first, restored
        with lock_path().open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with capture(source) as stats:
                adapter = YoloAdapter.__new__(YoloAdapter)
                adapter.report, adapter.stats = report, stats
                adapter.model = make_model(weights,metadata)
                result = adapter.predict({'image':str(ROOT/'bus.jpg')})
    actual = np.load(Path(result['output'])/'output.npy')
    np.testing.assert_array_equal(actual,np.load(a.reference))
    def write(name,data):
        (source/name).write_text(json.dumps(data,indent=2))
    write('weights_index.json',{'weights':[{'name':MODEL_FILE,'external_path':str(weights),'sha256':sha(weights)}]})
    write('runtime_config.json',{'project_sources':sources(),'input_shape':[1,3,640,640],
                               'compiler_policy':'recorded-runnables-only','task':'predict'})
    write('memory_plan.json',{'allocator':'SDK','fusion':8,'mesh':'Single','dtype':'float32'})
    write('manifest.json',{'model':{'name':'giangndm/yolo11-onnx','revision':REVISION,'modality':'image-to-detections'},
                          'compatibility':{'packages':environment(),'hardware':'rngd','fusion':8},
                          'runtime':{'adapter':'yolo11n640_optimized'}})
    build(source,PACKAGE)
    result.update(package=str(PACKAGE),capture_exact_match=True,
                  saved_runnables=len(json.loads((source/'execution_plan.json').read_text())['steps']))
else:
    from fxbex.runtime import Session
    with Session(PACKAGE,report_dir=report) as session:
        result = session.predict({'image':str(ROOT/'bus.jpg')})
        assert result['compiler_calls']==0 and result['replay_hits']>0
        np.testing.assert_array_equal(np.load(Path(result['output'])/'output.npy'),np.load(a.reference))
        with torch.no_grad():
            from yolo11_optimized import prepare_image
            _,data = prepare_image(ROOT/'bus.jpg')
            value = torch.from_numpy(data).to('furiosa:0')
            timings=[]
            for _ in range(3):
                start=time.perf_counter()
                actual=session.adapter.model(value).cpu().numpy()
                timings.append(time.perf_counter()-start)
                np.testing.assert_array_equal(actual,np.load(a.reference))
        result.update(replay_exact_match=True,warm_s=timings)
(report/'package_result.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)

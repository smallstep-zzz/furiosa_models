"""Inspect the public ONNX graph and validate conversion against ONNX Runtime."""
import json
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from huggingface_hub import hf_hub_download
from onnx2torch import convert

ROOT = Path(__file__).resolve().parent
REVISION = '8b180c762d5cf217886d16e64403508042207e4d'
torch.set_num_threads(8)
path = hf_hub_download('giangndm/yolo11-onnx', 'yolo11n_640.onnx', revision=REVISION,
                       local_dir=ROOT / 'models')
model = onnx.load(path)
onnx.checker.check_model(model)
report = {'revision': REVISION, 'operators': dict(Counter(n.op_type for n in model.graph.node)),
          'opsets': {o.domain: o.version for o in model.opset_import},
          'inputs': [str(x) for x in model.graph.input], 'outputs': [str(x) for x in model.graph.output]}
(ROOT / 'inventory.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report), flush=True)
module = convert(model).eval()
torch.manual_seed(20261103)
x = torch.rand(1, 3, 640, 640)
opts = ort.SessionOptions()
opts.intra_op_num_threads = 8
session = ort.InferenceSession(path, sess_options=opts, providers=['CPUExecutionProvider'])
with torch.no_grad():
    actual = module(x)
expected = session.run(None, {session.get_inputs()[0].name: x.numpy()})[0]
if isinstance(actual, (tuple, list)):
    actual = actual[0]
torch.testing.assert_close(actual, torch.from_numpy(expected), rtol=1e-4, atol=1e-3)
np.savez(ROOT / 'cpu_reference.npz', input=x.numpy(), output=expected)
report['conversion'] = {'passed': True, 'max_abs': float(np.max(np.abs(actual.numpy()-expected))),
                        'shape': list(expected.shape)}
(ROOT / 'inventory.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report['conversion']), flush=True)

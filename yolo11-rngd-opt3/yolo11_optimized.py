"""Pinned YOLO11n/640 adapter for the independent FXBEX runtime."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
MODEL_FILE = 'yolo11n_640.onnx'
REVISION = '8b180c762d5cf217886d16e64403508042207e4d'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sources():
    files = [ROOT / n for n in ('yolo11_optimized.py', 'fast_pack.py', 'resident_model.py', 'conv_adapter.py',
                                'pool_adapter.py', 'static_adapter.py', 'image_demo.py')]
    files += sorted(Path('/root/rngd-model-forge/compiler_engine').glob('*.py'))
    return {str(p): sha(p) for p in files}


def configure():
    os.environ['FURIOSA_REQUIRE_NPU'] = '1'
    os.environ.setdefault('RAYON_NUM_THREADS', '16')
    os.environ.setdefault('OMP_NUM_THREADS', '8')
    sys.path.insert(0, '/root/rngd-model-forge')
    import torch
    from compiler_engine.renegade_diffusion_compiler import _configure_fused_pes
    from furiosa.torch.config import set_compiler_config, set_cpu_fallback, get_cpu_fallback
    from resident_model import CONFIG, build_linear, COUNTS
    torch.set_num_threads(8)
    _configure_fused_pes()
    set_cpu_fallback(False)
    assert get_cpu_fallback() is False
    set_compiler_config(CONFIG)
    build_linear.cache_clear()
    COUNTS.clear()
    from fast_pack import build_pack
    build_pack.cache_clear()


def make_model(weights, metadata):
    import onnx
    from onnx2torch import convert
    from static_adapter import specialize_saved
    from resident_model import adapt
    # Static metadata is validated at production time; loading performs no CPU model forward.
    model = specialize_saved(convert(onnx.load(weights)).eval(), (1, 3, 640, 640), metadata)
    return adapt(model).to('furiosa:0')


def prepare_image(path):
    import numpy as np
    from PIL import Image
    image = Image.open(path).convert('RGB')
    scale = min(640/image.width, 640/image.height)
    small = image.resize((round(image.width*scale), round(image.height*scale)), Image.Resampling.BILINEAR)
    canvas = Image.new('RGB', (640,640), (114,114,114))
    canvas.paste(small, ((640-small.width)//2, (640-small.height)//2))
    return canvas, np.asarray(canvas, dtype=np.float32).transpose(2,0,1)[None]/255


class YoloAdapter:
    @staticmethod
    def preflight(root, weights_dir=None):
        from fxbex.package import PackageError
        root = Path(root)
        config = json.loads((root/'runtime_config.json').read_text())
        if config['project_sources'] != sources():
            raise PackageError('YOLO installed adapter/project source hash mismatch')
        index = json.loads((root/'weights_index.json').read_text())['weights']
        if len(index) != 1 or index[0]['name'] != MODEL_FILE:
            raise PackageError('YOLO requires its exact pinned ONNX weights')
        weight = Path(weights_dir)/MODEL_FILE if weights_dir else Path(index[0]['external_path'])
        if sha(weight) != index[0]['sha256']:
            raise PackageError('YOLO ONNX checksum mismatch')
        if config['input_shape'] != [1,3,640,640]:
            raise PackageError('Only batch1, RGB640 is validated')
        return {'adapter': 'yolo11n640_optimized', 'inference_executed': False}

    def __init__(self, root, report, stack, weights_dir=None):
        from fxbex.replay import replay
        self.preflight(root, weights_dir)
        configure()
        import torch
        self.report = Path(report)
        self.stats = stack.enter_context(replay(root))
        index = json.loads((Path(root)/'weights_index.json').read_text())['weights'][0]
        weights = Path(weights_dir)/MODEL_FILE if weights_dir else Path(index['external_path'])
        with torch.no_grad():
            self.model = make_model(weights, Path(root)/'static_metadata.json')

    def predict(self, inputs):
        if set(inputs) != {'image'}:
            raise ValueError('Expected {"image": "/absolute/path/to/image"}')
        import numpy as np
        import torch
        from PIL import ImageDraw
        from compiler_engine.execution_auditor import NPUExecutionAuditor
        from contextlib import nullcontext
        from image_demo import detections
        output = self.report / ('predict_'+str(time.time_ns()))
        output.mkdir(parents=True, exist_ok=False)
        os.environ['FURIOSA_GRAPH_AUDIT_PATH'] = str(output/'compiler_graphs.jsonl')
        canvas, data = prepare_image(inputs['image'])
        with torch.no_grad():
            resident = torch.from_numpy(data).to('furiosa:0')
            audit_run = not hasattr(self, 'validated_audit')
            preparation_s = 0.0
            if audit_run:
                # Keep SDK export/preparation outside the inference profiler.
                preparing = time.perf_counter()
                self.model(resident).cpu()
                preparation_s = time.perf_counter()-preparing
            with (NPUExecutionAuditor(output, 'yolo11n-fxbex') if audit_run else nullcontext()) as audit:
                with (audit.span('model_component::yolo11n_full') if audit else nullcontext()):
                    start = time.perf_counter()
                    value = self.model(resident)
                    assert value.device.type == 'furiosa'
                with (audit.span('host::output_transfer') if audit else nullcontext()):
                    actual = value.cpu().numpy()
                    seconds = time.perf_counter()-start
        np.save(output/'output.npy', actual)
        boxes, scores, labels = detections(actual)
        draw = ImageDraw.Draw(canvas)
        results = []
        for box, score, label in zip(boxes.tolist(), scores.tolist(), labels.tolist()):
            draw.rectangle(box, outline=(255,60,30), width=3)
            draw.text((box[0]+3,box[1]+3),f'{label}: {score:.3f}',fill=(255,60,30))
            results.append({'class_id':label,'score':score,'box_xyxy_letterboxed':box})
        canvas.save(output/'detections.png')
        tasks = None
        if audit_run:
            summary = json.loads((output/'summary.json').read_text())
            tasks = summary['profiler']['npu_task_events']
            if not tasks or summary['cpu']['sdk_cpu_execution_count']:
                raise RuntimeError('Missing NPU trace or observed SDK CPU execution')
            self.validated_audit = str(output/'summary.json')
        result = {'output':str(output),'detections':results,'inference_s':seconds,'preparation_s':preparation_s,
                  'npu_tasks':tasks,'current_request_traced':audit_run,'validated_audit':self.validated_audit,'compiler_calls':self.stats.compile_calls,'replay_hits':self.stats.replay_hits,
                  'host_work':['RGB resize, letterbox, normalize','NMS and drawing'],
                  'cpu_model_forward_during_load':False}
        (output/'result.json').write_text(json.dumps(result,indent=2))
        return result

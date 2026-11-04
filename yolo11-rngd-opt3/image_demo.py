"""Prepare a real-image ONNX reference or annotate an RNGD output (CPU host work)."""
import argparse
import json
from pathlib import Path
import urllib.request

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw
import torch
from torchvision.ops import batched_nms, box_iou

ROOT = Path(__file__).resolve().parent
SOURCE = 'https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/bus.jpg'


def detections(output):
    rows = torch.from_numpy(output[0].T.copy())
    scores, labels = rows[:, 4:].max(dim=1)
    keep = scores >= 0.25
    rows, scores, labels = rows[keep], scores[keep], labels[keep]
    boxes = torch.cat((rows[:, :2]-rows[:, 2:4]/2, rows[:, :2]+rows[:, 2:4]/2), dim=1)
    indices = batched_nms(boxes, scores, labels, 0.45)[:300]
    return boxes[indices], scores[indices], labels[indices]


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, help='Existing NPU run directory; omit to prepare input')
    p.add_argument('--image', type=Path, help='User image; defaults to the upstream bus sample')
    a = p.parse_args()
    torch.set_num_threads(8)
    if a.output is None:
        image_path = a.image or ROOT / 'bus.jpg'
        if not image_path.exists():
            if a.image:
                raise FileNotFoundError(image_path)
            urllib.request.urlretrieve(SOURCE, image_path)
        image = Image.open(image_path).convert('RGB')
        scale = min(640/image.width, 640/image.height)
        resized = image.resize((round(image.width*scale), round(image.height*scale)), Image.Resampling.BILINEAR)
        canvas = Image.new('RGB', (640, 640), (114, 114, 114))
        canvas.paste(resized, ((640-resized.width)//2, (640-resized.height)//2))
        canvas.save(ROOT / 'input_640.png')
        x = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1)[None]/255
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 8
        session = ort.InferenceSession(str(ROOT / 'models/yolo11n_640.onnx'), sess_options=opts,
                                       providers=['CPUExecutionProvider'])
        y = session.run(None, {session.get_inputs()[0].name: x})[0]
        np.savez(ROOT / 'image_reference.npz', input=x, output=y)
        print(json.dumps({'input': list(x.shape), 'cpu_detections': len(detections(y)[0]), 'source': str(a.image) if a.image else SOURCE}))
    else:
        expected = np.load(ROOT / 'image_reference.npz')['output']
        actual = np.load(a.output / 'output.npy')
        cpu_boxes, cpu_scores, cpu_labels = detections(expected)
        boxes, scores, labels = detections(actual)
        assert len(boxes) == len(cpu_boxes) > 0
        matched = set()
        matches = []
        for i in range(len(boxes)):
            ious = box_iou(boxes[i:i+1], cpu_boxes)[0]
            ious[cpu_labels != labels[i]] = -1
            best = int(ious.argmax())
            assert best not in matched and float(ious[best]) >= 0.99
            assert abs(float(scores[i]-cpu_scores[best])) <= 0.002
            matched.add(best)
            matches.append(float(ious[best]))
        classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus']
        image = Image.open(ROOT / 'input_640.png').convert('RGB')
        draw = ImageDraw.Draw(image)
        rows = []
        for box, score, label in zip(boxes.tolist(), scores.tolist(), labels.tolist()):
            name = classes[label] if label < len(classes) else str(label)
            draw.rectangle(box, outline=(255, 60, 30), width=3)
            draw.text((box[0]+3, box[1]+3), f'{name} {score:.3f}', fill=(255, 60, 30), stroke_width=1)
            rows.append({'class_id': label, 'class': name, 'score': score, 'box_xyxy': box})
        image.save(a.output / 'detections.png')
        report = {'status': 'passed', 'cpu_count': len(cpu_boxes), 'npu_count': len(boxes),
                  'minimum_matched_iou': min(matches), 'detections': rows,
                  'host_work': ['RGB resize/letterbox/normalization before inference', 'NMS and annotation after inference']}
        (a.output / 'detections.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report))

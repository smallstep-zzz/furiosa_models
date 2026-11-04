"""YOLO image controls on the shared FXBEX load/run/unload page."""
from pathlib import Path
import shutil
import time
import gradio as gr
from fxbex.ui import build_ui

ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT/'fxbex/yolo11n-fp32comp-opt3-640-fusion8-sdk2026.4.0-portable.fxbex'
OUTPUT = ROOT/'web_outputs'
OUTPUT.mkdir(exist_ok=True)
SAMPLE = OUTPUT/'sample.png'
if not SAMPLE.exists():
    shutil.copy2(ROOT/'validated_detections.png', SAMPLE)
CLASSES = (ROOT/'models/classes.txt').read_text().splitlines()


def model_controls(run):
    def detect(path, progress=gr.Progress()):
        if not path:
            raise gr.Error('이미지를 먼저 선택해 주세요.')
        progress(0.1, desc='NPU 검출 중 · 첫 요청에는 실행 준비 시간이 추가됩니다.')
        result, timing = run('predict', {'image': path})
        source = Path(result['output'])/'detections.png'
        target = OUTPUT/f'detection_{time.time_ns()}.png'
        shutil.copy2(source, target)
        rows = [[CLASSES[d['class_id']], round(d['score'],4),
                 *[round(v,1) for v in d['box_xyxy_letterboxed']]] for d in result['detections']]
        progress(1, desc='검출 완료')
        return str(target), rows, (f"{len(rows)}개 검출 · 모델 추론·출력 전송 {result['inference_s']*1000:.0f}ms\n"
                                   +timing.splitlines()[0].replace('Request:', '전체 요청:'))

    with gr.Row():
        with gr.Column():
            image = gr.Image(value=str(ROOT/'bus.jpg'), type='filepath', label='입력 이미지', height=480)
            button = gr.Button('객체 검출', variant='primary')
            status = gr.Textbox(label='검출 상태 / 시간', value='객체 검출을 누르세요. 모델이 해제돼 있다면 Load로 적재하세요.')
        with gr.Column():
            output = gr.Image(value=str(SAMPLE), label='검출 결과 · 640×640', height=480, interactive=False)
    table = gr.Dataframe(headers=['클래스','신뢰도','x1','y1','x2','y2'],
                         label='검출 목록 · 좌표는 640×640 결과 이미지 기준', interactive=False)
    button.click(detect, image, [output,table,status], api_name='detect', concurrency_limit=1)
    image.change(lambda:(None,[], '이미지가 변경됐습니다. 객체 검출을 눌러 주세요.'),
                 outputs=[output,table,status], queue=False, api_name=False)


with gr.Blocks(title='YOLO11 · RNGD 객체 검출') as app:
    gr.Markdown('# YOLO11 · RNGD 객체 검출\n'
                '이미지 선택 → **객체 검출**로 확인하세요. 모델이 해제돼 있다면 먼저 Load를 누르세요. Unload로 모델을 내릴 수 있습니다.\n\n'
                '640×640 · FP32 입출력, 보정 BF16 곱셈 모델입니다. 최신 반복 추론 시간은 아래 검출 결과에 표시됩니다. '
                '모델을 새로 적재한 뒤 첫 요청에는 수십 초의 준비 시간이 필요합니다. 오른쪽은 실제 NPU 검증 결과입니다.')
    gr.File(value=str(PACKAGE), label='가중치·실행 코드 포함 FXBEX 다운로드', interactive=False)
    unload = build_ui(default_package=str(PACKAGE), package_dir=PACKAGE.parent,
                      mode='predict', controls=model_controls)
## The worker is shared: browser refresh must not evict the warmed model.
## Explicit Unload and server shutdown still release it.

if __name__ == '__main__':
    try:
        app.queue(default_concurrency_limit=1).launch(server_name='127.0.0.1', server_port=7863,
            allowed_paths=[str(OUTPUT),str(ROOT/'bus.jpg'),str(PACKAGE)], max_file_size='100mb',
            theme=gr.themes.Soft(), footer_links=[])
    finally:
        unload()

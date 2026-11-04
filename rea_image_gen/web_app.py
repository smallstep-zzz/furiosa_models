#!/usr/bin/env python3
"""
[Furiosa Renegade NPU & AI Image Studio Web UI]
- Gradio 기반 웹 대시보드
- 기능 1: NPU 가속 이미지 생성 (SDXL-Turbo / FLUX.1-dev / FLUX 3 Klein / Qwen-Image-2.1)
- 기능 2: fal/AuraSR-v2 4배 초해상화(Upscaling) 업스케일러
- 기능 3: 색상 다양성 및 생성 품질 정밀 검증 (3D 컬러 엔트로피, 채널별 표준편차, 고유 색상 비율, PASS/FAIL 판정)
"""

import os
import time
import tempfile
import multiprocessing
from pathlib import Path
from PIL import Image

TOTAL_CORES = multiprocessing.cpu_count()
TARGET_THREADS = max(1, TOTAL_CORES - 2)
os.environ.setdefault("RAYON_NUM_THREADS", str(TARGET_THREADS))
os.environ.setdefault("OMP_NUM_THREADS", "8")

# HF 토큰 설정
TOKEN_FILE = Path(__file__).parent / "hf_token.env"
if TOKEN_FILE.exists():
    with open(TOKEN_FILE) as f:
        for line in f:
            if line.startswith("HF_TOKEN="):
                os.environ["HF_TOKEN"] = line.strip().split("=", 1)[1].strip("\"'")
                break
if "HF_TOKEN" not in os.environ and Path("/root/.local/share/huggingface/token").exists():
    os.environ["HF_TOKEN"] = Path("/root/.local/share/huggingface/token").read_text().strip()

import gradio as gr
from run_image_gen_npu import run_pipeline

from verify_color_diversity import analyze_image_color_diversity

OUTPUT_DIR = Path("/root/rea_image_gen/web_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_image_ui(model_name: str, prompt: str, steps: int, seed: int):
    """NPU 기반 이미지 생성 및 자동 품질 검증"""
    if not prompt or not prompt.strip():
        return None, "⚠️ 프롬프트를 입력해 주세요.", ""

    if model_name != "sdxl":
        return None, "현재 웹 검증 대상은 SDXL Turbo입니다.", ""
    timestamp = time.time_ns()
    out_path = OUTPUT_DIR / f"{model_name}_{timestamp}.png"

    try:
        t0 = time.time()
        analysis = run_pipeline(
            model_type=model_name,
            prompt=prompt,
            output_path=str(out_path),
            steps=int(steps),
            seed=int(seed)
        )
        elapsed = time.time() - t0

        img = Image.open(out_path)

        # 검증 결과 텍스트 포맷팅
        status_badge = "✅ [PASS] 정상 판정" if analysis["is_valid"] else "❌ [FAIL] 비정상 판정"
        verdict_text = (
            f"### {status_badge}\n\n"
            f"- **소요 시간**: {elapsed:.2f}초\n"
            f"- **이미지 해상도**: {analysis['resolution']}\n"
            f"- **품질/색상 점수**: **{analysis['quality_score']} / 100.0**\n"
            f"- **3D 색상 엔트로피**: {analysis['color_entropy']} / {analysis['max_entropy']}\n"
            f"- **채널별 표준편차**: R={analysis['channel_std']['R']}, G={analysis['channel_std']['G']}, B={analysis['channel_std']['B']} (평균 {analysis['channel_std']['avg']})\n"
            f"- **고유 색상 비율**: {analysis['unique_color_ratio']*100:.2f}% ({analysis['unique_colors']:,}개)\n"
        )
        if analysis["issues"]:
            verdict_text += f"\n- **감지된 결함**: {', '.join(analysis['issues'])}"

        import json
        metrics = analysis.get("work_metrics", {})
        verdict_text += f"\n\n### 작업별 시간\n- 파이프라인 재사용: {metrics.get('pipeline_reused')}\n- 전체 요청: {metrics.get('request_wall_s')}초\n- 추론 호출(필요시 컴파일 포함): {metrics.get('inference_call_wall_s')}초\n"
        verdict_text += "\n|작업|시간(초)|호출 횟수|\n|---|---:|---:|\n"
        for name, seconds in metrics.get("seconds", {}).items():
            verdict_text += f"|{name}|{seconds:.4f}|{metrics['counts'].get(name,0)}|\n"
        verdict_text += "\n시간 구간은 중첩될 수 있습니다. 순수 NPU 커널 시간은 공식 trace로 확인합니다.\n"
        memory = metrics.get("memory", {})
        verdict_text += "\n### 메모리 진단\n- SMI 사용량과 SDK 텐서 논리 용량은 다른 지표입니다. 실제 할당량 API는 확인되지 않았습니다.\n"
        for component, devices in memory.get("components", {}).items():
            for device, item in devices.items():
                verdict_text += f"- {component} / {device}: 가중치 논리 용량 {item['parameter_logical_bytes']/2**30:.3f} GiB\n"
        verdict_text += "- 실제 HBM 상주 용량: 미확정. 아래 상세 로그의 비TCL 대조 실험과 메모리 진단을 참고하세요.\n"
        log_text = json.dumps(metrics, ensure_ascii=False, indent=2) + f"\n저장 경로: {out_path.resolve()}\nSDXL 파이프라인 재사용 · CPU 모델 fallback 금지 · 최초 실행 trace 별도 저장 · BF16 수치 검증 일부 미완료"
        return img, verdict_text, log_text
    except Exception as e:
        error_msg = f"❌ 실행 오류 발생: {str(e)}"
        if "gated repo" in str(e).lower() or "403" in str(e):
            error_msg += "\n\n(참고: 게이트 모델의 경우 Hugging Face 웹사이트에서 라이선스 동의가 필요합니다.)"
        return None, error_msg, str(e)


def upscale_image_ui(input_image: Image.Image, max_batch_size: int):
    from aurasr_upscaler import upscale_image_with_aurasr
    """fal/AuraSR-v2 기반 4x 초해상화(Upscaling)"""
    if input_image is None:
        return None, "⚠️ 업스케일할 이미지를 업로드하거나 이전 단계에서 전달해 주세요."

    try:
        t0 = time.time()
        orig_w, orig_h = input_image.size
        upscaled = upscale_image_with_aurasr(input_image, max_batch_size=int(max_batch_size))
        elapsed = time.time() - t0
        new_w, new_h = upscaled.size

        timestamp = time.time_ns()
        save_path = OUTPUT_DIR / f"aurasr_upscaled_{timestamp}.png"
        upscaled.save(save_path)

        info = (
            f"### ✅ fal/AuraSR-v2 4x 업스케일 완료!\n\n"
            f"- **입력 해상도**: {orig_w}x{orig_h}\n"
            f"- **출력 해상도**: **{new_w}x{new_h} (4배 확대)**\n"
            f"- **처리 소요 시간**: {elapsed:.2f}초\n"
            f"- **저장 파일**: `{save_path.resolve()}`\n"
        )
        return upscaled, info
    except Exception as e:
        return None, f"❌ 업스케일 처리 중 오류: {str(e)}"


def verify_image_ui(input_image: Image.Image):
    """색상 다양성 및 이미지 품질 정밀 검증"""
    if input_image is None:
        return "⚠️ 검증할 이미지를 업로드해 주세요."

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        input_image.save(tmp.name)
        tmp_path = tmp.name

    try:
        res = analyze_image_color_diversity(tmp_path, verbose=False)
        badge = "🟢 [PASS] 정상 다양성 충족" if res["is_valid"] else "🔴 [FAIL] 단색화 또는 비정상 이미지"
        out_md = (
            f"## {badge}\n\n"
            f"- **종합 다양성 점수**: **{res['quality_score']} / 100점**\n"
            f"- **분석 해상도**: {res['resolution']} (총 {res['total_pixels']:,} 픽셀)\n"
            f"- **고유 색상 수**: {res['unique_colors']:,}개 (전체 대비 {res['unique_color_ratio']*100:.2f}%)\n"
            f"- **3D 컬러 히스토그램 엔트로피**: {res['color_entropy']} (이론상 최대: {res['max_entropy']})\n"
            f"- **채널별 표준편차**:\n"
            f"  - R 채널: {res['channel_std']['R']}\n"
            f"  - G 채널: {res['channel_std']['G']}\n"
            f"  - B 채널: {res['channel_std']['B']}\n"
            f"  - 평균 편차: {res['channel_std']['avg']}\n"
        )
        if res["issues"]:
            out_md += f"\n> ⚠️ **검출된 주의사항**: {', '.join(res['issues'])}\n"
        else:
            out_md += "\n> ✨ 색상 뭉침 및 붕괴 없이 풍부한 디테일이 유지되고 있습니다.\n"
        return out_md
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# UI 레이아웃 구성
with gr.Blocks(title="Furiosa Renegade NPU AI Studio") as demo:
    gr.Markdown(
        """
        # 🚀 Furiosa Renegade NPU & AI Image Studio
        **Furiosa Renegade NPU(`npu2`, 8 Cores, 48GB HBM) 하드웨어 가속 다중 모델 생성 & fal/AuraSR-v2 4x 업스케일러 & 정밀 검증 스튜디오**
        """
    )

    with gr.Tabs():
        # TAB 1: 이미지 생성
        with gr.TabItem("🎨 NPU 이미지 생성"):
            with gr.Row():
                with gr.Column(scale=1):
                    model_selector = gr.Dropdown(
                        label="생성 모델 선택",
                        choices=[
                            ("SDXL-Turbo (초고속 1-Step NPU 가속)", "sdxl"),
                        ],
                        value="sdxl",
                    )
                    prompt_input = gr.Textbox(
                        label="프롬프트 (Prompt)",
                        placeholder="생성하고 싶은 이미지 묘사를 입력하세요...",
                        value="A breathtaking cosmic dragon soaring through an aurora borealis nebula, vibrant scales, iridescent glowing light, hyper-detailed fantasy art 8k",
                        lines=4,
                    )
                    with gr.Row():
                        steps_input = gr.Slider(
                            label="추론 스텝 (Inference Steps)",
                            minimum=1,
                            maximum=4,
                            value=1,
                            step=1,
                        )
                        seed_input = gr.Number(label="시드 (Seed)", value=42, precision=0)

                    def update_steps_default(m):
                        defaults = {"sdxl": 1, "flux": 4, "qwen": 20, "flux3-klein": 16}
                        return gr.update(value=defaults.get(m, 4))

                    model_selector.change(
                        fn=update_steps_default,
                        inputs=[model_selector],
                        outputs=[steps_input],
                    )

                    gen_button = gr.Button("🚀 NPU 이미지 생성 시작", variant="primary", size="lg")

                with gr.Column(scale=1):
                    gen_output_image = gr.Image(label="생성 결과 이미지", type="pil")
                    gen_verdict = gr.Markdown(label="품질 및 다양성 검증")
                    gen_logs = gr.Textbox(label="작업별 시간·TCL 이력·저장 위치", lines=12, interactive=False)
                    send_to_upscale_btn = gr.Button("➡️ 이 이미지를 4x 업스케일러로 보내기")

            gen_button.click(
                fn=generate_image_ui,
                inputs=[model_selector, prompt_input, steps_input, seed_input],
                outputs=[gen_output_image, gen_verdict, gen_logs],
            )

        # TAB 2: AuraSR-v2 4x 업스케일러
        with gr.TabItem("🔍 fal/AuraSR-v2 4배 업스케일 (Super-Resolution)"):
            gr.Markdown(
                """
                ### fal/AuraSR-v2 GAN 기반 4x 초해상화 모델
                생성된 저해상도 이미지(512x512 등)를 선명하고 정밀한 4배 고화질(2048x2048 등)로 복원합니다.
                """
            )
            with gr.Row():
                with gr.Column():
                    upscale_input_image = gr.Image(label="업스케일할 원본 이미지", type="pil")
                    batch_size_input = gr.Slider(
                        label="타일 배치 크기 (Max Batch Size)",
                        minimum=1,
                        maximum=16,
                        value=8,
                        step=1,
                    )
                    upscale_button = gr.Button("✨ 4배 초해상화 실행", variant="primary", size="lg")
                with gr.Column():
                    upscale_output_image = gr.Image(label="4x 업스케일 결과", type="pil")
                    upscale_info = gr.Markdown()
                    send_to_verify_btn = gr.Button("➡️ 업스케일 결과를 품질 검증으로 보내기")

            upscale_button.click(
                fn=upscale_image_ui,
                inputs=[upscale_input_image, batch_size_input],
                outputs=[upscale_output_image, upscale_info],
            )

            send_to_upscale_btn.click(
                fn=lambda img: img,
                inputs=[gen_output_image],
                outputs=[upscale_input_image],
            )

        # TAB 3: 색상 다양성 및 품질 정밀 검증
        with gr.TabItem("📊 이미지 품질 & 색상 다양성 정밀 검증"):
            gr.Markdown(
                """
                ### 3D 히스토그램 엔트로피 & 색상 다양성 판정 엔진
                - 16x16x16 3D RGB 히스토그램 샤논 엔트로피(Shannon Entropy)
                - RGB 채널별 표준편차(Channel Standard Deviation)
                - 고유 색상 비율(Unique Color Ratio) 및 종합 품질 점수 자동 판정
                """
            )
            with gr.Row():
                with gr.Column():
                    verify_input_image = gr.Image(label="분석할 이미지", type="pil")
                    verify_button = gr.Button("🔬 품질 다양성 정밀 검사", variant="primary", size="lg")
                with gr.Column():
                    verify_result = gr.Markdown()

            verify_button.click(
                fn=verify_image_ui,
                inputs=[verify_input_image],
                outputs=[verify_result],
            )

            send_to_verify_btn.click(
                fn=lambda img: img,
                inputs=[upscale_output_image],
                outputs=[verify_input_image],
            )

    gr.Markdown(
        """
        ---
        💡 **팁**:
        - 생성 탭에서 생성된 이미지는 버튼 한 번으로 업스케일러나 품질 검증 탭으로 즉시 전달할 수 있습니다.
        - Furiosa Renegade 컴파일러는 멀티스레드 가속(`RAYON_NUM_THREADS=126`)으로 고속 바이너리를 빌드합니다.
        """
    )


if __name__ == "__main__":
    run_pipeline("sdxl", "A red fox beside a mountain lake at sunrise", str(OUTPUT_DIR / "startup.png"), 1, 20261102)
    demo.queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
    )

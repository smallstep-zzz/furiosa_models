#!/usr/bin/env python3
"""
Furiosa Renegade NPU 기반 SDXL 이미지 생성 및 색상 다양성 검증 통합 파이프라인
- Furiosa torch.compile backend를 SDXL 텍스트 인코더, UNet, VAE에 연결합니다.
  지원 그래프는 NPU에서 실행하고 미지원 그래프는 CPU eager fallback을 사용합니다.
- 생성된 이미지에 대해 3D 색상 공간 히스토그램 엔트로피와 채널별 표준편차를 통해 색상 다양성을 자동 검증합니다.
"""

import os
import sys
import multiprocessing
import time
import argparse
from pathlib import Path

# 허깅페이스 토큰 자동 로드
TOKEN_FILE = Path(__file__).parent / "hf_token.env"
if TOKEN_FILE.exists():
    with open(TOKEN_FILE) as f:
        for line in f:
            if line.startswith("HF_TOKEN="):
                token = line.strip().split("=", 1)[1].strip("\"'")
                os.environ["HF_TOKEN"] = token

# Furiosa backend 연동 설정
THREADS = max(1, multiprocessing.cpu_count() - 2)
os.environ["RAYON_NUM_THREADS"] = str(THREADS)
os.environ["OMP_NUM_THREADS"] = str(THREADS)

import torch
from diffusers import AutoPipelineForText2Image
from verify_color_diversity import analyze_image_color_diversity
from compiler_engine.renegade_diffusion_compiler import enable_npu_graphs


def generate_image_npu(
    prompt: str,
    output_path: str = "npu_generated.png",
    steps: int = 1,
    seed: int = 42,
    model_id: str = "stabilityai/sdxl-turbo",
):
    print("=" * 65)
    print("1. 모델 로드 (SDXL-Turbo)")
    print(f"   Model: {model_id}")
    print("=" * 65)

    pipe = AutoPipelineForText2Image.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        variant="fp16",
        token=os.environ.get("HF_TOKEN")
    )
    pipe.to("cpu")

    # 지원 그래프를 NPU에 컴파일하는 backend 연결
    enable_npu_graphs(pipe, "sdxl")

    print("\n" + "=" * 65)
    print("2. SDXL 이미지 생성 실행 (Furiosa backend; 미지원 그래프는 CPU fallback)")
    print(f"   - 프롬프트: \"{prompt}\"")
    print(f"   - NPU 디바이스: furiosa:0")
    print(f"   - 추론 스텝: {steps}, Seed: {seed}")
    print("=" * 65)

    generator = torch.manual_seed(seed)
    start_time = time.time()
    result = pipe(
        prompt=prompt,
        num_inference_steps=steps,
        guidance_scale=0.0,
        generator=generator,
    )
    image = result.images[0]
    elapsed = time.time() - start_time
    print(f"   총 생성 소요 시간: {elapsed:.2f}초")

    # 저장
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_file)
    print(f"   이미지 저장 완료: {out_file.resolve()}\n")

    # 3. 색상 다양성 자동 검증
    analysis = analyze_image_color_diversity(str(out_file.resolve()), verbose=True)
    return analysis


def main():
    parser = argparse.ArgumentParser(description="Furiosa Renegade NPU 가속 SDXL 이미지 생성 및 색상 검증")
    parser.add_argument(
        "--prompt",
        default="A majestic cybernetic tiger with neon glowing stripes in a dark futuristic jungle, 8k cinematic masterpiece",
        help="생성할 이미지 프롬프트"
    )
    parser.add_argument(
        "-o", "--output",
        default="npu_result.png",
        help="출력 파일 경로 (기본: npu_result.png)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1,
        help="추론 스텝 (기본: 1)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="랜덤 시드 (기본: 42)"
    )

    args = parser.parse_args()
    generate_image_npu(
        prompt=args.prompt,
        output_path=args.output,
        steps=args.steps,
        seed=args.seed
    )


if __name__ == "__main__":
    main()

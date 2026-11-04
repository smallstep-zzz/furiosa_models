#!/usr/bin/env python3
"""
SDXL-Turbo 이미지 생성 및 색상 다양성 자동 검증 파이프라인
"""

import os
import sys
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
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

import torch
from diffusers import AutoPipelineForText2Image
from verify_color_diversity import analyze_image_color_diversity

def generate_and_verify(
    prompt: str,
    output_path: str = "output.png",
    steps: int = 1,
    guidance_scale: float = 0.0,
    seed: int = 42,
    model_id: str = "stabilityai/sdxl-turbo",
):
    print("=" * 60)
    print("1. 모델 로드 (SDXL-Turbo)...")
    print(f"   Model: {model_id}")
    print("=" * 60)
    
    generator = torch.manual_seed(seed)
    
    pipe = AutoPipelineForText2Image.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        variant="fp16",
        token=os.environ.get("HF_TOKEN")
    )
    pipe.to("cpu")

    print("\n" + "=" * 60)
    print(f"2. 이미지 생성 시작")
    print(f"   - 프롬프트: \"{prompt}\"")
    print(f"   - 스텝 수: {steps}, Seed: {seed}")
    print("=" * 60)

    start_time = time.time()
    result = pipe(
        prompt=prompt,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )
    image = result.images[0]
    elapsed = time.time() - start_time
    print(f"   생성 소요 시간: {elapsed:.2f}초")

    # 결과 저장
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_file)
    print(f"   이미지 저장 완료: {out_file.resolve()}\n")

    # 3. 색상 다양성 자동 검증
    analysis = analyze_image_color_diversity(str(out_file.resolve()), verbose=True)
    return analysis

def main():
    parser = argparse.ArgumentParser(description="SDXL-Turbo 이미지 생성 및 색상 다양성 자동 검증")
    parser.add_argument(
        "--prompt",
        default="A vibrant cyberpunk city at night with neon lights, flying vehicles, and colorful reflections in the rain, cinematic photorealistic 8k",
        help="생성할 이미지 프롬프트"
    )
    parser.add_argument(
        "-o", "--output",
        default="generated_image.png",
        help="출력 파일 경로 (기본: generated_image.png)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1,
        help="추론 스텝 (SDXL-Turbo 권장: 1~2, 기본: 1)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="랜덤 시드 (기본: 42)"
    )

    args = parser.parse_args()
    generate_and_verify(
        prompt=args.prompt,
        output_path=args.output,
        steps=args.steps,
        seed=args.seed
    )

if __name__ == "__main__":
    main()


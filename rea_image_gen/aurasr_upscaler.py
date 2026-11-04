import os
import json
import torch
from pathlib import Path
from PIL import Image
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from aura_sr import AuraSR

_aura_sr_instance = None

def load_aurasr_model(device: str = "cpu", model_id: str = "fal/AuraSR-v2") -> AuraSR:
    """
    fal/AuraSR-v2 초해상화(Upscaler) 모델을 로드하여 싱글톤 인스턴스로 반환합니다.
    """
    global _aura_sr_instance
    if _aura_sr_instance is not None:
        return _aura_sr_instance

    print(f"\n[AuraSR-v2] 모델 로딩 중: {model_id} (Device: {device})...")
    hf_model_path = Path(snapshot_download(model_id))
    config = json.loads((hf_model_path / "config.json").read_text())
    
    model = AuraSR(config, device=device)
    checkpoint = load_file(hf_model_path / "model.safetensors")
    model.upsampler.load_state_dict(checkpoint, strict=True)
    print(" > ✅ fal/AuraSR-v2 업스케일러 모델 로드 완료!")
    
    _aura_sr_instance = model
    return _aura_sr_instance


def upscale_image_with_aurasr(image: Image.Image, max_batch_size: int = 8) -> Image.Image:
    """
    입력 PIL 이미지를 fal/AuraSR-v2를 통해 4배 고해상도(Super-Resolution)로 업스케일합니다.
    (AuraSR 최소 타일 크기 64x64 보정 지원)
    """
    sr = load_aurasr_model()
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # 64x64 미만인 경우 최소 크기(64x64)로 리사이즈
    w, h = image.size
    if w < 64 or h < 64:
        new_w = max(64, w)
        new_h = max(64, h)
        image = image.resize((new_w, new_h), Image.Resampling.BICUBIC)

    return sr.upscale_4x_overlapped(image, max_batch_size=max_batch_size)


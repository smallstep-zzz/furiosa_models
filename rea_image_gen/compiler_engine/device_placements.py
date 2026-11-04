"""Device normalization for model operations that create helper tensors on CPU."""

import torch


def install_qwen_rotary_device_fix():
    """Move Qwen3-VL rotary factors to Furiosa before Dynamo traces mul."""
    from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen3_vl

    original = qwen3_vl.apply_rotary_pos_emb
    if getattr(original, "_furiosa_device_safe", False):
        return

    def device_safe_apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
        device = torch.device("furiosa:0") if torch._dynamo.is_compiling() else q.device
        k = k.to(device)
        cos = cos.to(device)
        sin = sin.to(device)
        return original(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)

    device_safe_apply_rotary_pos_emb._furiosa_device_safe = True
    qwen3_vl.apply_rotary_pos_emb = device_safe_apply_rotary_pos_emb

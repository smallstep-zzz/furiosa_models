"""Small FX rewrites used before handing Diffusers graphs to Furiosa Torch."""

import os
import torch
import torch.nn.functional as F


_FUSED_SDPA_TARGET = F.scaled_dot_product_attention
_ORIGINAL_TORCH_CAT = torch.cat
_ORIGINAL_TORCH_CONCAT = torch.concat
_ORIGINAL_GROUP_NORM = F.group_norm


def dtype_promoting_cat(tensors, dim=0, *, out=None):
    tensors = tuple(tensors)
    if tensors:
        promoted_dtype = tensors[0].dtype
        for tensor in tensors[1:]:
            promoted_dtype = torch.promote_types(promoted_dtype, tensor.dtype)
        if any(tensor.dtype != promoted_dtype for tensor in tensors):
            tensors = tuple(
                tensor.to(dtype=promoted_dtype)
                if tensor.dtype != promoted_dtype else tensor
                for tensor in tensors
            )
    if out is None:
        return _ORIGINAL_TORCH_CAT(tensors, dim=dim)
    return _ORIGINAL_TORCH_CAT(tensors, dim=dim, out=out)


def _dispatch_cat(tensors, dim=0, *, out=None):
    from .operator_compat import run_operator
    return run_operator("aten.cat", tensors, dim=dim, out=out)


def install_dtype_promoting_cat():
    """Match PyTorch cat dtype promotion before Furiosa's same-dtype-only kernel."""
    torch.cat = _dispatch_cat
    torch.concat = _dispatch_cat




def device_safe_scaled_dot_product_attention(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False,
    scale=None, enable_gqa=False,
):
    """Normalize SDPA operands before Dynamo performs FakeTensor propagation."""
    if key.device != query.device:
        key = key.to(query.device)
    if value.device != query.device:
        value = value.to(query.device)
    if isinstance(attn_mask, torch.Tensor) and attn_mask.device != query.device:
        attn_mask = attn_mask.to(query.device)
    return _FUSED_SDPA_TARGET(
        query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
        is_causal=is_causal, scale=scale, enable_gqa=enable_gqa,
    )


def install_device_safe_sdpa():
    """Normalize cross-device mask inputs before Transformers calls fused SDPA."""
    if F.scaled_dot_product_attention is not device_safe_scaled_dot_product_attention:
        F.scaled_dot_product_attention = device_safe_scaled_dot_product_attention


class GroupNormMeanSquare(torch.nn.Module):
    """Public PyTorch GroupNorm decomposition that avoids var_mean lowering."""

    def __init__(self, source):
        super().__init__()
        self.num_groups = source.num_groups
        self.num_channels = source.num_channels
        self.eps = source.eps
        self.weight = source.weight
        self.bias = source.bias

    def forward(self, value):
        return group_norm_via_mean_square(
            value, self.num_groups, self.weight, self.bias, self.eps
        )


def group_norm_via_mean_square(value, num_groups, weight=None, bias=None, eps=1e-5):
    """Public PyTorch GroupNorm decomposition using mean-square variance."""
    if value.ndim < 2 or value.shape[1] % num_groups:
        raise ValueError("GroupNorm channel count must be divisible by num_groups")
    batch, channels = value.shape[:2]
    if weight is not None and weight.numel() != channels:
        raise ValueError("GroupNorm weight size must equal the channel count")
    if bias is not None and bias.numel() != channels:
        raise ValueError("GroupNorm bias size must equal the channel count")
    grouped = value.float().reshape(batch, num_groups, -1)
    mean = grouped.mean(dim=-1, keepdim=True)
    centered = grouped - mean
    variance = (centered * centered).mean(dim=-1, keepdim=True)
    normalized = (centered * torch.rsqrt(variance + eps)).reshape(value.shape)
    affine_shape = [1, channels] + [1] * (value.ndim - 2)
    if weight is not None:
        normalized = normalized * weight.float().reshape(affine_shape)
    if bias is not None:
        normalized = normalized + bias.float().reshape(affine_shape)
    return normalized.to(value.dtype)


def install_group_norm_decomposition():
    """Route GroupNorm through public tensor ops before Dynamo captures it."""
    if F.group_norm is not group_norm_via_mean_square:
        F.group_norm = group_norm_via_mean_square


def manual_scaled_dot_product_attention(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
):
    """Decompose SDPA into matmul, mask, softmax, and matmul primitives."""
    if enable_gqa:
        if query.shape[-3] % key.shape[-3]:
            raise ValueError("query heads must be divisible by key/value heads for GQA")
        repeats = query.shape[-3] // key.shape[-3]
        key = key.repeat_interleave(repeats, dim=-3)
        value = value.repeat_interleave(repeats, dim=-3)

    output_dtype = query.dtype
    # Match PyTorch math SDPA: low-precision inputs use FP32 intermediates.
    if os.environ.get("FURIOSA_SDPA_FP32", "0") == "1" and query.dtype in (torch.bfloat16, torch.float16):
        query, key, value = query.float(), key.float(), value.float()
    scale_factor = scale if scale is not None else query.shape[-1] ** -0.5
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale_factor
    if is_causal:
        if attn_mask is not None:
            raise ValueError("scaled_dot_product_attention cannot combine is_causal and attn_mask")
        q_pos = torch.arange(query.shape[-2], device=query.device).unsqueeze(-1)
        k_pos = torch.arange(key.shape[-2], device=query.device).unsqueeze(0)
        scores = torch.where(k_pos <= q_pos, scores, float("-inf"))
    elif attn_mask is not None:
        if isinstance(attn_mask, torch.Tensor) and attn_mask.device != query.device:
            attn_mask = attn_mask.to(query.device)
        if attn_mask.dtype == torch.bool:
            scores = scores.masked_fill(~attn_mask, float("-inf"))
        else:
            scores = scores + attn_mask

    probabilities = torch.softmax(scores, dim=-1)
    if dropout_p:
        probabilities = F.dropout(probabilities, p=dropout_p, training=True)
    return torch.matmul(probabilities, value).to(output_dtype)




def _pair(value):
    return value if isinstance(value, tuple) else (value, value)


def conv2d_via_unfold(value, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    # Public PyTorch decomposition for non-dilated, ungrouped conv2d.
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)
    if groups != 1 or dilation != (1, 1):
        raise NotImplementedError("decomposition currently supports groups=1 and dilation=1")
    if stride[0] < 1 or stride[1] < 1:
        raise ValueError("stride must be positive")
    padded = F.pad(value, (padding[1], padding[1], padding[0], padding[0]))
    windows = padded.unfold(2, weight.shape[2], stride[0]).unfold(
        3, weight.shape[3], stride[1]
    )
    batch, channels, height, width, kernel_h, kernel_w = windows.shape
    columns = windows.permute(0, 1, 4, 5, 2, 3).reshape(
        batch, channels * kernel_h * kernel_w, height * width
    )
    output = torch.matmul(weight.reshape(weight.shape[0], -1), columns)
    if bias is not None:
        output = output + bias.reshape(1, -1, 1)
    return output.reshape(batch, weight.shape[0], height, width)


def conv2d_via_spatial_matmul(value, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """Express an ungrouped conv as explicit spatial slices and small channel matmuls."""
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)
    if groups != 1 or dilation != (1, 1):
        raise NotImplementedError("decomposition currently supports groups=1 and dilation=1")
    if min(stride) < 1:
        raise ValueError("stride must be positive")
    batch, channels, height, width = value.shape
    output_channels, weight_channels, kernel_h, kernel_w = weight.shape
    if channels != weight_channels:
        raise ValueError("input and filter channel counts do not match")
    output_h = (height + 2 * padding[0] - kernel_h) // stride[0] + 1
    output_w = (width + 2 * padding[1] - kernel_w) // stride[1] + 1
    padded = F.pad(value, (padding[1], padding[1], padding[0], padding[0]))
    columns = []
    for kh in range(kernel_h):
        for kw in range(kernel_w):
            window = padded[:, :, kh:kh + output_h * stride[0]:stride[0],
                            kw:kw + output_w * stride[1]:stride[1]]
            matrix = window.permute(0, 2, 3, 1).reshape(batch * output_h * output_w, channels)
            columns.append(matrix)
    columns = torch.cat(columns, dim=1)
    filters = weight.permute(0, 2, 3, 1).reshape(output_channels, -1).transpose(0, 1)
    output = torch.matmul(columns, filters)
    if bias is not None:
        output = output + bias.reshape(1, output_channels)
    return output.reshape(batch, output_h, output_w, output_channels).permute(0, 3, 1, 2)


def _supports_sdxl_conv_decomposition(value, weight, bias=None, stride=1, padding=1, dilation=1, groups=1):
    return (
        isinstance(value, torch.Tensor)
        and isinstance(weight, torch.Tensor)
        and tuple(value.shape) == (1, 4, 64, 64)
        and tuple(weight.shape) == (320, 4, 3, 3)
        and (bias is None or tuple(bias.shape) == (320,))
        and value.dtype == weight.dtype == torch.bfloat16
        and (bias is None or bias.dtype == value.dtype)
        and _pair(stride) == (1, 1)
        and _pair(padding) == (1, 1)
        and _pair(dilation) == (1, 1)
        and groups == 1
    )


def lower_scaled_dot_product_attention(graph_module):
    """Replace captured fused SDPA calls with explicit primitive operations."""
    from .operator_compat import resolve_operator
    replacement = resolve_operator("aten.scaled_dot_product_attention").implementation
    target = _FUSED_SDPA_TARGET
    count = 0
    for node in graph_module.graph.nodes:
        if node.op == "call_function" and node.target is target:
            node.target = replacement
            count += 1
    if count:
        graph_module.graph.lint()
        graph_module.recompile()
    return count


# Register the already-tested public PyTorch decompositions for explicit use.
from .operator_compat import register_operator

register_operator("aten.cat", "python_decomposition", dtype_promoting_cat)
register_operator(
    "aten.scaled_dot_product_attention",
    "python_decomposition",
    manual_scaled_dot_product_attention,
)


register_operator(
    "conv2d",
    "python_decomposition",
    conv2d_via_spatial_matmul,
    supports=_supports_sdxl_conv_decomposition,
)
register_operator(
    "conv2d",
    "python_decomposition",
    conv2d_via_unfold,
    supports=lambda value, weight, bias=None, stride=1, padding=0, dilation=1, groups=1:
        not _supports_sdxl_conv_decomposition(value, weight, bias, stride, padding, dilation, groups),
)

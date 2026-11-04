"""Public FP32 Conv2d decomposition; device support requires separate validation."""
import torch
from torch import nn
from torch.nn import functional as F


class SpatialLinearConv2d(nn.Module):
    """Pack spatial taps on the input device and contract with one Linear."""

    def __init__(self, conv: nn.Conv2d):
        super().__init__()
        if conv.dilation != (1, 1) or conv.padding_mode != "zeros":
            raise ValueError("Requires dilation=1 and zero padding")
        if not isinstance(conv.padding, tuple) or conv.weight.dtype != torch.float32:
            raise ValueError("Requires integer padding and FP32 weights")
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.stride = conv.stride
        self.padding = conv.padding
        inner = conv.in_channels * conv.kernel_size[0] * conv.kernel_size[1]
        self.inner_padding = (-inner) % 32
        # Tap order matches the channel-last concatenation in forward.
        weight = conv.weight.detach()
        if conv.groups != 1:
            dense = weight.new_zeros(conv.out_channels, conv.in_channels, *conv.kernel_size)
            ins, outs = conv.in_channels//conv.groups, conv.out_channels//conv.groups
            for group in range(conv.groups):
                dense[group*outs:(group+1)*outs, group*ins:(group+1)*ins] = weight[group*outs:(group+1)*outs]
            weight = dense
        weight = weight.permute(0, 2, 3, 1).reshape(conv.out_channels, inner)
        self.linear = nn.Linear(inner + self.inner_padding, conv.out_channels,
                                bias=conv.bias is not None, device=weight.device)
        self.linear.weight = nn.Parameter(F.pad(weight, (0, self.inner_padding)).contiguous(),
                                          requires_grad=False)
        self.linear.bias = (None if conv.bias is None else
                            nn.Parameter(conv.bias.detach().clone(), requires_grad=False))

    def forward(self, value):
        packed = self.pack(value)
        return self.unpack(self.linear(packed), value.shape)

    def pack(self, value):
        if value.ndim != 4 or value.shape[1] != self.in_channels or value.dtype != torch.float32:
            raise ValueError("Requires FP32 NCHW input with matching channels")
        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding
        oh = (value.shape[-2] + 2 * ph - kh) // sh + 1
        ow = (value.shape[-1] + 2 * pw - kw) // sw + 1
        # NCHW constant_pad_nd fails SDK group-offset planning at C=16/H=W=320.
        padded = value
        if pw:
            zeros = value.new_zeros(*value.shape[:-1], pw)
            padded = torch.cat((zeros, padded, zeros), dim=-1)
        if ph:
            zeros = value.new_zeros(*padded.shape[:-2], ph, padded.shape[-1])
            padded = torch.cat((zeros, padded, zeros), dim=-2)
        taps = [padded[:, :, y:y + oh * sh:sh, x:x + ow * sw:sw].permute(0, 2, 3, 1)
                for y in range(kh) for x in range(kw)]
        packed = torch.cat(taps, dim=-1)
        packed = F.pad(packed, (0, self.inner_padding))
        return packed.reshape(-1, self.linear.in_features)

    def unpack(self, result, shape):
        oh = (shape[-2] + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        ow = (shape[-1] + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1
        return result.reshape(shape[0], oh, ow, self.out_channels).permute(0, 3, 1, 2).contiguous()

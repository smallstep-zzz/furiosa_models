"""Exact stride-one MaxPool using resident padding and pairwise maxima."""
import torch


class ResidentMaxPool(torch.nn.Module):
    def __init__(self, original):
        super().__init__()
        def pair(value):
            return (value, value) if isinstance(value, int) else tuple(value)
        self.kernel = pair(original.kernel_size)
        self.padding = pair(original.padding)
        if pair(original.stride) != (1, 1) or pair(original.dilation) != (1, 1) or original.ceil_mode or original.return_indices:
            raise ValueError('Only stride-one, dilation-one, no-index MaxPool is validated')
        if any(p > k//2 for p, k in zip(self.padding, self.kernel)):
            raise ValueError('Each output window must overlap the original input')

    def forward(self, value):
        ph, pw = self.padding
        h, w = value.shape[-2:]
        padded = value
        # Replicating an already-included border value preserves each window's max.
        if pw:
            padded = torch.cat([value[..., :1]]*pw + [padded] + [value[..., -1:]]*pw, dim=-1)
        if ph:
            padded = torch.cat([padded[..., :1, :]]*ph + [padded] + [padded[..., -1:, :]]*ph, dim=-2)
        oh, ow = h+2*ph-self.kernel[0]+1, w+2*pw-self.kernel[1]+1
        result = padded[..., :oh, :ow].contiguous()
        for y in range(self.kernel[0]):
            for x in range(self.kernel[1]):
                if y or x:
                    result = torch.maximum(result, padded[..., y:y+oh, x:x+ow].contiguous())
        return result

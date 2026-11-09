"""
CSFPN: Compact Scalable Feature Pyramid Network
Lightweight neck using Partial Conv Upsampling Block (PCUB) and
Lightweight Multi-scale Conv Block (LMCB) with Softmax BiFPN fusion.

Based on:
- EMCAD (2024): MSCB + EUCB for efficient medical image segmentation
- FasterNet (CVPR 2023): Partial Convolution for sparse efficient processing
- BiFPN (CVPR 2020): Weighted bidirectional feature pyramid network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.conv import Conv, DWConv, autopad


__all__ = ['CSFPN', 'CSFPNBlock']


def channel_shuffle(x, groups):
    """Channel shuffle: [N, C, H, W] → [N, groups, C/groups, H, W] → permute → [N, C, H, W]"""
    B, C, H, W = x.shape
    return x.view(B, groups, C // groups, H, W).transpose(1, 2).contiguous().view(B, C, H, W)


class PartialConv(nn.Module):
    """Partial Convolution: only processes first C/4 channels, rest pass through."""

    def __init__(self, dim):
        super().__init__()
        self.partial_dim = max(dim // 4, 1)
        self.conv = nn.Conv2d(self.partial_dim, self.partial_dim, 3, padding=1, bias=False)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.partial_dim, x.shape[1] - self.partial_dim], dim=1)
        return torch.cat([self.conv(x1), x2], dim=1)


class PCUB(nn.Module):
    """Partial Conv Upsampling Block.

    Bilinear upsample → Partial Conv (only C/4 channels) → 1×1 pointwise conv.
    Reduces upsampling computation by ~75% compared to full-channel DWConv.
    """

    def __init__(self, c1, c2):
        super().__init__()
        self.pconv = PartialConv(c1)
        self.pwconv = Conv(c1, c2, 1, 1)

    def forward(self, x, target_size=None):
        if target_size is not None:
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        else:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.pconv(x)
        return self.pwconv(x)


class MSDC(nn.Module):
    """Multi-Scale Depthwise Convolution: 1×1 + 3×3 DWConv, concatenated."""

    def __init__(self, channels):
        super().__init__()
        self.dw1 = nn.Conv2d(channels, channels, 1, groups=channels, bias=False)
        self.dw3 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)

    def forward(self, x):
        return torch.cat([self.dw1(x), self.dw3(x)], dim=1)


class PartialShuffle(nn.Module):
    """Partial Channel Shuffle: shuffles only first C/2 channels, rest unchanged."""

    def forward(self, x):
        C = x.shape[1]
        x1, x2 = x.chunk(2, dim=1)
        x1 = channel_shuffle(x1, groups=4)
        return torch.cat([x1, x2], dim=1)


class LMCB(nn.Module):
    """Lightweight Multi-scale Conv Block.

    1×1 expand → MSDC(1×1+3×3 DW) → Partial Shuffle → 1×1 compress + residual.
    """

    def __init__(self, c1, c2, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.expand = Conv(c1, c_ * 2, 1, 1)  # expand for MSDC (doubles via concat)
        self.msdc = MSDC(c_ * 2)  # outputs 4*c_
        self.pshuffle = PartialShuffle()
        self.compress = Conv(c_ * 4, c2, 1, 1)
        self.shortcut = Conv(c1, c2, 1, 1) if c1 != c2 else nn.Identity()

    def forward(self, x):
        out = self.expand(x)
        out = self.msdc(out)
        out = self.pshuffle(out)
        out = self.compress(out)
        return out + self.shortcut(x)


class CSP_LMCB(nn.Module):
    """CSP-style LMCB: splits input, processes one half through LMCB, concats with other half."""

    def __init__(self, c1, c2, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.lmcb = LMCB(c_, c_, e=1.0)
        self.cv3 = Conv(c_ * 2, c2, 1, 1)

    def forward(self, x):
        x1 = self.lmcb(self.cv1(x))
        x2 = self.cv2(x)
        return self.cv3(torch.cat([x1, x2], dim=1))


class CSFPNBlock(nn.Module):
    """CSFPN processing block with LMCB (VoV-style CSP)."""

    def __init__(self, c1, c2, shortcut=True, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.lmcb = LMCB(c_, c_, e=1.0)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)

    def forward(self, x):
        x1 = self.lmcb(self.cv1(x))
        x2 = self.cv2(x)
        return self.cv3(torch.cat((x2, x1), dim=1))


class CSFPN(nn.Module):
    """CSFPN: Compact Shuffle FPN module (C2f-style wrapper).

    Lightweight neck replacement using LMCB + VoVCSP.
    For the full BiFPN topology with Softmax fusion, use PCUB + CSP_LMCB directly
    in the model neck configuration."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(CSFPNBlock(self.c, self.c, shortcut, e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

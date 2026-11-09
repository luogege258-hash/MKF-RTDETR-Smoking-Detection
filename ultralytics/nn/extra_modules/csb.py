"""
CSB: Calibrated Star Block
Lightweight backbone block using star operations (element-wise multiplication of
two 1x1 conv branches) with SE-based channel calibration and asymmetric DWConv.

Based on:
- StarNet (2024): Star operation for efficient nonlinear feature interaction
- SE-Net (CVPR 2018): Squeeze-and-Excitation channel attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.conv import Conv, autopad


__all__ = ['CSB', 'CSBBlock', 'C2f_CSB']


class StarOperation(nn.Module):
    """Star operation: SiLU(f1(x)) ⊙ f2(x) with channel expansion.

    Two parallel 1×1 convolutions project input to expanded channels,
    then element-wise multiplication produces rich nonlinear interaction.
    """

    def __init__(self, dim_in, expansion=3):
        super().__init__()
        dim_expanded = dim_in * expansion
        self.f1 = nn.Conv2d(dim_in, dim_expanded, 1, bias=False)
        self.f2 = nn.Conv2d(dim_in, dim_expanded, 1, bias=False)

    def forward(self, x):
        return F.silu(self.f1(x)) * self.f2(x)


class SE_Light(nn.Module):
    """Lightweight Squeeze-and-Excitation for channel calibration.

    GAP → FC(C'/r) → ReLU → FC(C') → Sigmoid, with reduction ratio r=16.
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        inner = max(channels // reduction, 8)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, inner, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(inner, channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        s = self.gap(x)
        s = self.fc2(self.relu(self.fc1(s)))
        return x * self.sigmoid(s)


class CSBBlock(nn.Module):
    """CSB bottleneck: DWConv 5×5 → Star Op → SE_light → 1×1 down → DWConv 3×3 + residual.

    Asymmetric DWConv design:
      - Pre-star 5×5: retains moderate context for small cigarette/hand cues
      - Post-star 3×3: lightweight refinement after high-order interaction
    """

    def __init__(self, c1, c2, shortcut=True, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)

        # Pre-star DWConv 5×5
        self.pre_dw = nn.Sequential(
            nn.Conv2d(c_, c_, 5, padding=2, groups=c_, bias=False),
            nn.BatchNorm2d(c_),
            nn.SiLU(inplace=True)
        )

        # Star operation with expansion ratio α=3
        self.star = StarOperation(c_, expansion=3)

        # SE_light calibration (input channels = c_ * 3)
        self.se_light = SE_Light(c_ * 3, reduction=16)

        # 1×1 down-dim from expanded (c_*3) to c_
        self.down_conv = Conv(c_ * 3, c_, 1, 1)

        # Post-star DWConv 3×3
        self.post_dw = nn.Sequential(
            nn.Conv2d(c_, c_, 3, padding=1, groups=c_, bias=False),
            nn.BatchNorm2d(c_),
            nn.SiLU(inplace=True)
        )

        self.cv2 = Conv(c_, c2, 1, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        residual = x
        out = self.cv1(x)
        out = self.pre_dw(out)
        out = self.star(out)
        out = self.se_light(out)
        out = self.down_conv(out)
        out = self.post_dw(out)
        out = self.cv2(out)
        return out + residual if self.add else out


class CSB(nn.Module):
    """CSB: Calibrated Star Block (C2f-style wrapper).
    Lightweight backbone replacement using star operations."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(CSBBlock(self.c, self.c, shortcut, e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C2f_CSB(nn.Module):
    """C2f variant with CSB blocks for lightweight backbone."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(CSBBlock(self.c, self.c, shortcut, e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

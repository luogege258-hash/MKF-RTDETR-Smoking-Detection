"""AMKF modules aligned with manuscript equations (1)--(7).

The formula-defined AMKF transform expands C to C'=alpha*C, performs five
depthwise-separable branches, applies kernel-selection/dual-path attention and
channel calibration, adds the projected residual, and finally projects back to
C. ``AMKF`` wraps this transform in the C2f stage used by the paper.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.conv import Conv


__all__ = ["AMKF", "C2f_AMKF", "AMKFBlock", "AMKFModule"]


class DepthwiseSeparableBranch(nn.Module):
    """Depthwise + pointwise convolution followed once by BN and SiLU."""

    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size,
                padding=kernel_size // 2,
                groups=channels,
                bias=False,
            ),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class DualPathAttention(nn.Module):
    """Spatial and channel paths from manuscript equations (3)--(5)."""

    def __init__(self, channels: int, reduction: int = 16, strip_kernel: int = 11) -> None:
        super().__init__()
        squeeze = max(4, channels // reduction)
        spatial_hidden = max(4, channels // 2)
        self.spatial_conv1 = nn.Conv2d(1, spatial_hidden, 1, bias=False)
        self.spatial_dw_h = nn.Conv2d(
            spatial_hidden,
            spatial_hidden,
            (strip_kernel, 1),
            padding=(strip_kernel // 2, 0),
            groups=spatial_hidden,
            bias=False,
        )
        self.spatial_dw_w = nn.Conv2d(
            spatial_hidden,
            spatial_hidden,
            (1, strip_kernel),
            padding=(0, strip_kernel // 2),
            groups=spatial_hidden,
            bias=False,
        )
        self.spatial_conv2 = nn.Conv2d(spatial_hidden, channels, 1)
        self.channel_fc1 = nn.Conv2d(channels, squeeze, 1)
        self.channel_fc2 = nn.Conv2d(squeeze, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial = x.mean(dim=1, keepdim=True)
        spatial = self.spatial_conv2(self.spatial_dw_w(self.spatial_dw_h(self.spatial_conv1(spatial))))
        spatial = torch.sigmoid(spatial)
        channel = F.adaptive_avg_pool2d(x, 1)
        channel = torch.sigmoid(self.channel_fc2(F.relu(self.channel_fc1(channel), inplace=True)))
        return x * spatial * channel


class ChannelCalibration(nn.Module):
    """Sigmoid channel calibration from equations (6)--(7)."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        squeeze = max(4, channels // reduction)
        self.fc1 = nn.Conv2d(channels, squeeze, 1)
        self.fc2 = nn.Conv2d(squeeze, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = F.adaptive_avg_pool2d(x, 1)
        scale = torch.sigmoid(self.fc2(F.relu(self.fc1(scale), inplace=True)))
        return x * scale


class AMKFBlock(nn.Module):
    """Formula-defined AMKF transform with equal input/output channels."""

    def __init__(
        self,
        channels: int,
        expansion: float = 2.0,
        reduction: int = 16,
        kernels: Sequence[int] = (3, 5, 7, 9, 11),
        strip_kernel: int = 11,
    ) -> None:
        super().__init__()
        hidden = max(8, int(channels * expansion))
        squeeze = max(4, hidden // reduction)
        self.hidden = hidden
        self.kernels = tuple(kernels)

        self.expand = nn.Conv2d(channels, hidden, 1, bias=False)
        self.branches = nn.ModuleList(DepthwiseSeparableBranch(hidden, k) for k in self.kernels)
        self.kernel_fc1 = nn.Linear(hidden, squeeze)
        self.kernel_fc2 = nn.Linear(squeeze, len(self.kernels) * hidden)
        self.fuse_projection = nn.Conv2d(hidden, hidden, 1, bias=False)
        self.dual_attention = DualPathAttention(hidden, reduction, strip_kernel)
        self.calibration = ChannelCalibration(hidden, reduction)
        self.residual_projection = nn.Conv2d(channels, hidden, 1, bias=False)
        self.output_projection = nn.Conv2d(hidden, channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        expanded = self.expand(x)
        stacked = torch.stack([branch(expanded) for branch in self.branches], dim=1)
        # Stack K branches and globally average to obtain the channel descriptor z.
        descriptor = stacked.mean(dim=1).mean(dim=(2, 3))
        weights = self.kernel_fc2(F.relu(self.kernel_fc1(descriptor), inplace=True))
        weights = weights.view(x.shape[0], len(self.kernels), self.hidden, 1, 1).softmax(dim=1)
        fused = self.fuse_projection((stacked * weights).sum(dim=1))
        calibrated = self.calibration(self.dual_attention(fused))
        return self.output_projection(calibrated + self.residual_projection(x))


class AMKFModule(AMKFBlock):
    """Backward-compatible name for the equation-aligned AMKF transform."""


class AMKF(nn.Module):
    """C2f stage whose repeated internal transforms are equation-aligned AMKF blocks."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        super().__init__()
        del shortcut, g  # Kept in the signature for Ultralytics YAML compatibility.
        self.c = max(1, int(c2 * e))
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1, 1)
        self.m = nn.ModuleList(AMKFBlock(self.c) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = list(self.cv1(x).chunk(2, dim=1))
        features.extend(block(features[-1]) for block in self.m)
        return self.cv2(torch.cat(features, dim=1))


class C2f_AMKF(AMKF):
    """Alias retained for existing configuration files."""


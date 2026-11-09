"""ESRFPN aligned with manuscript equations (11)--(15).

The network consumes S3 (128 channels), S4 (256 channels), and the EPLA output
S5' (256 channels) together. Enhanced PCE therefore operates on the exact
640-channel concatenation specified in the paper, followed by three RCM-ESRCA
branches and top-down/bottom-up Bi-Fuse paths.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ESRFPN", "ESRCA", "BiFuse", "RCM_ESRCA"]


class _ScaleWeightGenerator(nn.Module):
    """GAP-MLP-Softmax scale weighting for one directional branch group."""

    def __init__(self, channels: int, num_scales: int, reduction: int = 4) -> None:
        super().__init__()
        total = channels * num_scales
        hidden = max(8, total // reduction)
        self.channels = channels
        self.num_scales = num_scales
        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(total, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, total, 1),
        )

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        batch = features[0].shape[0]
        logits = self.mlp(torch.cat(tuple(features), dim=1))
        return logits.view(batch, self.num_scales, self.channels, 1, 1).softmax(dim=1)


class ESRCA(nn.Module):
    """Multi-scale rectangular self-calibration attention from section 2.4."""

    def __init__(self, channels: int, kernels: Sequence[int] = (5, 11, 21)) -> None:
        super().__init__()
        self.kernels = tuple(kernels)
        self.vertical = nn.ModuleList(
            nn.Conv2d(
                channels,
                channels,
                (kernel, 1),
                padding=(kernel // 2, 0),
                bias=False,
            )
            for kernel in self.kernels
        )
        self.horizontal = nn.ModuleList(
            nn.Conv2d(
                channels,
                channels,
                (1, kernel),
                padding=(0, kernel // 2),
                bias=False,
            )
            for kernel in self.kernels
        )
        self.vertical_gate = _ScaleWeightGenerator(channels, len(self.kernels))
        self.horizontal_gate = _ScaleWeightGenerator(channels, len(self.kernels))
        self.norm = nn.BatchNorm2d(channels)
        self.local = nn.Conv2d(channels, channels, 5, padding=2, groups=channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height_pool = x.mean(dim=3, keepdim=True)
        width_pool = x.mean(dim=2, keepdim=True)
        vertical_features = [conv(height_pool) for conv in self.vertical]
        horizontal_features = [conv(width_pool) for conv in self.horizontal]
        vertical = (
            torch.stack(vertical_features, dim=1) * self.vertical_gate(vertical_features)
        ).sum(dim=1)
        horizontal = (
            torch.stack(horizontal_features, dim=1) * self.horizontal_gate(horizontal_features)
        ).sum(dim=1)
        spatial_attention = torch.sigmoid(self.norm(vertical + horizontal))
        return self.local(x) * spatial_attention


class _ConvMlp(nn.Module):
    def __init__(self, channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, 1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class RCM_ESRCA(nn.Module):
    """Original RCM sequence with RCA replaced only by the ESRCA token mixer."""

    def __init__(self, channels: int, mlp_ratio: float = 2.0, layer_scale_init: float = 1e-6) -> None:
        super().__init__()
        self.token_mixer = ESRCA(channels)
        self.norm = nn.BatchNorm2d(channels)
        self.mlp = _ConvMlp(channels, max(channels, int(channels * mlp_ratio)))
        self.layer_scale = nn.Parameter(layer_scale_init * torch.ones(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        refined = self.mlp(self.norm(self.token_mixer(x)))
        return x + refined * self.layer_scale.view(1, -1, 1, 1)


class BiFuse(nn.Module):
    """Bidirectional cross-level fusion implementing equations (13)--(15)."""

    def __init__(self, main_channels: int, context_channels: int | None = None) -> None:
        super().__init__()
        context_channels = main_channels if context_channels is None else context_channels
        self.context_gate = nn.Conv2d(context_channels, main_channels, 1)
        self.context_projection = nn.Conv2d(context_channels, main_channels, 1)
        self.main_gate = nn.Conv2d(main_channels, main_channels, 1)

    def forward(self, main: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if context.shape[-2:] != main.shape[-2:]:
            context = F.interpolate(context, size=main.shape[-2:], mode="nearest")
        forward_path = main * torch.sigmoid(self.context_gate(context))
        reverse_path = self.context_projection(context) * torch.sigmoid(self.main_gate(main))
        return forward_path + reverse_path


class ESRFPN(nn.Module):
    """Three-input Enhanced PCE + RCM-ESRCA + Bi-Fuse feature pyramid."""

    def __init__(self, in_channels: Sequence[int], out_channels: int = 256) -> None:
        super().__init__()
        if len(in_channels) != 3:
            raise ValueError("ESRFPN expects exactly three input feature levels")
        self.in_channels = tuple(int(channel) for channel in in_channels)
        self.out_channels = int(out_channels)
        merged_channels = sum(self.in_channels)
        squeeze = max(4, merged_channels // 16)
        self.scale_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(merged_channels, squeeze, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(squeeze, merged_channels, 1),
            nn.Sigmoid(),
        )
        self.context_blocks = nn.ModuleList(RCM_ESRCA(channel) for channel in self.in_channels)
        self.level_projections = nn.ModuleList(
            nn.Identity() if channel == self.out_channels else nn.Conv2d(channel, self.out_channels, 1, bias=False)
            for channel in self.in_channels
        )

        self.top_to_mid = BiFuse(self.out_channels)
        self.mid_to_low = BiFuse(self.out_channels)
        self.down_low = nn.Conv2d(self.out_channels, self.out_channels, 3, stride=2, padding=1, bias=False)
        self.down_mid = nn.Conv2d(self.out_channels, self.out_channels, 3, stride=2, padding=1, bias=False)
        self.low_to_mid = BiFuse(self.out_channels)
        self.mid_to_top = BiFuse(self.out_channels)

    def _enhanced_pce(
        self, features: Sequence[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(features) != 3:
            raise ValueError("ESRFPN expects S3, S4, and S5_prime")
        low, middle, high = features
        expected_channels = self.in_channels
        actual_channels = (low.shape[1], middle.shape[1], high.shape[1])
        if actual_channels != expected_channels:
            raise ValueError(f"expected input channels {expected_channels}, got {actual_channels}")
        target_size = high.shape[-2:]
        concatenated = torch.cat(
            (
                F.adaptive_avg_pool2d(low, target_size),
                F.adaptive_avg_pool2d(middle, target_size),
                high,
            ),
            dim=1,
        )
        weighted = concatenated * self.scale_attention(concatenated)
        parts = weighted.split(self.in_channels, dim=1)
        context = [block(part) for block, part in zip(self.context_blocks, parts)]
        restored = (
            F.interpolate(context[0], size=low.shape[-2:], mode="nearest") + low,
            F.interpolate(context[1], size=middle.shape[-2:], mode="nearest") + middle,
            context[2] + high,
        )
        return tuple(
            projection(feature) for projection, feature in zip(self.level_projections, restored)
        )

    def forward(self, features: Sequence[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c3, c4, c5 = self._enhanced_pce(features)
        p4_top_down = self.top_to_mid(c4, c5)
        p3_out = self.mid_to_low(c3, p4_top_down)
        # In equations (13)--(15), the downsampled low-level feature is F_main
        # and the adjacent high-level feature is F_ctx.
        p4_out = self.low_to_mid(self.down_low(p3_out), p4_top_down)
        p5_out = self.mid_to_top(self.down_mid(p4_out), c5)
        return p3_out, p4_out, p5_out


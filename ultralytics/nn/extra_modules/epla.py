"""EPLA aligned with manuscript equations (8)--(10).

The implementation deliberately follows ``LinearAttn(Q,K,V)=Q(K^T V)``
without an extra denominator, concatenates all four polarity paths, learns a
complex 1-D filter at every sequence-frequency/channel location, supplements
the value tensor with 3/5/7 depthwise-separable branches, and applies the gate
before the final output projection.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["EPLA", "TransformerEncoderLayer_EPLA"]


class _DepthwiseSeparableConv(nn.Module):
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


class PolarizedLinearAttention(nn.Module):
    """Four-path positive/negative linear attention from equation (8)."""

    def __init__(self, dim: int, num_heads: int = 8, polar_power: int = 3) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.polar_power = polar_power
        self.qg_proj = nn.Linear(dim, dim * 2)
        self.kv_proj = nn.Linear(dim, dim * 2)
        # The paper uses the same channel temperature tau for Q and K.
        self.temperature = nn.Parameter(torch.ones(num_heads, self.head_dim))
        self.polar_projection = nn.Linear(dim * 4, dim)

    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        return x.view(batch, length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    @staticmethod
    def _linear_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # Exact manuscript definition: Q(K^T V), with no normalization denominator.
        kv = torch.einsum("bhnd,bhne->bhde", k, v)
        return torch.einsum("bhnd,bhde->bhne", q, kv)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, length, _ = x.shape
        q, gate = self.qg_proj(x).chunk(2, dim=-1)
        k, value = self.kv_proj(x).chunk(2, dim=-1)
        q_heads = self._reshape_heads(q)
        k_heads = self._reshape_heads(k)
        value_heads = self._reshape_heads(value)
        temperature = self.temperature.unsqueeze(0).unsqueeze(2)
        q_pos = F.relu(q_heads * temperature).pow(self.polar_power)
        q_neg = F.relu(-q_heads * temperature).pow(self.polar_power)
        k_pos = F.relu(k_heads * temperature).pow(self.polar_power)
        k_neg = F.relu(-k_heads * temperature).pow(self.polar_power)
        polarized = torch.cat(
            (
                self._linear_attention(q_pos, k_pos, value_heads),
                self._linear_attention(q_pos, k_neg, value_heads),
                self._linear_attention(q_neg, k_pos, value_heads),
                self._linear_attention(q_neg, k_neg, value_heads),
            ),
            dim=-1,
        )
        polarized = polarized.permute(0, 2, 1, 3).reshape(batch, length, self.dim * 4)
        return self.polar_projection(polarized), gate, value


class FrequencyEnhancement(nn.Module):
    """Learnable complex 1-D frequency filter from equations (9)--(10)."""

    def __init__(self, dim: int, frequency_length: int = 625) -> None:
        super().__init__()
        if frequency_length <= 0:
            raise ValueError("frequency_length must be positive")
        self.frequency_length = frequency_length
        frequency_bins = frequency_length // 2 + 1
        self.freq_real = nn.Parameter(torch.ones(frequency_bins, dim))
        self.freq_imag = nn.Parameter(torch.zeros(frequency_bins, dim))
        self.gamma = nn.Parameter(torch.zeros(()))

    def _filter(self, length: int) -> torch.Tensor:
        # fp32 also avoids cuFFT half-precision restrictions for N=625.
        target_bins = length // 2 + 1
        real = self.freq_real.float()
        imag = self.freq_imag.float()
        if target_bins != self.freq_real.shape[0]:
            real = F.interpolate(
                real.transpose(0, 1).unsqueeze(0), size=target_bins, mode="linear", align_corners=False
            ).squeeze(0).transpose(0, 1)
            imag = F.interpolate(
                imag.transpose(0, 1).unsqueeze(0), size=target_bins, mode="linear", align_corners=False
            ).squeeze(0).transpose(0, 1)
        imag = imag.clone()
        imag[0] = 0
        if length % 2 == 0:
            imag[-1] = 0
        return torch.complex(real, imag).unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[1]
        spectrum = torch.fft.rfft(x.float(), dim=1)
        response = torch.fft.irfft(
            spectrum * self._filter(length), n=length, dim=1
        ).to(x.dtype)
        return x + torch.tanh(self.gamma) * response


class MultiScaleLocalSupplement(nn.Module):
    """Restore V spatially and fuse 3/5/7 depthwise-separable branches."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.branches = nn.ModuleList(_DepthwiseSeparableConv(dim, k) for k in (3, 5, 7))
        squeeze = max(4, dim // 16)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, squeeze, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(squeeze, 3 * dim, 1),
        )

    def forward(self, value: torch.Tensor, height: int, width: int) -> torch.Tensor:
        batch, length, channels = value.shape
        if channels != self.dim or height * width != length:
            raise ValueError("V shape and spatial size are inconsistent")
        feature_map = value.transpose(1, 2).reshape(batch, channels, height, width)
        branches = torch.stack([branch(feature_map) for branch in self.branches], dim=1)
        # The manuscript specifies SE-style fusion, so weights use Sigmoid.
        weights = torch.sigmoid(self.se(feature_map)).view(batch, 3, channels, 1, 1)
        return (branches * weights).sum(dim=1).flatten(2).transpose(1, 2)


class EPLA(nn.Module):
    """Equation-aligned EPLA sequence transform."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        polar_power: int = 3,
        frequency_length: int = 625,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.polar_attention = PolarizedLinearAttention(dim, num_heads, polar_power)
        self.frequency_enhancement = FrequencyEnhancement(dim, frequency_length)
        self.local_supplement = MultiScaleLocalSupplement(dim)
        self.output_projection = nn.Linear(dim, dim)

    def forward(
        self,
        x: torch.Tensor,
        height: int | None = None,
        width: int | None = None,
    ) -> torch.Tensor:
        _, length, channels = x.shape
        if channels != self.dim:
            raise ValueError(f"expected {self.dim} channels, got {channels}")
        if height is None or width is None:
            side = int(math.sqrt(length))
            if side * side != length:
                raise ValueError("height and width are required for non-square sequences")
            height = width = side
        global_feature, gate, value = self.polar_attention(x)
        frequency_enhanced = self.frequency_enhancement(global_feature)
        local_feature = self.local_supplement(value, height, width)
        return self.output_projection((frequency_enhanced + local_feature) * torch.sigmoid(gate))


class TransformerEncoderLayer_EPLA(nn.Module):
    """BCHW/BNC adapter for EPLA, without formula-external Transformer layers."""

    def __init__(
        self,
        c1: int,
        cm: int = 2048,
        num_heads: int = 8,
        dropout: float = 0.0,
        act: nn.Module = nn.GELU(),
        normalize_before: bool = False,
        frequency_length: int = 625,
        polar_power: int = 3,
    ) -> None:
        super().__init__()
        # cm/dropout/act/normalize_before remain in the signature only so that
        # existing RT-DETR YAML parser calls stay compatible. The manuscript
        # defines EPLA itself as the AIFI replacement and does not add an outer
        # residual, LayerNorm or FFN around equations (8)--(10).
        del cm, dropout, act, normalize_before
        self.epla = EPLA(
            c1,
            num_heads=num_heads,
            polar_power=polar_power,
            frequency_length=frequency_length,
        )

    def forward(self, x: torch.Tensor, src_mask=None, src_key_padding_mask=None, pos=None) -> torch.Tensor:
        del src_mask, src_key_padding_mask, pos  # Not part of the manuscript's EPLA equations.
        batch, channels, height, width = x.shape
        src = x.flatten(2).transpose(1, 2)
        src = self.epla(src, height, width)
        return src.transpose(1, 2).reshape(batch, channels, height, width).contiguous()


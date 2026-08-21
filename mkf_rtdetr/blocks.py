"""MKF-RT-DETR 第三章的核心网络模块。

模块实现对应论文 3.2 节：AMKF、EPLA 和 ESRFPN。它们保持 RT-DETR
常用的 BCHW 特征图接口，ESRFPN 输出统一 256 通道的 P3/P4/P5。
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _conv_bn_act(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 1,
    stride: int = 1,
    groups: int = 1,
    activation: bool = True,
) -> nn.Sequential:
    padding = kernel_size // 2
    layers: list[nn.Module] = [
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
        nn.BatchNorm2d(out_channels),
    ]
    if activation:
        layers.append(nn.SiLU(inplace=True))
    return nn.Sequential(*layers)


class DepthwiseSeparableConv(nn.Module):
    """深度卷积加逐点卷积，用于多尺度核和局部细化。"""

    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            _conv_bn_act(channels, channels, kernel_size, groups=channels),
            _conv_bn_act(channels, channels, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class AMKFBlock(nn.Module):
    """自适应多尺度核融合块（论文 3.2.1）。

    五组深度可分离卷积通过逐通道核选择注意力融合，随后进行空间/通道双路
    注意力和通道校准残差。输入、输出保持相同的通道数和空间尺寸。
    """

    def __init__(
        self,
        channels: int,
        expansion: float = 2.0,
        reduction: int = 16,
        kernels: Sequence[int] = (3, 5, 7, 9, 11),
        stripe_kernel: int = 11,
    ) -> None:
        super().__init__()
        hidden = max(8, int(channels * expansion))
        squeeze = max(4, hidden // reduction)
        self.hidden = hidden
        self.kernels = tuple(kernels)

        self.expand = _conv_bn_act(channels, hidden, 1)
        self.kernel_branches = nn.ModuleList(
            [DepthwiseSeparableConv(hidden, kernel) for kernel in self.kernels]
        )
        self.kernel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden, squeeze, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(squeeze, len(self.kernels) * hidden, 1),
        )

        spatial_hidden = max(4, hidden // 2)
        self.spatial_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d((None, 1)),
            nn.Conv2d(hidden, spatial_hidden, 1, bias=False),
            nn.BatchNorm2d(spatial_hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                spatial_hidden,
                spatial_hidden,
                (stripe_kernel, 1),
                padding=(stripe_kernel // 2, 0),
                groups=spatial_hidden,
                bias=False,
            ),
            nn.Conv2d(
                spatial_hidden,
                spatial_hidden,
                (1, stripe_kernel),
                padding=(0, stripe_kernel // 2),
                groups=spatial_hidden,
                bias=False,
            ),
            nn.Conv2d(spatial_hidden, hidden, 1),
            nn.Sigmoid(),
        )
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden, squeeze, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(squeeze, hidden, 1),
            nn.Sigmoid(),
        )
        self.calibration = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden, squeeze, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(squeeze, hidden, 1),
            nn.Sigmoid(),
        )
        self.shortcut = _conv_bn_act(channels, hidden, 1, activation=False)
        self.project = _conv_bn_act(hidden, channels, 1, activation=False)
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        expanded = self.expand(x)
        branch_features = torch.stack([branch(expanded) for branch in self.kernel_branches], dim=1)
        batch = x.shape[0]
        weights = self.kernel_gate(branch_features.mean(dim=1))
        weights = weights.view(batch, len(self.kernels), self.hidden, 1, 1).softmax(dim=1)
        fused = (branch_features * weights).sum(dim=1)
        attended = fused * self.spatial_attention(fused) * self.channel_attention(fused)
        calibrated = attended * self.calibration(attended)
        return self.activation(self.project(calibrated + self.shortcut(x)))


class EPLA(nn.Module):
    """增强型极化线性注意力（论文 3.2.2）。

    输入为 BxNxC 序列。正负极化的四路线性注意力避免构造 NxN 注意力矩阵；
    频率增强和多尺度局部卷积分支补充全局注意力的局部细节。
    """

    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 8,
        power: int = 3,
        local_kernels: Sequence[int] = (3, 5, 7),
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.power = power
        self.qg_proj = nn.Linear(dim, dim * 2)
        self.kv_proj = nn.Linear(dim, dim * 2)
        self.out_proj = nn.Linear(dim * 2, dim)
        self.temperature_q = nn.Parameter(torch.ones(num_heads, self.head_dim))
        self.temperature_k = nn.Parameter(torch.ones(num_heads, self.head_dim))
        self.freq_real = nn.Parameter(torch.ones(dim))
        self.freq_imag = nn.Parameter(torch.zeros(dim))
        self.freq_gate = nn.Parameter(torch.zeros(()))
        self.local_branches = nn.ModuleList(
            [DepthwiseSeparableConv(dim, kernel) for kernel in local_kernels]
        )
        squeeze = max(4, dim // 16)
        self.local_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, squeeze, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(squeeze, len(local_kernels) * dim, 1),
        )

    def _reshape_heads(self, x: Tensor) -> Tensor:
        batch, length, _ = x.shape
        return x.view(batch, length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    @staticmethod
    def _linear_attention(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # q/k: B,H,N,D; v: B,H,N,D. The denominator is the normalized linear attention factor.
        kv = torch.einsum("bhnd,bhne->bhde", k, v)
        k_sum = k.sum(dim=2)
        denominator = torch.einsum("bhnd,bhd->bhn", q, k_sum).clamp_min(1e-6)
        return torch.einsum("bhnd,bhde->bhne", q, kv) / denominator.unsqueeze(-1)

    def _local_features(self, value: Tensor, height: int, width: int) -> Tensor:
        batch, _, length, _ = value.shape
        if height * width != length:
            raise ValueError("height * width must equal the sequence length")
        feature = value.permute(0, 2, 1, 3).reshape(batch, length, self.dim)
        feature_map = feature.transpose(1, 2).reshape(batch, self.dim, height, width)
        branches = torch.stack([branch(feature_map) for branch in self.local_branches], dim=1)
        weights = self.local_gate(feature_map).view(batch, len(self.local_branches), self.dim, 1, 1)
        weights = weights.softmax(dim=1)
        return (branches * weights).sum(dim=1).flatten(2).transpose(1, 2)

    def forward(self, x: Tensor, height: int | None = None, width: int | None = None) -> Tensor:
        batch, length, channels = x.shape
        if channels != self.dim:
            raise ValueError(f"expected {self.dim} channels, got {channels}")
        if height is None or width is None:
            side = int(math.sqrt(length))
            if side * side != length:
                raise ValueError("height and width are required for non-square sequences")
            height = width = side

        q, gate = self.qg_proj(x).chunk(2, dim=-1)
        k, v = self.kv_proj(x).chunk(2, dim=-1)
        q, k, v = self._reshape_heads(q), self._reshape_heads(k), self._reshape_heads(v)
        q_temp = self.temperature_q.unsqueeze(0).unsqueeze(2)
        k_temp = self.temperature_k.unsqueeze(0).unsqueeze(2)
        q_pos, q_neg = F.relu(q * q_temp).pow(self.power), F.relu(-q * q_temp).pow(self.power)
        k_pos, k_neg = F.relu(k * k_temp).pow(self.power), F.relu(-k * k_temp).pow(self.power)
        attention = torch.cat(
            [
                self._linear_attention(q_pos, k_pos, v),
                self._linear_attention(q_pos, k_neg, v),
                self._linear_attention(q_neg, k_pos, v),
                self._linear_attention(q_neg, k_neg, v),
            ],
            dim=-1,
        )
        attention = attention.view(batch, self.num_heads, length, 4, self.head_dim).mean(dim=3)
        attention = attention.permute(0, 2, 1, 3).reshape(batch, length, self.dim)

        spectrum = torch.fft.fft(attention, dim=1)
        frequency_filter = torch.complex(self.freq_real, self.freq_imag).view(1, 1, self.dim)
        frequency = torch.fft.ifft(spectrum * frequency_filter, dim=1).real
        attention = attention + torch.tanh(self.freq_gate) * frequency
        local = self._local_features(v, height, width)
        return self.out_proj(torch.cat([attention + local, attention], dim=-1)) * torch.sigmoid(gate)


class ESRCA(nn.Module):
    """多尺度矩形自校准注意力（论文 3.2.3.3）。"""

    def __init__(self, channels: int, kernels: Sequence[int] = (5, 11, 21)) -> None:
        super().__init__()
        self.kernels = tuple(kernels)
        self.vertical = nn.ModuleList(
            [nn.Conv2d(channels, channels, (kernel, 1), padding=(kernel // 2, 0), groups=channels, bias=False)
             for kernel in self.kernels]
        )
        self.horizontal = nn.ModuleList(
            [nn.Conv2d(channels, channels, (1, kernel), padding=(0, kernel // 2), groups=channels, bias=False)
             for kernel in self.kernels]
        )
        squeeze = max(4, channels // 16)
        self.scale_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, squeeze, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(squeeze, len(self.kernels) * 2, 1),
        )
        self.norm = nn.BatchNorm2d(channels)
        self.local = _conv_bn_act(channels, channels, 5, groups=channels)

    def forward(self, x: Tensor) -> Tensor:
        batch, _, height, width = x.shape
        h_pool = x.mean(dim=3, keepdim=True)
        w_pool = x.mean(dim=2, keepdim=True)
        gate = self.scale_gate(x).view(batch, 2, len(self.kernels), 1, 1).softmax(dim=2)
        vertical = sum(weight * conv(h_pool) for weight, conv in zip(gate[:, 0].unbind(1), self.vertical))
        horizontal = sum(weight * conv(w_pool) for weight, conv in zip(gate[:, 1].unbind(1), self.horizontal))
        attention = torch.sigmoid(self.norm(vertical + horizontal))
        return self.local(x) * attention


class BiFuse(nn.Module):
    """双向注意力融合（论文 3.2.3.4）。"""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.context_gate = nn.Conv2d(channels, channels, 1)
        self.context_proj = nn.Conv2d(channels, channels, 1)
        self.main_gate = nn.Conv2d(channels, channels, 1)

    def forward(self, main: Tensor, context: Tensor) -> Tensor:
        context = F.interpolate(context, size=main.shape[-2:], mode="nearest")
        forward = main * torch.sigmoid(self.context_gate(context))
        reverse = self.context_proj(context) * torch.sigmoid(self.main_gate(main))
        return forward + reverse


class RefinementBlock(nn.Module):
    """ESRFPN 融合节点的 ESRCA 加局部卷积精炼。"""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.attention = ESRCA(channels)
        self.refine = nn.Sequential(
            _conv_bn_act(channels, channels, 3),
            _conv_bn_act(channels, channels, 3, activation=False),
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(x + self.refine(self.attention(x)))


class ESRFPN(nn.Module):
    """增强型空间重建特征金字塔（论文 3.2.3）。"""

    def __init__(self, in_channels: Sequence[int] = (128, 256, 256), out_channels: int = 256) -> None:
        super().__init__()
        if len(in_channels) != 3:
            raise ValueError("ESRFPN expects exactly three feature levels")
        self.out_channels = out_channels
        self.input_proj = nn.ModuleList([_conv_bn_act(channels, out_channels, 1) for channels in in_channels])
        squeeze = max(4, out_channels // 16)
        self.scale_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels * 3, squeeze, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(squeeze, out_channels * 3, 1),
            nn.Sigmoid(),
        )
        self.context_blocks = nn.Sequential(
            RefinementBlock(out_channels * 3),
            RefinementBlock(out_channels * 3),
            RefinementBlock(out_channels * 3),
        )
        self.top_refine = RefinementBlock(out_channels)
        self.mid_refine = RefinementBlock(out_channels)
        self.low_refine = RefinementBlock(out_channels)
        self.top_to_mid = BiFuse(out_channels)
        self.mid_to_low = BiFuse(out_channels)
        self.down_low = _conv_bn_act(out_channels, out_channels, 3, stride=2)
        self.down_mid = _conv_bn_act(out_channels, out_channels, 3, stride=2)
        self.low_to_mid = BiFuse(out_channels)
        self.mid_to_top = BiFuse(out_channels)
        self.output_refine = nn.ModuleList([RefinementBlock(out_channels) for _ in range(3)])

    def _enhanced_pce(self, features: Sequence[Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        low, mid, high = [proj(feature) for proj, feature in zip(self.input_proj, features)]
        target_size = high.shape[-2:]
        merged = torch.cat(
            [F.adaptive_avg_pool2d(low, target_size), F.adaptive_avg_pool2d(mid, target_size), high], dim=1
        )
        merged = merged * self.scale_attention(merged)
        context = self.context_blocks(merged)
        c_low, c_mid, c_high = context.chunk(3, dim=1)
        return (
            F.interpolate(c_low, size=low.shape[-2:], mode="nearest") + low,
            F.interpolate(c_mid, size=mid.shape[-2:], mode="nearest") + mid,
            c_high + high,
        )

    def forward(self, s3: Tensor, s4: Tensor, s5: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        c3, c4, c5 = self._enhanced_pce((s3, s4, s5))
        p5 = self.top_refine(c5)
        p4 = self.mid_refine(self.top_to_mid(c4, p5))
        p3 = self.low_refine(self.mid_to_low(c3, p4))
        p4 = self.low_to_mid(p4, self.down_low(p3))
        p5 = self.mid_to_top(p5, self.down_mid(p4))
        return tuple(refine(feature) for refine, feature in zip(self.output_refine, (p3, p4, p5)))

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
    """深度卷积加逐点卷积，并在分支末尾执行 BN 和 SiLU。"""

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

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class _ChannelAveragePool(nn.Module):
    """沿通道维平均，保留完整空间尺寸供空间注意力使用。"""

    def forward(self, x: Tensor) -> Tensor:
        return x.mean(dim=1, keepdim=True)


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

        self.expand = nn.Conv2d(channels, hidden, 1, bias=False)
        self.kernel_branches = nn.ModuleList(
            [DepthwiseSeparableConv(hidden, kernel) for kernel in self.kernels]
        )
        self.kernel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden, squeeze, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(squeeze, len(self.kernels) * hidden, 1),
        )
        # 论文式(2)：各尺度分支加权求和后再经过 1x1 逐点卷积。
        self.fuse_projection = nn.Conv2d(hidden, hidden, 1, bias=False)

        spatial_hidden = max(4, hidden // 2)
        self.spatial_attention = nn.Sequential(
            _ChannelAveragePool(),
            nn.Conv2d(1, spatial_hidden, 1, bias=False),
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
        self.shortcut = nn.Conv2d(channels, hidden, 1, bias=False)
        self.project = nn.Conv2d(hidden, channels, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        expanded = self.expand(x)
        branch_features = torch.stack([branch(expanded) for branch in self.kernel_branches], dim=1)
        batch = x.shape[0]
        # 对分支维和空间维做平均，得到论文中的通道描述向量 z。
        weights = self.kernel_gate(branch_features.mean(dim=1))
        weights = weights.view(batch, len(self.kernels), self.hidden, 1, 1).softmax(dim=1)
        fused = self.fuse_projection((branch_features * weights).sum(dim=1))
        attended = fused * self.spatial_attention(fused) * self.channel_attention(fused)
        calibrated = attended * self.calibration(attended)
        return self.project(calibrated + self.shortcut(x))


class C2fAMKF(nn.Module):
    """以 AMKFBlock 为内部变换的 C2f 阶段。"""

    def __init__(self, in_channels: int, out_channels: int, repeats: int = 2, expansion: float = 0.5) -> None:
        super().__init__()
        if repeats < 1:
            raise ValueError("repeats must be positive")
        hidden = max(1, int(out_channels * expansion))
        self.input_projection = _conv_bn_act(in_channels, hidden * 2, 1)
        self.blocks = nn.ModuleList(AMKFBlock(hidden) for _ in range(repeats))
        self.output_projection = _conv_bn_act(hidden * (2 + repeats), out_channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        features = list(self.input_projection(x).chunk(2, dim=1))
        features.extend(block(features[-1]) for block in self.blocks)
        return self.output_projection(torch.cat(features, dim=1))


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
        frequency_length: int = 625,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        if frequency_length <= 0:
            raise ValueError("frequency_length must be positive")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.power = power
        self.frequency_length = frequency_length
        self.qg_proj = nn.Linear(dim, dim * 2)
        self.kv_proj = nn.Linear(dim, dim * 2)
        # 论文规定四路极化结果沿通道维拼接，再线性投影为全局特征 F。
        self.polar_projection = nn.Linear(dim * 4, dim)
        self.out_proj = nn.Linear(dim, dim)
        # 论文式(8)对 Q、K 使用同一个逐通道温度参数 tau。
        self.temperature = nn.Parameter(torch.ones(num_heads, self.head_dim))
        # 固定 800x800 输入时 S5' 为 25x25，即 N=625。对实值特征采用
        # rFFT 的独立频点（N//2+1），其余频点由共轭对称性唯一确定；这样
        # iFFT 响应严格为实值，不需要丢弃论文式(9)产生的虚部。
        frequency_bins = frequency_length // 2 + 1
        self.freq_real = nn.Parameter(torch.ones(frequency_bins, dim))
        self.freq_imag = nn.Parameter(torch.zeros(frequency_bins, dim))
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
        # 论文定义 LinearAttn(Q,K,V)=Q(K^T V)，不附加归一化分母。
        kv = torch.einsum("bhnd,bhne->bhde", k, v)
        return torch.einsum("bhnd,bhde->bhne", q, kv)

    def _frequency_filter(self, length: int) -> Tensor:
        """返回与当前序列长度匹配的可学习复数频率滤波器。"""

        # cuFFT 对 fp16 且长度为 625（非 2 的幂）的序列支持受限；频域
        # 运算统一在 fp32 中进行，再在输出处恢复原始 dtype。
        target_bins = length // 2 + 1
        real = self.freq_real.float()
        imag = self.freq_imag.float()
        if target_bins != self.freq_real.shape[0]:
            real = F.interpolate(
                real.transpose(0, 1).unsqueeze(0),
                size=target_bins,
                mode="linear",
                align_corners=False,
            ).squeeze(0).transpose(0, 1)
            imag = F.interpolate(
                imag.transpose(0, 1).unsqueeze(0),
                size=target_bins,
                mode="linear",
                align_corners=False,
            ).squeeze(0).transpose(0, 1)
        # DC（以及偶数长度时的 Nyquist）频点必须为实数，才能表示实值
        # 时域信号的共轭对称频谱。
        imag = imag.clone()
        imag[0] = 0
        if length % 2 == 0:
            imag[-1] = 0
        return torch.complex(real, imag).unsqueeze(0)

    def _local_features(self, value: Tensor, height: int, width: int) -> Tensor:
        batch, _, length, _ = value.shape
        if height * width != length:
            raise ValueError("height * width must equal the sequence length")
        feature = value.permute(0, 2, 1, 3).reshape(batch, length, self.dim)
        feature_map = feature.transpose(1, 2).reshape(batch, self.dim, height, width)
        branches = torch.stack([branch(feature_map) for branch in self.local_branches], dim=1)
        weights = self.local_gate(feature_map).view(batch, len(self.local_branches), self.dim, 1, 1)
        # 正文称为 SE 式权重融合，故使用逐通道 Sigmoid，而非核间 Softmax。
        weights = torch.sigmoid(weights)
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
        temperature = self.temperature.unsqueeze(0).unsqueeze(2)
        q_pos, q_neg = F.relu(q * temperature).pow(self.power), F.relu(-q * temperature).pow(self.power)
        k_pos, k_neg = F.relu(k * temperature).pow(self.power), F.relu(-k * temperature).pow(self.power)
        polarized = torch.cat(
            [
                self._linear_attention(q_pos, k_pos, v),
                self._linear_attention(q_pos, k_neg, v),
                self._linear_attention(q_neg, k_pos, v),
                self._linear_attention(q_neg, k_neg, v),
            ],
            dim=-1,
        )
        polarized = polarized.permute(0, 2, 1, 3).reshape(batch, length, self.dim * 4)
        global_feature = self.polar_projection(polarized)

        spectrum = torch.fft.rfft(global_feature.float(), dim=1)
        frequency = torch.fft.irfft(
            spectrum * self._frequency_filter(length), n=length, dim=1
        )
        frequency = frequency.to(global_feature.dtype)
        frequency_enhanced = global_feature + torch.tanh(self.freq_gate) * frequency
        local = self._local_features(v, height, width)
        fused = frequency_enhanced + local
        return self.out_proj(fused * torch.sigmoid(gate))


class _ScaleWeightGenerator(nn.Module):
    """由同一方向的多尺度分支生成逐通道 Softmax 权重。"""

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

    def forward(self, features: Sequence[Tensor]) -> Tensor:
        batch = features[0].shape[0]
        logits = self.mlp(torch.cat(tuple(features), dim=1))
        return logits.view(batch, self.num_scales, self.channels, 1, 1).softmax(dim=1)


class ESRCA(nn.Module):
    """多尺度矩形自校准注意力（论文 3.2.3.3）。"""

    def __init__(self, channels: int, kernels: Sequence[int] = (5, 11, 21)) -> None:
        super().__init__()
        self.kernels = tuple(kernels)
        self.vertical = nn.ModuleList(
            [nn.Conv2d(channels, channels, (kernel, 1), padding=(kernel // 2, 0), bias=False)
             for kernel in self.kernels]
        )
        self.horizontal = nn.ModuleList(
            [nn.Conv2d(channels, channels, (1, kernel), padding=(0, kernel // 2), bias=False)
             for kernel in self.kernels]
        )
        self.vertical_gate = _ScaleWeightGenerator(channels, len(self.kernels))
        self.horizontal_gate = _ScaleWeightGenerator(channels, len(self.kernels))
        self.norm = nn.BatchNorm2d(channels)
        self.local = nn.Conv2d(channels, channels, 5, padding=2, groups=channels, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        h_pool = x.mean(dim=3, keepdim=True)
        w_pool = x.mean(dim=2, keepdim=True)
        vertical_features = [conv(h_pool) for conv in self.vertical]
        horizontal_features = [conv(w_pool) for conv in self.horizontal]
        vertical_branches = torch.stack(vertical_features, dim=1)
        horizontal_branches = torch.stack(horizontal_features, dim=1)
        vertical = (vertical_branches * self.vertical_gate(vertical_features)).sum(dim=1)
        horizontal = (horizontal_branches * self.horizontal_gate(horizontal_features)).sum(dim=1)
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
    """保持原 RCM 框架，仅以 ESRCA 替换其中的 RCA token mixer。"""

    def __init__(
        self,
        channels: int,
        mlp_ratio: float = 2.0,
        layer_scale_init: float = 1e-6,
    ) -> None:
        super().__init__()
        hidden = max(channels, int(channels * mlp_ratio))
        self.attention = ESRCA(channels)
        self.norm = nn.BatchNorm2d(channels)
        self.refine = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
        )
        self.layer_scale = nn.Parameter(layer_scale_init * torch.ones(channels))

    def forward(self, x: Tensor) -> Tensor:
        refined = self.refine(self.norm(self.attention(x)))
        return x + refined * self.layer_scale.view(1, -1, 1, 1)


class ESRFPN(nn.Module):
    """增强型空间重建特征金字塔（论文 3.2.3）。"""

    def __init__(self, in_channels: Sequence[int] = (128, 256, 256), out_channels: int = 256) -> None:
        super().__init__()
        if len(in_channels) != 3:
            raise ValueError("ESRFPN expects exactly three feature levels")
        self.in_channels = tuple(in_channels)
        self.out_channels = out_channels
        merged_channels = sum(self.in_channels)
        squeeze = max(4, merged_channels // 16)
        self.scale_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(merged_channels, squeeze, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(squeeze, merged_channels, 1),
            nn.Sigmoid(),
        )
        # F_cat 按论文中的 128/256/256 通道拆分，三个 RCM-ESRCA 分支
        # 分别处理，而不是在 768 通道上串行处理。
        self.context_blocks = nn.ModuleList(
            [RefinementBlock(channels) for channels in self.in_channels]
        )
        # C3/C4/C5 恢复原分辨率并完成同层残差后，再统一到 256 通道，
        # 以满足后续 Bi-Fuse 的逐元素运算和最终输出通道要求。
        self.level_projection = nn.ModuleList(
            [
                nn.Identity()
                if channels == out_channels
                else nn.Conv2d(channels, out_channels, 1, bias=False)
                for channels in self.in_channels
            ]
        )
        self.top_to_mid = BiFuse(out_channels)
        self.mid_to_low = BiFuse(out_channels)
        self.down_low = nn.Conv2d(
            out_channels, out_channels, 3, stride=2, padding=1, bias=False
        )
        self.down_mid = nn.Conv2d(
            out_channels, out_channels, 3, stride=2, padding=1, bias=False
        )
        self.low_to_mid = BiFuse(out_channels)
        self.mid_to_top = BiFuse(out_channels)

    def _enhanced_pce(self, features: Sequence[Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        low, mid, high = features
        target_size = high.shape[-2:]
        merged = torch.cat(
            [F.adaptive_avg_pool2d(low, target_size), F.adaptive_avg_pool2d(mid, target_size), high], dim=1
        )
        merged = merged * self.scale_attention(merged)
        parts = merged.split(self.in_channels, dim=1)
        context = [block(part) for block, part in zip(self.context_blocks, parts)]
        restored = (
            F.interpolate(context[0], size=low.shape[-2:], mode="nearest") + low,
            F.interpolate(context[1], size=mid.shape[-2:], mode="nearest") + mid,
            context[2] + high,
        )
        return tuple(proj(feature) for proj, feature in zip(self.level_projection, restored))

    def forward(self, s3: Tensor, s4: Tensor, s5: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        c3, c4, c5 = self._enhanced_pce((s3, s4, s5))
        p4_top_down = self.top_to_mid(c4, c5)
        p3_out = self.mid_to_low(c3, p4_top_down)
        # 式(13)--(15)始终令低层（此处先下采样）为 F_main，
        # 相邻高层为 F_ctx，保持正向与反向调制的变量语义一致。
        p4_out = self.low_to_mid(self.down_low(p3_out), p4_top_down)
        p5_out = self.mid_to_top(self.down_mid(p4_out), c5)
        return p3_out, p4_out, p5_out

"""MKF-RT-DETR 混合编码器集成接口。"""

from __future__ import annotations

from torch import Tensor, nn

from .blocks import EPLA, ESRFPN


class MKFHybridEncoder(nn.Module):
    """将 S3/S4/S5 接入 EPLA 和 ESRFPN，输出 RT-DETR 解码器所需的金字塔特征。"""

    def __init__(self, s3_channels: int = 128, s4_channels: int = 256, s5_channels: int = 512, dim: int = 256) -> None:
        super().__init__()
        self.s5_projection = nn.Sequential(
            nn.Conv2d(s5_channels, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(inplace=True),
        )
        self.epla_norm = nn.LayerNorm(dim)
        self.epla = EPLA(dim=dim, num_heads=8)
        self.epla_ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        self.ffn_norm = nn.LayerNorm(dim)
        self.fpn = ESRFPN((s3_channels, s4_channels, dim), out_channels=dim)

    def forward(self, s3: Tensor, s4: Tensor, s5: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        s5 = self.s5_projection(s5)
        batch, channels, height, width = s5.shape
        sequence = s5.flatten(2).transpose(1, 2)
        sequence = sequence + self.epla(self.epla_norm(sequence), height, width)
        sequence = sequence + self.epla_ffn(self.ffn_norm(sequence))
        s5_enhanced = sequence.transpose(1, 2).reshape(batch, channels, height, width)
        return self.fpn(s3, s4, s5_enhanced)

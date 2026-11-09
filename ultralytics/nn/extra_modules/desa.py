"""
DESA: Dual-path Efficient Self-Attention
Lightweight TransformerEncoderLayer using channel-split dual-path design:
  - Global path: single-head self-attention on ~C/4 channels for long-range modeling
  - Local path: 3×3 DWConv on ~3C/4 channels for spatial detail extraction
  - Cross-path gating: mutual modulation between global and local features

Based on:
- SHViT (2024): Single-Head Self-Attention with partial channel processing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['TransformerEncoderLayer_DESA']


class DualPathEfficientAttention(nn.Module):
    """DESA attention: channel split → global SA + local DWConv → cross-path gating.

    Splits input into p ≈ C/4 channels for global self-attention and C-p channels
    for lightweight 3×3 DWConv local processing. Two paths exchange information
    through mutual Sigmoid-gated modulation before concatenation.
    """

    def __init__(self, dim, global_ratio=0.25):
        super().__init__()
        self.p = max(int(dim * global_ratio), 16)
        self.c_local = dim - self.p

        # Global path: group norm + 1×1 conv for Q,K,V
        self.gn = nn.GroupNorm(1, self.p)
        self.qkv_conv = nn.Conv2d(self.p, self.p * 3, 1, bias=False)

        # Local path: 3×3 depthwise conv
        self.local_dw = nn.Conv2d(self.c_local, self.c_local, 3, padding=1,
                                   groups=self.c_local, bias=False)
        self.local_bn = nn.BatchNorm2d(self.c_local)

        # Cross-path gating: 1×1 convs for cross-modulation
        self.gate_l2g = nn.Conv2d(self.c_local, self.p, 1, bias=False)
        self.gate_g2l = nn.Conv2d(self.p, self.c_local, 1, bias=False)

        self.out_conv = nn.Conv2d(dim, dim, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.silu = nn.SiLU(inplace=True)

    def forward(self, x):
        """x: [B, C, H, W] → output: [B, C, H, W]"""
        B, C, H, W = x.shape

        # Channel split: x_g for global SA, x_l for local conv
        x_g, x_l = torch.split(x, [self.p, self.c_local], dim=1)

        # --- Global path: single-head self-attention ---
        x_g = self.gn(x_g)
        qkv = self.qkv_conv(x_g)  # [B, 3p, H, W]
        q, k, v = qkv.chunk(3, dim=1)
        # Flatten spatial dims: [B, p, N]
        N = H * W
        d_k = self.p
        q = q.view(B, d_k, N)
        k = k.view(B, d_k, N)
        v = v.view(B, d_k, N)
        # Scaled dot-product attention
        scale = d_k ** -0.5
        attn = (q.transpose(-2, -1) @ k) * scale  # [B, N, N]
        attn = F.softmax(attn, dim=-1)
        g_out = (v @ attn.transpose(-2, -1)).view(B, self.p, H, W)  # [B, p, H, W]

        # --- Local path: DWConv → BN → SiLU ---
        l_out = self.local_dw(x_l)
        l_out = self.local_bn(l_out)
        l_out = self.silu(l_out)

        # --- Cross-path gating ---
        # Local modulates global: g + l features
        g_gated = g_out * self.sigmoid(self.gate_l2g(l_out))
        # Global modulates local
        l_gated = l_out * self.sigmoid(self.gate_g2l(g_out))

        # Concatenate and output
        out = torch.cat([g_gated, l_gated], dim=1)
        return self.out_conv(self.silu(out))


class TransformerEncoderLayer_DESA(nn.Module):
    """DESA: Dual-path Efficient Self-Attention transformer encoder layer.

    Uses channel-split dual-path self-attention with cross-path gating.
    Global path models long-range dependencies on ~C/4 channels;
    local path extracts spatial details on ~3C/4 channels.

    Follows AIFI registration pattern: (c1, cm=2048, num_heads=8, dropout=0, ...)
    """

    def __init__(self, c1, cm=2048, num_heads=8, dropout=0.0, act=nn.GELU(), normalize_before=False):
        super().__init__()
        self.attn = DualPathEfficientAttention(c1, global_ratio=0.25)

        # Lightweight FFN using Conv instead of Linear (for 4D features)
        self.fc1 = nn.Conv2d(c1, cm, 1)
        self.fc2 = nn.Conv2d(cm, c1, 1)

        self.norm1 = nn.GroupNorm(1, c1)
        self.norm2 = nn.GroupNorm(1, c1)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout = nn.Dropout(dropout)
        self.act = act

    def forward(self, x, src_mask=None, src_key_padding_mask=None, pos=None):
        """x: (B, C, H, W) → (B, C, H, W)"""
        # Dual-path efficient attention
        src2 = self.attn(x)
        x = x + self.dropout1(src2)
        x = self.norm1(x)

        # FFN
        src2 = self.fc2(self.dropout(self.act(self.fc1(x))))
        x = x + self.dropout2(src2)
        x = self.norm2(x)

        return x.contiguous()

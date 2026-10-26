"""使用论文图 3-1 的 S3/S4/S5 特征接口进行基础形状验证。"""

import torch

from mkf_rtdetr import AMKFBlock, C2fAMKF, MKFHybridEncoder
from mkf_rtdetr.blocks import BiFuse, EPLA, ESRCA


def require(condition: bool, message: str) -> None:
    """Raise even when Python is launched with optimization enabled."""

    if not condition:
        raise AssertionError(message)


def main() -> None:
    torch.manual_seed(0)
    amkf = AMKFBlock(128).eval()
    require(hasattr(amkf, "fuse_projection"), "AMKF is missing the equation-(2) projection")
    feature = torch.randn(1, 128, 100, 100)
    with torch.inference_mode():
        require(amkf(feature).shape == feature.shape, "AMKF changed the feature shape")

        c2f_stage = C2fAMKF(64, 128, repeats=2).eval()
        require(
            c2f_stage(torch.randn(1, 64, 16, 16)).shape == (1, 128, 16, 16),
            "C2f-AMKF wrapper shape mismatch",
        )

        q = torch.randn(1, 2, 4, 3)
        k = torch.randn(1, 2, 4, 3)
        value = torch.randn(1, 2, 4, 5)
        linear = EPLA._linear_attention(q, k, value)
        reference = q @ (k.transpose(-2, -1) @ value)
        torch.testing.assert_close(linear, reference)

        # 显式覆盖 batch>1，验证 ESRCA 的尺度权重沿 K 维广播。
        esrca = ESRCA(32).eval()
        require(
            esrca(torch.randn(2, 32, 8, 8)).shape == (2, 32, 8, 8),
            "ESRCA batch broadcasting failed",
        )

        bifuse = BiFuse(4).eval()
        main = torch.randn(1, 4, 8, 8)
        context = torch.randn(1, 4, 4, 4)
        resized = torch.nn.functional.interpolate(context, size=(8, 8), mode="nearest")
        expected = main * torch.sigmoid(bifuse.context_gate(resized))
        expected += bifuse.context_proj(resized) * torch.sigmoid(bifuse.main_gate(main))
        torch.testing.assert_close(bifuse(main, context), expected)

        encoder = MKFHybridEncoder().eval()
        require(not hasattr(encoder, "epla_norm"), "EPLA adapter must not add LayerNorm")
        require(not hasattr(encoder, "epla_ffn"), "EPLA adapter must not add an FFN")
        require(
            encoder.epla.polar_projection.in_features == 4 * encoder.epla.dim,
            "four polarity branches are not concatenated",
        )
        require(encoder.epla.polar_projection.out_features == encoder.epla.dim, "EPLA projection mismatch")
        require(encoder.epla.out_proj.in_features == encoder.epla.dim, "EPLA output projection mismatch")
        require(
            encoder.epla.temperature.shape == (encoder.epla.num_heads, encoder.epla.head_dim),
            "EPLA temperature is not per channel",
        )
        require(not hasattr(encoder.epla, "temperature_q"), "Q and K must share one temperature")
        require(
            encoder.epla.freq_real.shape == (625 // 2 + 1, encoder.epla.dim),
            "real-valued FFT filter has the wrong independent-bin count",
        )
        require(sum(encoder.fpn.in_channels) == 640, "Enhanced-PCE input must have 640 channels")
        p3, p4, p5 = encoder(
            torch.randn(1, 128, 100, 100),
            torch.randn(1, 256, 50, 50),
            torch.randn(1, 512, 25, 25),
        )
    require(p3.shape == (1, 256, 100, 100), "P3 paper-resolution shape mismatch")
    require(p4.shape == (1, 256, 50, 50), "P4 paper-resolution shape mismatch")
    require(p5.shape == (1, 256, 25, 25), "P5 paper-resolution shape mismatch")
    require(all(torch.isfinite(output).all().item() for output in (p3, p4, p5)), "non-finite output")
    print("MKF-RT-DETR chapter 3 module check passed")


if __name__ == "__main__":
    main()

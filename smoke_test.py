"""使用论文图 3-1 的 S3/S4/S5 特征接口进行基础形状验证。"""

import torch

from mkf_rtdetr import AMKFBlock, MKFHybridEncoder


def main() -> None:
    torch.manual_seed(0)
    amkf = AMKFBlock(128).eval()
    feature = torch.randn(1, 128, 80, 80)
    with torch.inference_mode():
        assert amkf(feature).shape == feature.shape

        encoder = MKFHybridEncoder().eval()
        p3, p4, p5 = encoder(
            torch.randn(1, 128, 80, 80),
            torch.randn(1, 256, 40, 40),
            torch.randn(1, 512, 20, 20),
        )
    assert p3.shape == (1, 256, 80, 80)
    assert p4.shape == (1, 256, 40, 40)
    assert p5.shape == (1, 256, 20, 20)
    print("MKF-RT-DETR chapter 3 module check passed")


if __name__ == "__main__":
    main()

# MKF-RT-DETR Smoking Detection

This repository contains the official implementation of MKF-RT-DETR used for
the experiments reported in the manuscript. The released model includes three
enhanced modules:

- `AMKF`：自适应多尺度核融合模块，用于骨干网络 Stage 3--5 的多尺度特征提取。
- `EPLA`：增强型极化线性注意力，用于混合编码器最高层的特征交互。
- `ESRFPN`：增强型空间重建特征金字塔，用于跨尺度特征融合。

## Repository Layout

```text
mkf_rtdetr/
  blocks.py       # AMKF, C2fAMKF, EPLA, ESRCA, Bi-Fuse, ESRFPN
  model.py        # S3/S4/S5 to P3/P4/P5 encoder integration
  __init__.py
configs/
  mkf_rtdetr_r18.yaml
dataset/
  smoking.rar     # Dataset archive, managed with Git LFS
smoke_test.py     # Tensor shape check
requirements.txt
```

## Installation and Validation

Use this code in a Python environment that contains a complete RT-DETR project.
This repository provides the model enhancement modules, but does not include the
base RT-DETR decoder or trained weights. The dataset archive is managed with Git
LFS.

```bash
pip install -r requirements.txt
python smoke_test.py
```

The validation script uses the following feature interfaces:

- `S3`: `[B, 128, 100, 100]`
- `S4`: `[B, 256, 50, 50]`
- `S5`: `[B, 512, 25, 25]`
- 输出 `P3/P4/P5`: 统一为 256 通道。

These sizes correspond to the paper's `800x800` input. EPLA therefore uses a
default sequence length of `25x25=625`. The implementation uses `rFFT/irFFT`
so that the reconstructed feature is real-valued; when `N=625`, only
`floor(N/2)+1=313` independent complex frequency bins are parameterized. For
other input sizes, the filter parameters are interpolated along the independent
frequency-bin dimension.

## Architecture Specification

- AMKF applies the post-fusion `1x1` projection in Eq. (2), keeps the
  full `H x W` map in spatial attention, and uses Sigmoid for the channel
  calibration in Eq. (6).
- EPLA computes linear attention as `Q(K^T V)`, concatenates and projects the
  four polarity branches, applies the learnable 1-D complex filter with
  `rFFT/irFFT` in Eqs. (9)--(10), and follows the paper's add, gate,
  output-projection order. Its integration adapter only converts between BCHW
  and BNC layouts; it does not add an extra residual/LN/FFN transformer shell
  that is absent from the equations.
- Enhanced PCE constructs the paper's `128+256+256=640` channel `F_cat`, splits
  it into three RCM-ESRCA branches, restores the three spatial resolutions, and
  then performs bidirectional Bi-Fuse propagation. ESRFPN receives three input
  feature maps and produces the `P3/P4/P5` outputs.
- ESRCA's `5/11/21` horizontal and vertical strip operators are ordinary
  convolutions. Only the parallel local `5x5` branch is depthwise convolution.
  Formula-defined projections and stride-2 downsampling are plain convolutions,
  without implicit BN or activation.

The released implementation uses the following fixed settings: EPLA `heads=8`
and polarity power `p=3`; local SE reduction `16`; PCE reduction `16`; ESRCA
scale-weight reduction `4`; and RCM MLP ratio `2` with layer-scale
initialization `1e-6`.

The released model follows the numbered equations and the defined
AMKF/EPLA/Enhanced-PCE/ESRCA/Bi-Fuse data flow. The model definition and
configuration in this repository correspond to the implementation used in the
manuscript.

## RT-DETR Integration

1. Use C2f-AMKF stages in ResNet-18 stages 3--5 while preserving output channel
   counts and spatial sizes.
2. Project the top-level feature `S5` to 256 channels, flatten it, apply EPLA,
   and restore the feature map.
3. Feed `S3`, `S4`, and the enhanced `S5'` to `ESRFPN`, then connect the resulting
   `P3/P4/P5` features to the RT-DETR decoder and prediction head.

The configuration file records the model structure and training settings without
depending on a particular RT-DETR YAML parser.

`smoke_test.py` validates the module interfaces and output shapes in an
environment with PyTorch.

## Dataset

The dataset archive is located at `dataset/smoking.rar`. It is larger than GitHub's
normal file limit, so upload this repository with Git LFS enabled.

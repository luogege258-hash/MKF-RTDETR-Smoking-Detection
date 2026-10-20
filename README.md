# MKF-RT-DETR Smoking Detection

This repository provides the core implementation of MKF-RT-DETR for smoking
behavior detection. It includes three enhanced modules:

- `AMKF`：自适应多尺度核融合模块，替代骨干网络 Stage 3--5 中的固定 `3x3` 卷积。
- `EPLA`：增强型极化线性注意力，替代混合编码器最高层的 AIFI 自注意力。
- `ESRFPN`：增强型空间重建特征金字塔，替代 CCFM 跨尺度特征融合模块。

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
original RT-DETR decoder or trained weights. The dataset archive is managed with
Git LFS.

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

## Formula Alignment

- AMKF applies the post-fusion `1x1` projection specified by Eq. (2), keeps the
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
  then performs bidirectional Bi-Fuse propagation. ESRFPN is a complete
  three-input/three-output module; there is no single-input compatibility block.
- ESRCA's `5/11/21` horizontal and vertical strip operators are ordinary
  convolutions. Only the parallel local `5x5` branch is depthwise convolution.
  Formula-defined projections and stride-2 downsampling are plain convolutions,
  without implicit BN or activation.

These architecture corrections change parameter shapes and the computation
graph. Previous checkpoints are not fully compatible; retraining and fresh
Params/GFLOPs/FPS/accuracy measurements are required.

The following are reproducibility defaults exposed by the code but not yet
stated explicitly in the current manuscript: EPLA `heads=8` and polarity power
`p=3`; local SE reduction `16`; PCE reduction `16`; ESRCA scale-weight
reduction `4`; and RCM MLP ratio `2` with layer-scale initialization `1e-6`.
These values should be added to the manuscript's implementation details before
submission.

Scope: this implementation follows the numbered equations and the defined
AMKF/EPLA/Enhanced-PCE/ESRCA/Bi-Fuse data flow. The `DIF` and `RepC3` labels in
the overview figure are not implemented because the manuscript does not define
their operators or equations; their exact design must be specified before they
can be added reproducibly.

## RT-DETR Integration

1. Replace the BasicBlocks in ResNet-18 stages 3--5 with C2f stages whose
   internal transform is `AMKFBlock`, preserving output channel counts and
   spatial sizes.
2. Project the top-level feature `S5` to 256 channels, flatten it, apply EPLA,
   and restore the feature map.
3. Feed `S3`, `S4`, and the enhanced `S5'` to `ESRFPN`, then connect the resulting
   `P3/P4/P5` features to the RT-DETR decoder and prediction head.

The configuration file records the model structure and training settings without
depending on a particular RT-DETR YAML parser.

`smoke_test.py` is supplied for validation in an environment with PyTorch. This
README does not claim that a PyTorch forward pass has been completed on the
current editing machine.

## Dataset

The dataset archive is located at `dataset/smoking.rar`. It is larger than GitHub's
normal file limit, so upload this repository with Git LFS enabled.

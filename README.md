# MKF-RT-DETR Smoking Detection

This repository provides the core implementation of MKF-RT-DETR for smoking
behavior detection. It includes three enhanced modules:

- `AMKF`：自适应多尺度核融合模块，替代骨干网络 Stage 3--5 中的固定 `3x3` 卷积。
- `EPLA`：增强型极化线性注意力，替代混合编码器最高层的 AIFI 自注意力。
- `ESRFPN`：增强型空间重建特征金字塔，替代 CCFM 跨尺度特征融合模块。

## Repository Layout

```text
mkf_rtdetr/
  blocks.py       # AMKF, EPLA, ESRCA, Bi-Fuse, ESRFPN
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

- `S3`: `[B, 128, 80, 80]`
- `S4`: `[B, 256, 40, 40]`
- `S5`: `[B, 512, 20, 20]`
- 输出 `P3/P4/P5`: 统一为 256 通道。

## RT-DETR Integration

1. Replace the fixed `3x3` convolution units in ResNet-18 stages 3--5 with
   `AMKFBlock`, preserving output channel counts and spatial sizes.
2. Project the top-level feature `S5` to 256 channels, flatten it, apply EPLA,
   and restore the feature map.
3. Feed `S3`, `S4`, and the enhanced `S5` to `ESRFPN`, then connect the resulting
   `P3/P4/P5` features to the RT-DETR decoder and prediction head.

The configuration file records the model structure and training settings without
depending on a particular RT-DETR YAML parser.

## Dataset

The dataset archive is located at `dataset/smoking.rar`. It is larger than GitHub's
normal file limit, so upload this repository with Git LFS enabled.

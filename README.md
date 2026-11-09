# MKF-RT-DETR Smoking Detection

This repository is the complete training and evaluation implementation used for
the manuscript **“A Smoking Detection Method for Surveillance Scenes Enhanced
by Multi-scale Kernel Fusion and Frequency-domain Polarized Attention.”** It
contains the RT-DETR training framework, the MKF-RT-DETR modules, all eight
ablation configurations, the released dataset in directly browsable form, and
the scripts required to reproduce Table 2 and the MKF-RT-DETR row in Table 3.

## Reproducibility checklist

- Complete RT-DETR source code is included under `ultralytics/`.
- The 7,000 images, COCO annotations, and generated YOLO labels are stored as
  ordinary files under `dataset/Cigarette-ghnlk/`; no archive extraction is
  required.
- All eight Table 2 ablations are defined by local YAML files.
- Training, test-set validation, FPS/GFLOPs measurement, and table export have
  separate command-line entry points.
- Dependency versions and the Python/CUDA environment are pinned.
- Source comments and configuration comments are in English.
- Code is distributed under AGPL-3.0; see `LICENSE` and `NOTICE`.

For an immutable version, use the `v1.0-reproducible` Git tag created for the
submission. Record `git rev-parse HEAD` together with every reported run.

## Repository layout

```text
dataset/
  Cigarette-ghnlk/
    train/{images,labels}/
    valid/{images,labels}/
    test/{images,labels}/
    */_annotations.coco.json
  data_smoking.yaml
  MANIFEST.sha256
scripts/
  prepare_dataset.py          # COCO-to-YOLO conversion and integrity manifest
  check_dataset.py            # split, label, and class validation
  check_formula_shapes.py     # numerical checks for manuscript equations
  check_models.py             # explicit forward for all eight model graphs
  train_chapter3.py           # full model and ablation training
  validate_chapter3.py        # test-set metrics
  benchmark_chapter3.py       # Params, GFLOPs, latency, and FPS
  export_tables.py            # Table 2 and Table 3-ready CSV/Markdown files
ultralytics/
  cfg/models/rt-detr/         # baseline, ablations, and full-model YAML files
  nn/extra_modules/amkf.py
  nn/extra_modules/epla.py
  nn/extra_modules/esrfpn.py
weights/                      # released checkpoint location
```

## Dataset

The dataset contains 7,000 images at 800 x 800 pixels and 25,800 annotated
objects:

| Split | Images | Boxes |
|---|---:|---:|
| train | 5,000 | 18,500 |
| valid | 1,000 | 3,600 |
| test | 1,000 | 3,700 |

Class definitions are fixed throughout the repository:

- `0 = smoking`: a person exhibiting visible smoking behavior, identified by
  the cigarette/hand-to-mouth interaction and its local context.
- `1 = nosmoke`: a person included as a hard negative who is not exhibiting
  smoking behavior in the annotated instance.

Verify the released files and labels before starting a run:

```bash
python scripts/prepare_dataset.py
python scripts/check_dataset.py
```

## Environment

The reference environment is Python 3.10, PyTorch 2.1.2, torchvision 0.16.2,
CUDA 12.1, and an NVIDIA RTX 3090 (24 GB).

Linux/CUDA installation:

```bash
git clone https://github.com/luogege258-hash/MKF-RTDETR-Smoking-Detection.git
cd MKF-RTDETR-Smoking-Detection
git checkout v1.0-reproducible
bash scripts/setup_server.sh
source .venv/bin/activate
```

Equivalent manual installation:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-cu121.txt
python -m pip install -e . --no-deps
```

Before starting a long run, execute the implementation checks:

```bash
python scripts/check_formula_shapes.py
python scripts/check_models.py --imgsz 800
```

## Model-to-paper mapping

| Manuscript component | Implementation |
|---|---|
| AMKF | `ultralytics/nn/extra_modules/amkf.py` |
| EPLA | `ultralytics/nn/extra_modules/epla.py` |
| Enhanced PCE, RCM-ESRCA, and Bi-Fuse | `ultralytics/nn/extra_modules/esrfpn.py` |
| Full MKF-RT-DETR | `ultralytics/cfg/models/rt-detr/rtdetr-enhanced.yaml` |
| Eight Table 2 ablations | `scripts/experiment_config.py` |

The released implementation follows the equations and structure diagrams in
the manuscript: EPLA uses four polarized `Q(K^T V)` branches, a learnable 1-D
complex filter implemented with `rFFT/irFFT`, and 3/5/7 local branches. ESRFPN
jointly receives `S3`, `S4`, and `S5'`, constructs the 640-channel Enhanced-PCE
feature, and performs bidirectional Bi-Fuse propagation.

## Training

Train the full model with the paper settings (800 x 800, 300 epochs, batch 16,
AdamW, initial learning rate `1e-4`, weight decay `1e-4`, cosine schedule):

```bash
bash scripts/run_full_model.sh
```

Train all eight Table 2 configurations:

```bash
bash scripts/run_all_ablation.sh
```

Direct Python commands are also supported:

```bash
python scripts/train_chapter3.py --exp E04_full
python scripts/train_chapter3.py --exp all
python scripts/train_chapter3.py --resume runs/chapter3/E04_full/weights/last.pt
```

When GPU memory is insufficient, reduce the batch size while keeping all other
settings fixed:

```bash
BATCH=8 bash scripts/run_full_model.sh
```

## Validation, FPS, and tables

Evaluate the full model on the fixed test split:

```bash
python scripts/validate_chapter3.py --exp E04_full --batch 8
```

Evaluate every completed ablation:

```bash
python scripts/validate_chapter3.py --batch 8
```

Measure PyTorch inference efficiency using batch 1, 800 x 800 input, 100 warmup
iterations, and 300 timed iterations:

```bash
python scripts/benchmark_chapter3.py --exp E04_full --device 0
python scripts/benchmark_chapter3.py --all --device 0
```

Generate the files used to check Tables 2 and 3:

```bash
python scripts/export_tables.py
```

Outputs are written to `runs/chapter3/reproduction/`. The manuscript reports
83.9% mAP@0.5, 36.4% mAP@0.5:0.95, 25.8 M parameters, 72.4 GFLOPs, and 30 FPS
for MKF-RT-DETR. Reproduced values must be read from the generated JSON/CSV
files rather than manually copied into them.

### Scope of Table 3 reproduction

This codebase reproduces the proposed MKF-RT-DETR row. The remaining Table 3
rows are external YOLO-family implementations and are not silently substituted
by similarly named models in this modified RT-DETR framework. A complete
archival reproduction of those rows additionally requires, for each external
model, the exact upstream repository commit, model configuration, training
arguments, checkpoint digest, validation output, and benchmark output used in
the comparison. `export_tables.py` deliberately does not invent or copy those
measurements.

## Checkpoint

The expected released checkpoint name is `weights/E04_full_best.pt`. After
placing the actual training output there, verify it and run:

```bash
sha256sum weights/E04_full_best.pt
python scripts/validate_chapter3.py --weights weights/E04_full_best.pt --batch 8
python scripts/benchmark_chapter3.py --weights weights/E04_full_best.pt --device 0
```

The repository does not create or substitute synthetic weights. A checkpoint
must be the actual output of the formula-aligned full-model training run.

## Citation

Citation metadata are provided in `CITATION.cff`. Please cite the associated
article when using the code or dataset.

## License and third-party code

The code is licensed under the GNU Affero General Public License v3.0. This
repository contains a modified Ultralytics 8.0.201 codebase; attribution and
license details are recorded in `NOTICE`. Dataset redistribution terms are
documented separately in `dataset/README.md` and are not changed by the code
license.

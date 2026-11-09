#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "[1/5] Creating Python virtual environment"
"$PYTHON_BIN" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

echo "[2/5] Installing the locked PyTorch 2.1.2/CUDA 12.1 environment"
python -m pip install -r requirements-cu121.txt
python -m pip install -e . --no-deps

echo "[3/5] Checking CUDA and custom modules"
python - <<'PY'
import torch
from ultralytics.nn.extra_modules import AMKF, ESRFPN, TransformerEncoderLayer_EPLA
print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable; check the NVIDIA driver and GPU allocation.")
print("GPU:", torch.cuda.get_device_name(0))
print("VRAM(GB):", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1))
print("Custom modules: OK")
PY

echo "[4/5] Checking manuscript formulas and paper-size module shapes"
python scripts/check_formula_shapes.py

echo "[5/5] Running explicit forwards for all eight chapter-3 models"
python scripts/check_models.py --imgsz 256
echo "Setup completed. For the paper size, also run: python scripts/check_models.py --imgsz 800"
echo "Next: verify the included dataset with: python scripts/check_dataset.py"

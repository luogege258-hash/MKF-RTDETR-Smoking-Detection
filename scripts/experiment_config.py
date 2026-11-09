"""Shared paths and paper settings for the chapter-3 experiments."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "ultralytics" / "cfg" / "models" / "rt-detr"
DATA_YAML = ROOT / "dataset" / "data_smoking.yaml"
RUNS_DIR = ROOT / "runs" / "chapter3"

EXPERIMENTS = {
    "E01_baseline": MODEL_DIR / "rtdetr-r18.yaml",
    "E02_AMKF": MODEL_DIR / "rtdetr-AMKF.yaml",
    "E03_AMKF_ESRFPN": MODEL_DIR / "rtdetr-AMKF-ESRFPN.yaml",
    "E04_full": MODEL_DIR / "rtdetr-enhanced.yaml",
    "E05_ESRFPN_only": MODEL_DIR / "rtdetr-ESRFPN-only.yaml",
    "E06_EPLA_only": MODEL_DIR / "rtdetr-EPLA-only.yaml",
    "E07_AMKF_EPLA": MODEL_DIR / "rtdetr-AMKF-EPLA.yaml",
    "E08_ESRFPN_EPLA": MODEL_DIR / "rtdetr-ESRFPN-EPLA.yaml",
}

PAPER_ARGS = {
    "data": str(DATA_YAML),
    "epochs": 300,
    "imgsz": 800,
    "batch": 16,
    "workers": 4,
    "device": "0",
    "optimizer": "AdamW",
    "lr0": 1e-4,
    "weight_decay": 1e-4,
    "cos_lr": True,
    "patience": 300,
    "project": str(RUNS_DIR),
    "exist_ok": True,
    "save": True,
    "save_period": 20,
    "val": True,
    "plots": True,
    "seed": 42,
}

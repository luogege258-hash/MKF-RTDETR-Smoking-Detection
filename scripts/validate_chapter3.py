"""Evaluate trained chapter-3 models on the fixed test split and write JSON."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ultralytics import RTDETR  # noqa: E402
from experiment_config import DATA_YAML, RUNS_DIR  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp", help="evaluate one experiment directory")
    parser.add_argument("--weights", type=Path, help="evaluate one explicit checkpoint")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=800)
    args = parser.parse_args()
    if args.weights:
        explicit_weight = args.weights.resolve()
        if not explicit_weight.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {explicit_weight}")
        weights = [explicit_weight]
    else:
        weights = sorted(RUNS_DIR.glob("*/weights/best.pt"))
    if args.exp and not args.weights:
        weights = [p for p in weights if p.parents[1].name == args.exp]
    if not weights:
        raise FileNotFoundError(f"No best.pt checkpoint found under {RUNS_DIR}")
    results = []
    for weight in weights:
        name = weight.parents[1].name if weight.name == "best.pt" else weight.stem
        print(f"\nValidating {name}: {weight}")
        model = RTDETR(str(weight))
        metrics = model.val(data=str(DATA_YAML), imgsz=args.imgsz, batch=args.batch,
                            device=args.device, split="test", project=str(RUNS_DIR / "validation"), name=name)
        results.append({
            "name": name, "weight": str(weight),
            "mAP50": round(float(metrics.box.map50), 6),
            "mAP50_95": round(float(metrics.box.map), 6),
            "precision": round(float(metrics.box.mp), 6),
            "recall": round(float(metrics.box.mr), 6),
            "params_M": round(sum(p.numel() for p in model.model.parameters()) / 1e6, 3),
        })
    output = RUNS_DIR / "chapter3_test_results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nResults saved to: {output}")


if __name__ == "__main__":
    main()

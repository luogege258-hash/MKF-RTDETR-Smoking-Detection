"""Measure parameters, GFLOPs, latency, and FPS for trained chapter-3 models."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiment_config import RUNS_DIR  # noqa: E402
from ultralytics import RTDETR  # noqa: E402
from ultralytics.utils.torch_utils import get_flops  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp", default="E04_full", help="experiment directory name")
    parser.add_argument("--all", action="store_true", help="benchmark every available best.pt")
    parser.add_argument("--weights", type=Path, help="benchmark one explicit checkpoint")
    parser.add_argument("--device", default="0", help="CUDA index or 'cpu'")
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=300)
    return parser.parse_args()


def resolve_weights(args: argparse.Namespace) -> list[Path]:
    if args.weights:
        path = args.weights.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return [path]
    if args.all:
        weights = sorted(RUNS_DIR.glob("*/weights/best.pt"))
    else:
        weights = [RUNS_DIR / args.exp / "weights" / "best.pt"]
    missing = [path for path in weights if not path.is_file()]
    if missing:
        rendered = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Checkpoint files not found:\n{rendered}")
    if not weights:
        raise FileNotFoundError(f"No best.pt files found under {RUNS_DIR}")
    return weights


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark(weight: Path, args: argparse.Namespace, device: torch.device) -> dict:
    wrapper = RTDETR(str(weight))
    model = wrapper.model.to(device).eval()
    image = torch.zeros(1, 3, args.imgsz, args.imgsz, device=device)

    with torch.inference_mode():
        for _ in range(args.warmup):
            model(image)
        synchronize(device)
        start = time.perf_counter()
        for _ in range(args.iterations):
            model(image)
        synchronize(device)
        elapsed = time.perf_counter() - start

    latency_ms = elapsed * 1000.0 / args.iterations
    result = {
        "name": weight.parents[1].name if weight.name == "best.pt" else weight.stem,
        "weight": str(weight),
        "device": str(device),
        "image_size": args.imgsz,
        "batch_size": 1,
        "warmup_iterations": args.warmup,
        "timed_iterations": args.iterations,
        "params_M": round(sum(parameter.numel() for parameter in model.parameters()) / 1e6, 3),
        "GFLOPs": round(float(get_flops(model, args.imgsz)), 3),
        "latency_ms": round(latency_ms, 3),
        "FPS": round(1000.0 / latency_ms, 3),
    }
    del image, model, wrapper
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    args = parse_args()
    if args.warmup < 1 or args.iterations < 1:
        raise ValueError("warmup and iterations must both be positive")
    if args.device == "cpu":
        device = torch.device("cpu")
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; pass --device cpu for a CPU benchmark")
        device = torch.device(f"cuda:{args.device}")

    results = []
    for weight in resolve_weights(args):
        print(f"Benchmarking {weight} on {device} ...")
        result = benchmark(weight, args, device)
        results.append(result)
        print(
            f"{result['name']}: {result['params_M']} M params, {result['GFLOPs']} GFLOPs, "
            f"{result['latency_ms']} ms, {result['FPS']} FPS"
        )

    output = RUNS_DIR / "chapter3_benchmark_results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Benchmark results saved to: {output}")


if __name__ == "__main__":
    main()

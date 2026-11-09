"""Build all eight ablations, run an explicit forward, and reject non-finite output."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterator

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiment_config import EXPERIMENTS  # noqa: E402
from ultralytics.nn.tasks import RTDETRDetectionModel  # noqa: E402


def require(condition: bool, message: str) -> None:
    """Raise even when Python is started with ``-O`` (unlike bare assert)."""
    if not condition:
        raise AssertionError(message)


def tensor_leaves(value: Any) -> Iterator[torch.Tensor]:
    """Yield tensors from nested Ultralytics prediction/raw-output structures."""
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from tensor_leaves(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from tensor_leaves(item)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--imgsz",
        type=int,
        default=256,
        help="square forward size (default: 256 for a quick check; use 800 for the paper setting)",
    )
    parser.add_argument(
        "--nc",
        type=int,
        default=2,
        help="number of detector classes (default: 2, matching smoking/nosmoke)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require(args.imgsz > 0, "--imgsz must be positive")
    require(args.imgsz % 32 == 0, "--imgsz must be divisible by 32 (for example 256 or 800)")
    require(args.nc > 0, "--nc must be positive")
    require(len(EXPERIMENTS) == 8, f"expected 8 ablation models, found {len(EXPERIMENTS)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Explicit-forward device: {device}; image size: {args.imgsz}x{args.imgsz}; "
        f"classes: {args.nc}"
    )

    for name, yaml_path in EXPERIMENTS.items():
        require(yaml_path.is_file(), f"{name}: missing model YAML: {yaml_path}")
        # Training rebuilds the YAML with the dataset class count. Construct
        # the same nc=2 graph here instead of validating the YAML's nc=80
        # placeholder head.
        network = RTDETRDetectionModel(str(yaml_path), nc=args.nc, verbose=False).to(device).eval()
        image = torch.zeros(1, 3, args.imgsz, args.imgsz, device=device)
        with torch.inference_mode():
            output = network(image)

        leaves = list(tensor_leaves(output))
        require(bool(leaves), f"{name}: explicit forward returned no tensors")
        for index, tensor in enumerate(leaves):
            require(
                bool(torch.isfinite(tensor).all().item()),
                f"{name}: output tensor {index} contains NaN or Inf (shape={tuple(tensor.shape)})",
            )

        parameters = sum(parameter.numel() for parameter in network.parameters()) / 1e6
        shapes = ", ".join(str(tuple(tensor.shape)) for tensor in leaves[:3])
        suffix = ", ..." if len(leaves) > 3 else ""
        print(
            f"{name:<22} FORWARD OK  params={parameters:.2f}M  "
            f"tensor_outputs={len(leaves)}  first_shapes=[{shapes}{suffix}]"
        )

        del output, image, network
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(
        "All eight explicit forwards passed. "
        "For the manuscript input size, rerun: python scripts/check_models.py --imgsz 800"
    )


if __name__ == "__main__":
    main()

"""Train the baseline, single-module, combined, or full chapter-3 model."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import RTDETR  # noqa: E402
from experiment_config import EXPERIMENTS, PAPER_ARGS  # noqa: E402

STACK = ["E01_baseline", "E02_AMKF", "E03_AMKF_ESRFPN", "E04_full"]
SINGLE = ["E02_AMKF", "E05_ESRFPN_only", "E06_EPLA_only"]
ALL = [
    "E01_baseline",
    "E02_AMKF",
    "E05_ESRFPN_only",
    "E06_EPLA_only",
    "E03_AMKF_ESRFPN",
    "E07_AMKF_EPLA",
    "E08_ESRFPN_EPLA",
    "E04_full",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--exp", default="E04_full",
        choices=[*EXPERIMENTS, "stack", "single", "all"],
        help="train one experiment; 'all' runs all eight Table 2 configurations",
    )
    parser.add_argument("--batch", type=int, default=PAPER_ARGS["batch"])
    parser.add_argument("--device", default=PAPER_ARGS["device"])
    parser.add_argument("--workers", type=int, default=PAPER_ARGS["workers"])
    parser.add_argument("--epochs", type=int, default=PAPER_ARGS["epochs"])
    parser.add_argument("--imgsz", type=int, default=PAPER_ARGS["imgsz"])
    parser.add_argument("--resume", type=Path, help="resume from an explicit last.pt checkpoint")
    return parser.parse_args()


def selected_experiments(value):
    if value == "stack":
        return STACK
    if value == "single":
        return SINGLE
    if value == "all":
        return ALL
    return [value]


def train_one(name, args):
    settings = dict(PAPER_ARGS)
    settings.update(batch=args.batch, device=args.device, workers=args.workers,
                    epochs=args.epochs, imgsz=args.imgsz, name=name)
    print("\n" + "=" * 72)
    print(f"Experiment: {name}")
    print(f"Model: {EXPERIMENTS[name]}")
    print(f"imgsz={args.imgsz}, batch={args.batch}, epochs={args.epochs}, device={args.device}")
    print("=" * 72)
    model = RTDETR(str(EXPERIMENTS[name]))
    model.info()
    model.train(**settings)


def main():
    args = parse_args()
    if args.resume:
        if not args.resume.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
        RTDETR(str(args.resume)).train(resume=True)
        return
    for name in selected_experiments(args.exp):
        train_one(name, args)


if __name__ == "__main__":
    main()

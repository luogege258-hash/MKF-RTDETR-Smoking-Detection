"""Export reproducible Table 2 and MKF-RT-DETR Table 3 rows from JSON results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from experiment_config import RUNS_DIR


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results"
EXPERIMENT_LAYOUT = {
    "E01_baseline": (False, False, False),
    "E02_AMKF": (True, False, False),
    "E05_ESRFPN_only": (False, True, False),
    "E06_EPLA_only": (False, False, True),
    "E03_AMKF_ESRFPN": (True, True, False),
    "E07_AMKF_EPLA": (True, False, True),
    "E08_ESRFPN_EPLA": (False, True, True),
    "E04_full": (True, True, True),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics",
        type=Path,
        default=RUNS_DIR / "chapter3_test_results.json",
    )
    parser.add_argument(
        "--benchmarks",
        type=Path,
        default=RUNS_DIR / "chapter3_benchmark_results.json",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_index(path: Path) -> dict[str, dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Result file not found: {path}")
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {row["name"]: row for row in rows}


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, fields: list[str], rows: list[dict]) -> None:
    header = "| " + " | ".join(fields) + " |"
    divider = "|" + "|".join("---" for _ in fields) + "|"
    body = ["| " + " | ".join(str(row.get(field, "")) for field in fields) + " |" for row in rows]
    path.write_text("\n".join([header, divider, *body]) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    metrics = read_index(args.metrics)
    benchmarks = read_index(args.benchmarks)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    table2 = []
    for name, modules in EXPERIMENT_LAYOUT.items():
        if name not in metrics or name not in benchmarks:
            raise KeyError(f"Missing validation or benchmark result for {name}")
        metric = metrics[name]
        benchmark = benchmarks[name]
        table2.append(
            {
                "Experiment": name,
                "Baseline": "yes",
                "AMKF": "yes" if modules[0] else "no",
                "ESRFPN": "yes" if modules[1] else "no",
                "EPLA": "yes" if modules[2] else "no",
                "mAP@0.5 (%)": round(metric["mAP50"] * 100.0, 2),
                "mAP@0.5:0.95 (%)": round(metric["mAP50_95"] * 100.0, 2),
                "FPS": benchmark["FPS"],
                "Params (M)": benchmark["params_M"],
            }
        )
    table2_fields = list(table2[0])
    write_csv(output / "table2_ablation.csv", table2_fields, table2)
    write_markdown(output / "table2_ablation.md", table2_fields, table2)

    full_metric = metrics["E04_full"]
    full_benchmark = benchmarks["E04_full"]
    table3 = [
        {
            "Method": "MKF-RT-DETR",
            "mAP@0.5 (%)": round(full_metric["mAP50"] * 100.0, 2),
            "mAP@0.5:0.95 (%)": round(full_metric["mAP50_95"] * 100.0, 2),
            "Params (M)": full_benchmark["params_M"],
            "GFLOPs": full_benchmark["GFLOPs"],
            "FPS": full_benchmark["FPS"],
        }
    ]
    table3_fields = list(table3[0])
    write_csv(output / "table3_mkf_rtdetr_row.csv", table3_fields, table3)
    write_markdown(output / "table3_mkf_rtdetr_row.md", table3_fields, table3)
    print(f"Table files written to: {output}")


if __name__ == "__main__":
    main()

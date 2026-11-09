"""Check dataset paths, split sizes, YOLO labels, classes, and coordinates."""

from collections import Counter
from pathlib import Path

try:
    import yaml
except ModuleNotFoundError:  # Allow the dataset-only check before environment installation.
    yaml = None

ROOT = Path(__file__).resolve().parents[1]
YAML_PATH = ROOT / "dataset" / "data_smoking.yaml"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EXPECTED = {"train": 5000, "valid": 1000, "test": 1000}


def load_dataset_config():
    """Load the small dataset YAML, with a dependency-free fallback."""

    text = YAML_PATH.read_text(encoding="utf-8")
    if yaml is not None:
        return yaml.safe_load(text)
    config = {}
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        value = value.strip()
        if value and not value.startswith(("{", "[")):
            config[key.strip()] = value
    required = {"path", "train", "val", "test"}
    if not required.issubset(config):
        raise ValueError(f"Unable to parse required keys from {YAML_PATH}")
    return config


def main():
    config = load_dataset_config()
    root = Path(config["path"])
    dataset_root = (YAML_PATH.parent / root).resolve() if not root.is_absolute() else root
    print(f"Dataset root: {dataset_root}")
    errors, class_counts = [], Counter()
    rows = []
    for split, key in (("train", "train"), ("valid", "val"), ("test", "test")):
        image_dir = dataset_root / config[key]
        label_dir = image_dir.parent / "labels"
        images = sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES) if image_dir.is_dir() else []
        missing, invalid = 0, 0
        for image in images:
            label = label_dir / image.relative_to(image_dir).with_suffix(".txt")
            if not label.is_file():
                missing += 1
                continue
            for line in label.read_text(encoding="utf-8").splitlines():
                fields = line.split()
                try:
                    cls = int(fields[0]); coords = [float(v) for v in fields[1:]]
                    ok = len(fields) == 5 and cls in (0, 1) and all(0 <= v <= 1 for v in coords)
                except (ValueError, IndexError):
                    ok = False
                if not ok:
                    invalid += 1
                else:
                    class_counts[cls] += 1
        rows.append((split, len(images), missing, invalid))
        if len(images) != EXPECTED[split]:
            errors.append(f"{split}: found {len(images)} images; expected {EXPECTED[split]}")
        if missing:
            errors.append(f"{split}: {missing} images have no matching label file")
        if invalid:
            errors.append(f"{split}: {invalid} label rows are invalid")
    print("\nSplit   Images   Missing labels   Invalid lines")
    for row in rows:
        print(f"{row[0]:<7} {row[1]:>6} {row[2]:>16} {row[3]:>15}")
    print(f"\nClass 0 smoking boxes: {class_counts[0]}")
    print(f"Class 1 nosmoke boxes: {class_counts[1]}")
    if errors:
        print("\nDataset validation found the following problems:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print("\nDataset validation passed.")


if __name__ == "__main__":
    main()

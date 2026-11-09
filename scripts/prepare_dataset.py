"""Convert the released COCO annotations to YOLO labels and verify the split."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "dataset" / "Cigarette-ghnlk"
SPLITS = ("train", "valid", "test")
EXPECTED_IMAGES = {"train": 5000, "valid": 1000, "test": 1000}
EXPECTED_BOXES = {"train": 18500, "valid": 3600, "test": 3700}
EXPECTED_CLASSES = {0: "smoking", 1: "nosmoke"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help="write dataset/MANIFEST.sha256 for integrity checking",
    )
    return parser.parse_args()


def convert_split(dataset_root: Path, split: str) -> tuple[int, int]:
    split_root = dataset_root / split
    annotation_path = split_root / "_annotations.coco.json"
    image_root = split_root / "images"
    label_root = split_root / "labels"
    if not annotation_path.is_file():
        raise FileNotFoundError(f"Missing COCO annotation file: {annotation_path}")
    if not image_root.is_dir():
        raise FileNotFoundError(f"Missing image directory: {image_root}")

    document = json.loads(annotation_path.read_text(encoding="utf-8"))
    categories = {int(item["id"]): item["name"] for item in document["categories"]}
    if categories != EXPECTED_CLASSES:
        raise ValueError(f"Unexpected class mapping in {annotation_path}: {categories}")

    images = {int(item["id"]): item for item in document["images"]}
    annotations_by_image: dict[int, list[dict]] = defaultdict(list)
    for annotation in document["annotations"]:
        annotations_by_image[int(annotation["image_id"])].append(annotation)

    if len(images) != EXPECTED_IMAGES[split]:
        raise ValueError(
            f"{split} contains {len(images)} images; expected {EXPECTED_IMAGES[split]}"
        )
    if len(document["annotations"]) != EXPECTED_BOXES[split]:
        raise ValueError(
            f"{split} contains {len(document['annotations'])} boxes; "
            f"expected {EXPECTED_BOXES[split]}"
        )

    label_root.mkdir(parents=True, exist_ok=True)
    converted_boxes = 0
    for image_id, image in images.items():
        file_name = Path(image["file_name"]).name
        image_path = image_root / file_name
        if not image_path.is_file():
            raise FileNotFoundError(f"Image referenced by COCO JSON is missing: {image_path}")
        width = float(image["width"])
        height = float(image["height"])
        rows: list[str] = []
        for annotation in sorted(annotations_by_image[image_id], key=lambda item: int(item["id"])):
            category_id = int(annotation["category_id"])
            if category_id not in EXPECTED_CLASSES:
                raise ValueError(f"Unsupported category id {category_id} in {annotation_path}")
            x, y, box_width, box_height = map(float, annotation["bbox"])
            x_center = (x + box_width / 2.0) / width
            y_center = (y + box_height / 2.0) / height
            normalized_width = box_width / width
            normalized_height = box_height / height
            values = (x_center, y_center, normalized_width, normalized_height)
            if not all(0.0 <= value <= 1.0 for value in values):
                raise ValueError(f"Out-of-range box in {annotation_path}: {annotation['bbox']}")
            rows.append(
                f"{category_id} {x_center:.8f} {y_center:.8f} "
                f"{normalized_width:.8f} {normalized_height:.8f}"
            )
            converted_boxes += 1
        (label_root / Path(file_name).with_suffix(".txt")).write_text(
            "\n".join(rows) + ("\n" if rows else ""),
            encoding="utf-8",
        )

    return len(images), converted_boxes


def write_manifest(dataset_root: Path) -> Path:
    manifest_path = dataset_root.parent / "MANIFEST.sha256"
    files = sorted(
        path
        for path in dataset_root.rglob("*")
        if path.is_file() and path.name != manifest_path.name
    )
    rows = []
    for path in files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        relative = path.relative_to(dataset_root.parent).as_posix()
        rows.append(f"{digest}  {relative}")
    manifest_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return manifest_path


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset.resolve()
    totals = []
    for split in SPLITS:
        image_count, box_count = convert_split(dataset_root, split)
        totals.append((split, image_count, box_count))
        print(f"{split:<5}: {image_count:>4} images, {box_count:>5} boxes")
    print(
        f"total: {sum(item[1] for item in totals)} images, "
        f"{sum(item[2] for item in totals)} boxes"
    )
    if args.write_manifest:
        manifest_path = write_manifest(dataset_root)
        print(f"SHA-256 manifest written to: {manifest_path}")


if __name__ == "__main__":
    main()

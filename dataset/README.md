# Smoking detection dataset

The dataset is stored as ordinary, directly browsable files. No RAR or ZIP
archive is required.

## Structure

```text
Cigarette-ghnlk/
  train/images/                  5,000 JPEG images
  train/labels/                  5,000 YOLO label files
  train/_annotations.coco.json   18,500 COCO boxes
  valid/images/                  1,000 JPEG images
  valid/labels/                  1,000 YOLO label files
  valid/_annotations.coco.json    3,600 COCO boxes
  test/images/                   1,000 JPEG images
  test/labels/                   1,000 YOLO label files
  test/_annotations.coco.json     3,700 COCO boxes
```

All images are 800 x 800 pixels. YOLO rows use
`class_id x_center y_center width height`, with coordinates normalized to
`[0, 1]`.

## Class definitions

- `0 smoking`: a person exhibiting visible smoking behavior. The annotation
  uses the cigarette/hand-to-mouth interaction and the surrounding local
  context rather than smoke visibility alone.
- `1 nosmoke`: a person included as a hard negative who is not exhibiting
  smoking behavior in the annotated instance. It is an object class, not the
  absence of every smoking object in the entire image.

## Integrity and conversion

Run:

```bash
python scripts/prepare_dataset.py --write-manifest
python scripts/check_dataset.py
```

`prepare_dataset.py` deterministically converts the released COCO boxes to YOLO
labels. `MANIFEST.sha256` records the bytes of every released image,
annotation, and label file.

On Linux, verify the complete release with:

```bash
(cd dataset && sha256sum --check MANIFEST.sha256)
```

## Data rights

The AGPL-3.0 software license in the repository does not grant rights to source
images. Dataset users must follow the terms of the image sources and applicable
privacy and research-ethics requirements. The submitting authors should state
the final image provenance and redistribution permission in the article's Data
Availability Statement before publication.

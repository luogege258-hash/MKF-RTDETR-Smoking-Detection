# Released checkpoint

Place the actual formula-aligned full-model checkpoint at:

```text
weights/E04_full_best.pt
```

The checkpoint must come from the same code revision and data split released in
this repository. Do not rename an older or structurally incompatible
checkpoint. After adding it, publish its SHA-256 digest in this file and in the
GitHub Release notes.

Validation:

```bash
python scripts/validate_chapter3.py --weights weights/E04_full_best.pt --batch 8
python scripts/benchmark_chapter3.py --weights weights/E04_full_best.pt --device 0
```

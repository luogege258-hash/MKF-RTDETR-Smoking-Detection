#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"
source .venv/bin/activate
BATCH="${BATCH:-16}"
DEVICE="${DEVICE:-0}"
mkdir -p logs
python -u scripts/train_chapter3.py --exp E04_full --batch "$BATCH" --device "$DEVICE" \
  2>&1 | tee "logs/E04_full_$(date +%Y%m%d_%H%M%S).log"

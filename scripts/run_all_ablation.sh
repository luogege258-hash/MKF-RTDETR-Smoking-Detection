#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"
source .venv/bin/activate
BATCH="${BATCH:-16}"
DEVICE="${DEVICE:-0}"
mkdir -p logs
python -u scripts/train_chapter3.py --exp all --batch "$BATCH" --device "$DEVICE" \
  2>&1 | tee "logs/chapter3_all_$(date +%Y%m%d_%H%M%S).log"

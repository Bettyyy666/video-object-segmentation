#!/usr/bin/env bash
# usage: bash run_all.sh <video_dir> [device: auto|cuda|mps] [sam2: tiny|small|base_plus|large]
set -e
VID_DIR=${1:-data}
DEV=${2:-auto}
SAM=${3:-base_plus}
mkdir -p results
for v in "$VID_DIR"/*.mp4; do
  python segment.py "$v" --task-sheets data/task_sheets.json --out results --device "$DEV" --sam2 "$SAM" | tee -a results/run_log.txt
done

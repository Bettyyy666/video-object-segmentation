#!/bin/bash
# One zip for track 2 (run from the repo root):  ../objseg_track2.zip
#   <repo>/            code + README (no videos, no task sheet, no weights, no work files)
#   <repo>/results/    <video>.json + <video>_overlay.mp4 for the three required videos
# usage: bash pack.sh [results_dir]      (default: results_full)
set -e
cd "$(dirname "$0")"
R=${1:-results_full}
name=$(basename "$PWD")
out="$(cd .. && pwd)/objseg_track2.zip"
rm -f "$out"
( cd .. && zip -qr "$out" "$name" \
    -x "$name/task_sheets.json" "$name/videos/*" "$name/data/*" "$name/results*" "$name/checkpoints/*" \
       "$name/weights/*" "$name/*.pt" "$name/logs/*" "$name/report_all.txt" "*/__pycache__/*" "*.DS_Store" )
stage=$(mktemp -d -p ..)
mkdir -p "$stage/$name/results"
for v in 09_coil_and_stow_four_cables 12_fold_four_towels 26_sort_cutlery_drawer_insert; do
  for f in "$R/$v.json" "$R/${v}_overlay.mp4"; do
    [ -f "$f" ] || { echo "missing $f"; rm -rf "$stage"; exit 1; }
    cp "$f" "$stage/$name/results/"
  done
done
( cd "$stage" && zip -qr "$out" "$name/results" )
rm -rf "$stage"
ls -lh "$out"
unzip -l "$out" | grep -E "results/|README|segment.py"; unzip -l "$out" | tail -1

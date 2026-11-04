#!/usr/bin/env bash
# Run ONE video, time it, run the self-check, and collect everything the README needs.
# usage: bash run_required.sh <video_dir> [sam2: tiny|small|base_plus|large]
set -e
VIDEO=12_fold_four_towels_15s          # <- edit: file name in <video_dir>, without .mp4
VID_DIR=${1:-data}
SAM=${2:-small}
mkdir -p results/check
LOG=results/required_log.txt
{
  echo "== hardware"
  (system_profiler SPHardwareDataType 2>/dev/null | grep -E "Model Name|Chip|Total Number of Cores|Memory") \
    || nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || true
  python -c "import torch, platform; print('torch', torch.__version__, '| mps', torch.backends.mps.is_available(), '| cuda', torch.cuda.is_available(), '|', platform.platform())"
} | tee "$LOG"

for v in $VIDEO; do
  if [ ! -f "$VID_DIR/$v.mp4" ]; then echo "== $v: $VID_DIR/$v.mp4 not found" | tee -a "$LOG"; exit 1; fi
  start=$(date +%s)
  python segment.py "$VID_DIR/$v.mp4" --task-sheets data/task_sheets.json --out results --sam2 "$SAM" 2>&1 | tee -a "$LOG"
  end=$(date +%s)
  echo "== $v wall time: $((end - start)) s (sam2=$SAM)" | tee -a "$LOG"
  python tools/check_output.py "results/$v.json" --video "$VID_DIR/$v.mp4" --overlay "results/${v}_overlay.mp4" \
      --dump "results/check/$v" --sample 20 2>&1 | tee "results/check/$v.txt" | tee -a "$LOG"
done
echo "done -> $LOG ; frames to look at: results/check/<video>/flag_*.jpg and sample_*.jpg"

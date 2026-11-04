#!/usr/bin/env bash
# Local run on an Apple-silicon Mac (tested: MacBook Air M2, 16 GB) for one video, then the self-check.
#   bash mac_run.sh                                         # data/12_fold_four_towels_15s.mp4, sam2=small
#   bash mac_run.sh data/<video>.mp4 [sam2: tiny|small|base_plus|large] [data/task_sheets.json]
# First time only: bash mac_run.sh --setup   (creates conda env objseg, downloads weights into checkpoints/)
# Output: results/<video>.json, results/<video>_overlay.mp4, results/check/<video>/, results/<video>_log.txt
set -eo pipefail
cd "$(dirname "$0")"
ENV=objseg
source "$(conda info --base)/etc/profile.d/conda.sh"

# all model weights stay in checkpoints/ (YOLO-World, its CLIP text encoder, CLIP, Grounding DINO, SAM 2)
export YOLO_CONFIG_DIR="$PWD/checkpoints/ultralytics"    # project-local ultralytics settings: weights_dir=checkpoints
export HF_HOME="$PWD/checkpoints/hf"                     # Grounding DINO (only downloaded if YOLO-World misses a class)
export PYTORCH_ENABLE_MPS_FALLBACK=1

if [ "$1" = "--setup" ]; then
  conda env list | grep -q "^$ENV " || conda create -y -n $ENV python=3.11
  conda activate $ENV
  pip install "torch==2.5.1" "torchvision==0.20.1"   # newer torch calls MPS APIs that need macOS 15 (crash on 14)
  pip install -r requirements.txt
  pip install "git+https://github.com/ultralytics/CLIP.git"
  SAM2_BUILD_CUDA=0 pip install --no-build-isolation "git+https://github.com/facebookresearch/sam2.git"
  mkdir -p checkpoints/clip "$YOLO_CONFIG_DIR"
  for s in tiny small base_plus; do
    [ -f checkpoints/sam2.1_hiera_$s.pt ] || curl -L -o checkpoints/sam2.1_hiera_$s.pt \
      https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_$s.pt
  done
  [ -f checkpoints/yolov8x-worldv2.pt ] || curl -L -o checkpoints/yolov8x-worldv2.pt \
    https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8x-worldv2.pt
  yolo settings weights_dir="$PWD/checkpoints" sync=False >/dev/null
  echo "setup done"
  exit 0
fi

VIDEO=${1:-data/12_fold_four_towels_15s.mp4}
SAM=${2:-small}                         # small fits 16 GB comfortably; base_plus is slower
SHEETS=${3:-data/task_sheets.json}      # client file, gitignored with data/
v=$(basename "$VIDEO" .mp4)

conda activate $ENV
mkdir -p results/check
[ -f "$YOLO_CONFIG_DIR/Ultralytics/settings.json" ] || yolo settings weights_dir="$PWD/checkpoints" sync=False >/dev/null
LOG=results/${v}_log.txt
start=$(date +%s)
python segment.py "$VIDEO" --task-sheets "$SHEETS" --out results --device mps --sam2 "$SAM" 2>&1 | tee "$LOG"
echo "== $v wall time: $(( $(date +%s) - start )) s (sam2=$SAM)" | tee -a "$LOG"
rm -rf "results/check/$v"
python tools/check_output.py "results/$v.json" --video "$VIDEO" --overlay "results/${v}_overlay.mp4" \
    --dump "results/check/$v" --sample 20 2>&1 | tee -a "$LOG"
echo "done -> results/$v.json, results/${v}_overlay.mp4, frames to look at: results/check/$v/"

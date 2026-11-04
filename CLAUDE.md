# CLAUDE.md

Video object segmentation pipeline:
- **YOLO-World** detects objects from an open vocabulary. Categories it cannot see fall back to Grounding DINO.
- **SAM 2** segments and tracks them in 1-second windows.
- **CLIP** gives a second opinion on each whole track's category.

Output is filtered by the objects and counts in the task sheet. Each video produces `results/<video>.json` (per-frame COCO RLE) and `results/<video>_overlay.mp4`. Design rationale and known failure cases are in [README.md](README.md) (Chinese).

## Running from scratch on a new machine

### 1. Things not in git that you must provide

| Path | Contents | Source |
|---|---|---|
| `data/*.mp4` | input videos | client data, copy manually |
| `data/task_sheets.json` | task sheet for all 30 tasks, `{"tasks": [...]}` | client data, copy manually |
| `checkpoints/` | all model weights | downloaded during setup (below) |
| `results/` | outputs | created at run time |

`data/`, `checkpoints/` and `results/` are in `.gitignore`, so never commit them. Without `data/task_sheets.json`, `segment.py` raises `FileNotFoundError` and `tests/test_diagnostics.py` fails.

### 2a. Apple-silicon Mac (MPS)

```bash
conda --version                 # needs conda / miniforge
bash mac_run.sh --setup         # creates conda env objseg, installs deps, downloads weights into checkpoints/
bash mac_run.sh                 # default: data/12_fold_four_towels_15s.mp4, sam2=small
bash mac_run.sh data/<video>.mp4 [tiny|small|base_plus|large] [data/task_sheets.json]
```

What `mac_run.sh` does:
- Sets `YOLO_CONFIG_DIR` and `HF_HOME` so every weight file lands in `checkpoints/`.
- Runs `segment.py` with `--device mps`, then runs `tools/check_output.py`.
- Stops at the first failing step (`set -eo pipefail`).

### 2b. Linux + NVIDIA GPU (e.g. Brown CCV Oscar)

```bash
conda create -y -n objseg python=3.11 && conda activate objseg
pip install torch torchvision                      # default CUDA wheel
pip install -r requirements.txt
pip install "git+https://github.com/ultralytics/CLIP.git"           # seg/verify.py needs `import clip`
pip install --no-build-isolation "git+https://github.com/facebookresearch/sam2.git"
mkdir -p checkpoints/clip
curl -L -o checkpoints/sam2.1_hiera_base_plus.pt https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt
curl -L -o checkpoints/yolov8x-worldv2.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8x-worldv2.pt
export YOLO_CONFIG_DIR=$PWD/checkpoints/ultralytics HF_HOME=$PWD/checkpoints/hf
yolo settings weights_dir=$PWD/checkpoints sync=False     # once: YOLO-World's CLIP text encoder goes to checkpoints/clip
```

Re-export `YOLO_CONFIG_DIR` and `HF_HOME` in every new shell. Without them, Ultralytics and HuggingFace download weights into caches under your home directory. The pipeline still works; the weights just aren't in `checkpoints/`.

- **Oscar, one GPU job per video:** `bash oscar_run.sh [video names...]`. It hard-codes `PY=~/miniforge3/envs/objseg/bin/python` and `base_plus`, and writes logs to `logs/`.
- **Oscar, single job:** `sbatch oscar_job.sh` calls `run_required.sh`.

### 3. Verify the environment (CPU only, under a minute)

```bash
conda activate objseg
python tests/test_finalize.py        # last line should be: OK: ...
python tests/test_tracker_mock.py    # last line should be: OK: one id per object ...
python tests/test_diagnostics.py     # should print: Ran 6 tests ... OK (needs data/task_sheets.json and clip)
```

These are plain scripts, not pytest.

## Commands

```bash
python segment.py data/<video>.mp4 --task-sheets data/task_sheets.json --out results --device auto --sam2 base_plus
python segment.py ... --finalize-only              # reuse tracking in results/work/<video>/, redo outputs only
python segment.py ... --finalize-only --reverify   # same, and recompute CLIP scores
bash run_required.sh data small      # one video: edit VIDEO=<file name without .mp4> at the top of the file
bash run_all.sh data auto base_plus  # every .mp4 in data/
```

Self-check and diagnostics:

```bash
python tools/check_output.py results/<v>.json --video data/<v>.mp4 --overlay results/<v>_overlay.mp4 \
    --dump results/check/<v> --sample 20     # format + temporal consistency; dumps flagged and sampled frames
python tools/report.py results <v>           # expected vs found counts from the task sheet, events per id
python tools/why.py results/work/<v>/detections.jsonl --cat towel --frames 0-300   # why a detection did/didn't get an id
python tools/probe_det.py data/<v>.mp4 --n 16 --device auto --json results/probe_<v>.json  # detector-only vocabulary probe
```

## Code layout

- `segment.py`: entry point. Runs vocab → `Tracker.run` (pass 1) → `verify.run` (CLIP) → `finalize.run` (pass 2).
- `seg/config.py`: every tunable constant and the model paths (`DET_MODEL`, `SAM2_MODELS`, `GDINO_MODEL`).
- `seg/vocab.py`: turns task-sheet text into prompts, categories and expected counts. It matches a video to a task by name after stripping the numeric prefix (`12_fold_four_towels_15s` → `fold_four_towels`). If no name matches, it takes the Nth task by number.
- `configs/overrides.json`: hand-written prompts for the three required videos (09, 12, 26). They are keyed by task name, so `*_15s` clips use them too.
- `seg/tracker.py`: detection, SAM 2 window propagation, re-ID and backward fill. Writes `results/work/<v>/labels/*.png` (uint16 id maps), `tracks.json` and `detections.jsonl`.
- `seg/gdino.py`: Grounding DINO (transformers). Only used for categories YOLO-World cannot see on 16 sampled frames.
- `seg/verify.py`: CLIP ViT-B/32 second opinion; weights are in `checkpoints/clip`.
- `seg/finalize.py`: assigns categories under the counts, merges or drops tracks, and writes the RLE JSON and overlay (ffmpeg encode, CPU only).
- `seg/ffmpeg.py`: uses the system ffmpeg if present, otherwise the binary bundled with `imageio-ffmpeg`. No separate ffmpeg install is needed.

## Known pitfalls

- **macOS 14 cannot use a recent torch.** torch 2.15 crashes on MPS with `-[MPSNDArray arrayViewWithShape:strides:]: unrecognized selector` because that API needs macOS 15. `mac_run.sh --setup` pins `torch==2.5.1` and `torchvision==0.20.1`. SAM 2 requires torch ≥ 2.5.1, so don't go lower. You can drop the pin after upgrading to macOS 15.
- **Mac: don't build SAM 2's CUDA extension.** Install it with `SAM2_BUILD_CUDA=0`.
- **Speed.** On a MacBook Air M2 (16 GB) with SAM 2 small, the 15 s clip (750 frames) took about 8700 s to track. README's 1064 s figure came from a different Mac. An A5000 with base_plus takes about 130 s on Oscar. Use a GPU for full-length videos.
- **CLIP is not in requirements.txt.** Install it separately with `pip install git+https://github.com/ultralytics/CLIP.git`. Without it, `verify` prints `CLIP not available` and skips the check. Nothing errors, but output quality drops.
- `segment.py` already sets `PYTORCH_ENABLE_MPS_FALLBACK=1`.
- `verify` runs on the GPU only with `--device auto|cuda` and CUDA available. On a Mac, CLIP runs on CPU (about 10 s, which is fine).

## Conventions when changing code (from README)

- Never hand-correct masks, ids or object lists. Rules must be generic across all videos, with no special cases by video number or track id.
- After changing tracking or output logic, run the three CPU tests, then check the 15 s clip with `check_output.py` and `report.py`.

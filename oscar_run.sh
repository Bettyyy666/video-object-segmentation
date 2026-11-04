#!/bin/bash
# All required videos in parallel on Oscar (one GPU job each), then self-check + report.
#   bash oscar_run.sh                      # submits 3 jobs
#   bash oscar_run.sh 26_sort_cutlery_drawer_insert   # just one
# output: results/<video>.json, results/<video>_overlay.mp4, logs/<video>_<jobid>.out
cd "$(dirname "$0")"
OUT=${OUT:-results}                    # OUT=results_full bash oscar_run.sh ...  -> separate output folder
mkdir -p logs "$OUT"
PY=~/miniforge3/envs/objseg/bin/python
VIDS=${@:-09_coil_and_stow_four_cables 12_fold_four_towels 26_sort_cutlery_drawer_insert}
for v in $VIDS; do
  f=$(ls data/$v.mp4 2>/dev/null | head -1)
  [ -z "$f" ] && { echo "$v: not found in data/"; continue; }
  sbatch -J $v -p gpu --gres=gpu:1 -c 4 --mem=32G -t 12:00:00 -o logs/${v}_${OUT}_%j.out --wrap "
export PATH=\$HOME/miniforge3/envs/objseg/bin:/usr/local/bin:/usr/bin:/bin:\$PATH PYTHONNOUSERSITE=1
nvidia-smi --query-gpu=name --format=csv,noheader
start=\$(date +%s)
$PY segment.py $f --task-sheets data/task_sheets.json --out $OUT --sam2 base_plus || exit 1
echo \"== $v wall time: \$((\$(date +%s) - start)) s\"
rm -rf $OUT/check/$v
$PY tools/check_output.py $OUT/$v.json --video $f --overlay $OUT/${v}_overlay.mp4 --dump $OUT/check/$v --sample 20
$PY tools/report.py $OUT $v
"
done
squeue -u $USER

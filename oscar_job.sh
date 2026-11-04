#!/bin/bash
# Slurm job for Brown CCV Oscar: runs the required videos on one GPU.
# submit from the repo folder:   sbatch oscar_job.sh            (SAM 2 base_plus)
#                                SAM=small sbatch oscar_job.sh  (faster)
#SBATCH -J objseg
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 06:00:00
#SBATCH -o logs/objseg_%j.out

set -e
cd "$SLURM_SUBMIT_DIR"
source ~/miniforge3/etc/profile.d/conda.sh
conda activate objseg
nvidia-smi --query-gpu=name,memory.total --format=csv
python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
bash run_required.sh videos "${SAM:-base_plus}"

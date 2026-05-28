#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=256G
#SBATCH -p mit_normal_gpu
#SBATCH -t 6:00:00
#SBATCH -J ob_gnn
#SBATCH -o logs/ob_gnn_%j.out
#SBATCH -e logs/ob_gnn_%j.err

set -eo pipefail

mkdir -p logs

# (Optional but recommended) make sure bashrc is loaded so "conda" is defined
# Temporarily disable nounset while sourcing bashrc
set +u
source ~/.bashrc
set -u

module load miniforge
module load cuda
mamba activate ob_gnn

set -euo pipefail

export DGLBACKEND=pytorch
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

echo "[$(date)] Starting training on node $HOSTNAME"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"

# Run training
# example run train on bump A
srun python3 -u /orcd/home/002/jlu25/cfd_obgnn/src/main.py --epochs 20 --batch_size 128 --lr 1.5e-4 --bump_train_version train_A --checkpoint_dir ./model_checkpoints/checkpoints_A
# example run train on airfrans train only
srun python3 -u /orcd/home/002/jlu25/cfd_obgnn/src/main.py --epochs 12 --batch_size 128 --lr 1.5e-4 --checkpoint_dir ./model_checkpoints/checkpoints_O



echo "[$(date)] Finished job $SLURM_JOB_ID"

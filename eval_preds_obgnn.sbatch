#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=256G
#SBATCH -p mit_normal_gpu
#SBATCH -t 6:00:00
#SBATCH -J pred_ob_gnn
#SBATCH -o pred_logs/ob_gnn_%j.out
#SBATCH -e pred_logs/ob_gnn_%j.err

set -eo pipefail

mkdir -p pred_logs

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

echo "[$(date)] Starting inference on node $HOSTNAME"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"

# example evaluate and save preds of predictions of obgnn_A epoch 12
srun python3 -u /orcd/home/002/jlu25/cfd_obgnn/src/evaluate_predictions.py --model_version A_ep0012 --split test_bump --checkpoints_dir model_checkpoints/checkpoints_A --checkpoint_epoch 0012 --mse_csv A_summary_mse.csv --save_predictions True

echo "[$(date)] Finished job $SLURM_JOB_ID"

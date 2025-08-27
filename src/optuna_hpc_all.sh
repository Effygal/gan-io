#!/bin/bash
# =============================================================================
# HPC All Traces Optimization Script (SLURM Array)
# =============================================================================
# 
# USAGE:
#   sbatch optuna_hpc_all.sh
# 
# DESCRIPTION:
#   Runs Optuna optimization for ALL 8 traces in parallel using SLURM array
#   Each trace gets its own GPU node and runs independently
# 
# TRACES RUN:
#   v521, v766, v827, v538, w44, w82, w24, w11
# 
# RESOURCES:
#   - 8 parallel jobs (one per trace)
#   - Each job: 1 GPU, 4 CPUs, 16GB RAM, 12 hours max
#   - Output: logs/optuna_gan_<job_id>_<array_index>.out
#   - Error: logs/optuna_gan_<job_id>_<array_index>.err
# 
# MONITORING:
#   squeue -u $USER                    # Check job status
#   sacct -u $USER                     # Check job history
#   tail -f logs/optuna_gan_<id>_<idx>.out  # Monitor specific trace
# =============================================================================

#SBATCH --job-name=optuna_gan
#SBATCH --partition=short          # use gpu-short if each run <= 2h
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --array=0-7%8              # 8 traces -> indices 0..7; run up to 8 concurrently
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

module purge
module load anaconda3               # keep your local module name/version if different
# If conda is not yet initialized for non-interactive shells:
eval "$(conda shell.bash hook)"
conda activate pytorch_env

mkdir -p logs

# List your traces here (index must match the --array range above)
TRACES=(v521 v766 v827 v538 w44 w82 w24 w11)
TRACE="${TRACES[$SLURM_ARRAY_TASK_ID]}"

echo "[$(date)] Starting ${TRACE} on host ${HOSTNAME} (GPU assigned via Slurm)"
srun python optune.py \
  --trace-name "$TRACE" \
  --device cuda \
  --n-trials 50

echo "[$(date)] Done ${TRACE}"
#!/bin/bash
# =============================================================================
# HPC Single Trace Optimization Script
# =============================================================================
# 
# USAGE:
#   sbatch optuna_hpc_single.sh <trace_name> [n_trials]
# 
# EXAMPLES:
#   sbatch optuna_hpc_single.sh v521 50        # Run v521 with 50 trials
#   sbatch optuna_hpc_single.sh w44 30         # Run w44 with 30 trials
#   sbatch optuna_hpc_single.sh v766           # Run v766 with default 50 trials
# 
# AVAILABLE TRACES:
#   v521, v766, v827, v538, w44, w82, w24, w11
# 
# RESOURCES:
#   - 1 GPU, 4 CPUs, 16GB RAM, 12 hours max
#   - Output: logs/optuna_gan_<job_id>.out
#   - Error: logs/optuna_gan_<job_id>.err
# =============================================================================

#SBATCH --job-name=optuna_gan
#SBATCH --partition=gpu            # use gpu-short if each run <= 2h
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

module purge
module load anaconda3               # keep your local module name/version if different
# If conda is not yet initialized for non-interactive shells:
eval "$(conda shell.bash hook)"
conda activate pytorch_env

mkdir -p logs

# Get trace name from command line argument
TRACE_NAME=${1:-"v521"}
N_TRIALS=${2:-50}

echo "[$(date)] Starting ${TRACE_NAME} on host ${HOSTNAME} (GPU assigned via Slurm)"
srun python optune.py \
  --trace-name "$TRACE_NAME" \
  --device cuda \
  --n-trials "$N_TRIALS"

echo "[$(date)] Done ${TRACE_NAME}"

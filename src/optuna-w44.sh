#!/bin/bash
#SBATCH --partition=multigpu
#SBATCH --gres=gpu:v100-sxm2:8
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=10:00:00
#SBATCH --job-name=optuna_gan_tuning
#SBATCH -p reservation
#SBATCH --reservation=<reservation_name>
#SBATCH --output=output_%j.txt
#SBATCH --error=error_%j.txt

# module load anaconda3/2022.05 cuda/12.1
# conda create --name pytorch_env -c conda-forge python=3.10 -y
# source activate pytorch_env
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# conda install -c conda-forge optuna scikit-learn -y
conda activate pytorch_env
STORAGE="sqlite:///optuna_gan.db"
for i in {0..7}
do
    srun --mpi=pmi2 python optuna-w44.py --storage $STORAGE --study-name "gan_w44" > optuna_task_${i}.out 2>&1 &
done

wait
echo "Optuna parallel tuning completed at $(date)"

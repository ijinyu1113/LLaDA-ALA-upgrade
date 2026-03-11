#!/bin/bash
#SBATCH --job-name=ala-router-h100
#SBATCH --output=logs/finetune-gpu-%j.out
#SBATCH --error=logs/finetune-gpu-%j.err
#SBATCH --partition=ghx4
#SBATCH --account=benv-dtai-gh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=verbose,closest
#SBATCH --time=24:00:00

export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export MKL_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16
export VECLIB_MAXIMUM_THREADS=16
export TOKENIZERS_PARALLELISM=false

export TORCHDYNAMO_DISABLE=1

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0

# Load compiler + conda
##############################
module purge
module load gcc

# Make conda available (update path to your Miniforge)
source /sw/user/python/miniforge3-pytorch-24.11.3/etc/profile.d/conda.sh
conda activate nim-env

##############################
# Quick sanity check
##############################
echo "========================================="
echo "JOB $SLURM_JOB_ID on $(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

python -c "
import torch
import transformers
print(f'PyTorch {torch.__version__}')
print(f'Transformers {transformers.__version__}')
print(f'CUDA {torch.version.cuda}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'BF16 supported? {torch.cuda.is_bf16_supported()}')
"
echo "========================================="

# Step 1: Retrain router at α=0.1 (~1h at 10K steps)
time srun python3 /u/iyu1/nim_game_project/llada/train_router.py && \
# Step 2: Eval all benchmarks at inference α=0.02 (200 samples each)
time srun python3 /u/iyu1/nim_game_project/llada/run_benchmarks.py --benchmarks gsm8k math arc gpqa bbh


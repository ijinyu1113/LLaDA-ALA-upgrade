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
#SBATCH --time=8:00:00

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

module reset
module load gcc
module load python/miniforge3_pytorch/2.7.0

source $(conda info --base)/etc/profile.d/conda.sh
conda activate nim-env

echo "========================================="
echo "JOB $SLURM_JOB_ID on $(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA {torch.version.cuda}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'BF16: {torch.cuda.is_bf16_supported()}')
"
echo "========================================="

time srun python3 /u/iyu1/nim_game_project/llada/train_router.py

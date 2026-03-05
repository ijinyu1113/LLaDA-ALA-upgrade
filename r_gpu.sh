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

time srun python3 /u/iyu1/nim_game_project/llada/train_router.py && \
time srun python3 /u/iyu1/nim_game_project/llada/test_router.py
#!/bin/bash
#SBATCH --job-name=deep-ebm-train
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=5-00:00:00
#SBATCH --array=0-3
#SBATCH --output=/projects/prjs1237/project/tabpfgen/discrete-langevin/logs/deep_ebm_train_%A_%a.out
#SBATCH --error=/projects/prjs1237/project/tabpfgen/discrete-langevin/logs/deep_ebm_train_%A_%a.err

set -euo pipefail

PROJECT_ROOT="/projects/prjs1237/project/tabpfgen/discrete-langevin"
CONDA_ROOT="/home/xyan1/anaconda3"

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate tabpfgen
cd "$PROJECT_ROOT"

# Four independent Table 1 runs: dataset x sampler.  An array lets Slurm place
# them on separate GPUs/nodes.  No multi-GPU training is needed.
DATASETS=(static_mnist static_mnist omniglot omniglot)
SAMPLERS=(dula dmala dula dmala)
STEP_SIZES=(0.1 0.15 0.1 0.15)

DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
SAMPLER="${SAMPLERS[$SLURM_ARRAY_TASK_ID]}"
STEP_SIZE="${STEP_SIZES[$SLURM_ARRAY_TASK_ID]}"
OUTPUT_DIR="$PROJECT_ROOT/results/deep_ebm/$DATASET/$SAMPLER"
mkdir -p "$OUTPUT_DIR"

if [[ "$DATASET" == "static_mnist" ]]; then
    test -s datasets/MNIST_static/binarized_mnist_train.amat
    test -s datasets/MNIST_static/binarized_mnist_valid.amat
    test -s datasets/MNIST_static/binarized_mnist_test.amat
else
    test -s datasets/OMNIGLOT/chardata.mat
fi

PYTHON_BIN="$(command -v python)"
echo "Node: $(hostname)"
echo "Python: $PYTHON_BIN"
echo "Experiment: dataset=$DATASET sampler=$SAMPLER step_size=$STEP_SIZE"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L
"$PYTHON_BIN" - <<'PY'
import torch
import torchvision
import numpy
import scipy
import tqdm
import matplotlib
print("PyTorch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("This job received no CUDA-capable PyTorch device")
print("GPU:", torch.cuda.get_device_name(0))
PY

"$PYTHON_BIN" -u pcd_ebm_ema.py \
    --save_dir "$OUTPUT_DIR" \
    --dataset_name "$DATASET" \
    --sampler "$SAMPLER" \
    --step_size "$STEP_SIZE" \
    --sampling_steps 40 \
    --model resnet-64 \
    --lr 0.0001 \
    --n_iters 50000 \
    --buffer_size 10000 \
    --eval_every 5000 \
    --eval_sampling_steps 10000 \
    --base_dist \
    --buffer_init mean \
    --reinit_freq 0.0 \
    --batch_size 100 \
    --test_batch_size 100 \
    --warmup_iters 10000 \
    --ema 0.999 \
    --weight_decay 0.0 \
    --p_control 0.0 \
    --l2 0.0 \
    --print_every 10 \
    --viz_every 100 \
    --seed 1234567

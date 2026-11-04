#!/bin/bash
#SBATCH --job-name=deep-ebm-ais
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=5-00:00:00
#SBATCH --array=0-3
#SBATCH --output=/projects/prjs1237/project/tabpfgen/discrete-langevin/logs/deep_ebm_ais_%A_%a.out
#SBATCH --error=/projects/prjs1237/project/tabpfgen/discrete-langevin/logs/deep_ebm_ais_%A_%a.err

set -euo pipefail

PROJECT_ROOT="/projects/prjs1237/project/tabpfgen/discrete-langevin"
CONDA_ROOT="/home/xyan1/anaconda3"

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate tabpfgen
cd "$PROJECT_ROOT"

DATASETS=(static_mnist static_mnist omniglot omniglot)
TRAIN_SAMPLERS=(dula dmala dula dmala)
TRAIN_STEP_SIZES=(0.1 0.15 0.1 0.15)

DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
TRAIN_SAMPLER="${TRAIN_SAMPLERS[$SLURM_ARRAY_TASK_ID]}"
TRAIN_STEP_SIZE="${TRAIN_STEP_SIZES[$SLURM_ARRAY_TASK_ID]}"
OUTPUT_DIR="$PROJECT_ROOT/results/deep_ebm/$DATASET/$TRAIN_SAMPLER"
CHECKPOINT="$OUTPUT_DIR/best_ckpt_${DATASET}_${TRAIN_SAMPLER}_${TRAIN_STEP_SIZE}.pt"

if [[ ! -s "$CHECKPOINT" ]]; then
    echo "Missing best-validation checkpoint: $CHECKPOINT" >&2
    exit 1
fi

PYTHON_BIN="$(command -v python)"
echo "Node: $(hostname)"
echo "Python: $PYTHON_BIN"
echo "Evaluating: dataset=$DATASET trained_with=$TRAIN_SAMPLER step_size=$TRAIN_STEP_SIZE"
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

# The trained model is selected by TRAIN_SAMPLER.  As in the released ais.sh,
# GWG is deliberately used for every final AIS estimate for a fair comparison.
"$PYTHON_BIN" -u eval_ais.py \
    --save_dir "$OUTPUT_DIR" \
    --dataset_name "$DATASET" \
    --eval_sampler "$TRAIN_SAMPLER" \
    --eval_step_size "$TRAIN_STEP_SIZE" \
    --sampler gwg \
    --step_size 0.15 \
    --sampling_steps 40 \
    --model resnet-64 \
    --buffer_size 10000 \
    --n_iters 300000 \
    --base_dist \
    --n_samples 500 \
    --eval_sampling_steps 300000 \
    --ema \
    --batch_size 100 \
    --test_batch_size 100 \
    --viz_every 1000 \
    --seed 1234567

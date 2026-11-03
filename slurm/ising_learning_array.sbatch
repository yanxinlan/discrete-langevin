#!/bin/bash
#SBATCH --job-name=ising-learn
#SBATCH --array=0-11
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=/projects/prjs1237/project/tabpfgen/discrete-langevin/slurm/logs/%x-%A_%a.out
#SBATCH --error=/projects/prjs1237/project/tabpfgen/discrete-langevin/slurm/logs/%x-%A_%a.err

set -euo pipefail

# The job requests node-level memory with --mem. Clear memory-mode variables
# that may be inherited from the shell that called sbatch; srun rejects mixtures.
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU

CONDA_ROOT="/home/xyan1/anaconda3"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate tabpfgen

# Fixed paper/released-code configuration for Figure 5(a,b).
PROJECT_ROOT="/projects/prjs1237/project/tabpfgen/discrete-langevin"
PYTHON_BIN="$CONDA_PREFIX/bin/python"
SAMPLERS=(gibbs-1 gwg-1 dula dmala)
SAMPLER="${SAMPLERS[$((SLURM_ARRAY_TASK_ID % 4))]}"
REPLICATE="$((SLURM_ARRAY_TASK_ID / 4))"
SEED="$((123456 + REPLICATE))"
DATA_PATH="$PROJECT_ROOT/data/ising/lattice_25_coupling_0.25_gibbs1000000.pt"
OUTPUT_DIR="$PROJECT_ROOT/results/ising/figure5_learning/learning/${SAMPLER}_seed${SEED}"
cd "$PROJECT_ROOT"
mkdir -p "$OUTPUT_DIR"

# Paper Appendix I.4: 100 negative-phase steps for the main curves, PCD buffer
# 256, batch 256, Adam lr=0.001, 10,000 iterations, and L1 coefficient 0.01.
# DULA alpha=0.1 and DMALA alpha=0.15 are selected by learn.py defaults.
# UNPUBLISHED CHOICE: the paper does not state a seed/repetition count; this
# array uses three consecutive seeds while retaining released default 123456.
srun "$PYTHON_BIN" -m ising_pipeline.learn \
  --data "$DATA_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --sampler "$SAMPLER" \
  --sampling-steps 100 \
  --iterations 10000 \
  --batch-size 256 \
  --buffer-size 256 \
  --learning-rate 0.001 \
  --l1 0.01 \
  --record-every 1000 \
  --seed "$SEED" \
  --device cuda

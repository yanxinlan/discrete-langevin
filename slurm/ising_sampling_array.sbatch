#!/bin/bash
#SBATCH --job-name=ising-fig3
#SBATCH --array=0-17
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --output=/projects/prjs1237/project/tabpfgen/discrete-langevin/slurm/logs/%x-%A_%a.out
#SBATCH --error=/projects/prjs1237/project/tabpfgen/discrete-langevin/slurm/logs/%x-%A_%a.err

set -euo pipefail

# The job requests node-level memory with --mem. Clear memory-mode variables
# that may be inherited from the shell that called sbatch; srun rejects mixtures.
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU

CONDA_ROOT="/home/xyan1/anaconda3"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate tabpfgen

# Self-contained reproduction configuration. No environment variables are read.
PROJECT_ROOT="/projects/prjs1237/project/tabpfgen/discrete-langevin"
PYTHON_BIN="$CONDA_PREFIX/bin/python"
OUTPUT_DIR="$PROJECT_ROOT/results/ising/figure3_sampling/sampling"

# Paper/source settings:
#   target: periodic 5x5 lattice, coupling a=0.1, bias b=0.2
#   source defaults: 50,000 transitions, 2 chains, seed 1234567
#   DULA alpha=0.2; DMALA alpha=0.4
#   exact target mean: enumeration of all 2^25 states
SIDE=5
COUPLING=0.1
BIAS=0.2
STEPS=50000
CHAINS=2
RECORD_EVERY=1000
EXACT_CHUNK_SIZE=1000000

# Focused six-method reproduction with three seeds. The complete published
# figure additionally contains HB-10-1, LB-1, DSVGD, and RMALA.
SAMPLERS=(gibbs-1 gibbs-4 gwg-1 gwg-4 dula dmala)
STEP_SIZES=(1.0 1.0 1.0 1.0 0.2 0.4)
SAMPLER_INDEX="$((SLURM_ARRAY_TASK_ID % 6))"
REPLICATE="$((SLURM_ARRAY_TASK_ID / 6))"
SAMPLER="${SAMPLERS[$SAMPLER_INDEX]}"
STEP_SIZE="${STEP_SIZES[$SAMPLER_INDEX]}"
SEED="$((1234567 + REPLICATE))"

# Remaining differences:
# MISALIGNMENT: its ESS is based on recorded scalar magnetization; the original
# computes per-coordinate ESS from every transition after 10% burn-in.

cd "$PROJECT_ROOT"
mkdir -p "$OUTPUT_DIR"

srun "$PYTHON_BIN" -m ising_pipeline.benchmark_sampling \
  --output "$OUTPUT_DIR/${SAMPLER}_seed${SEED}.npz" \
  --sampler "$SAMPLER" \
  --side "$SIDE" \
  --coupling "$COUPLING" \
  --bias "$BIAS" \
  --step-size "$STEP_SIZE" \
  --steps "$STEPS" \
  --chains "$CHAINS" \
  --record-every "$RECORD_EVERY" \
  --reference exact \
  --exact-chunk-size "$EXACT_CHUNK_SIZE" \
  --seed "$SEED" \
  --device cuda

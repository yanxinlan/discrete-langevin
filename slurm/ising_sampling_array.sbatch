#!/bin/bash
#SBATCH --job-name=ising-fig3
#SBATCH --array=0-3
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --output=/projects/prjs1237/project/tabpfgen/discrete-langevin/slurm/logs/%x-%A_%a.out
#SBATCH --error=/projects/prjs1237/project/tabpfgen/discrete-langevin/slurm/logs/%x-%A_%a.err

set -euo pipefail

# Self-contained reproduction configuration. No environment variables are read.
PROJECT_ROOT="/projects/prjs1237/project/tabpfgen/discrete-langevin"
PYTHON_BIN="python"
OUTPUT_DIR="$PROJECT_ROOT/results/ising/figure3_sampling"

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
SEED=1234567
RECORD_EVERY=1000
EXACT_CHUNK_SIZE=1000000

# This is the supported four-method subset, not the complete published Figure 3.
# MISALIGNMENT: the published figure also contains Gibbs-4, GWG-4, HB-10-1,
# LB-1, DSVGD, and RMALA. Those methods are not implemented by the added pipeline.
SAMPLERS=(gibbs gwg dula dmala)
STEP_SIZES=(1.0 1.0 0.2 0.4)
SAMPLER="${SAMPLERS[$SLURM_ARRAY_TASK_ID]}"
STEP_SIZE="${STEP_SIZES[$SLURM_ARRAY_TASK_ID]}"

# Known Python-level differences that cannot be selected with current CLI flags:
# MISALIGNMENT: benchmark_sampling.py initializes Bernoulli(0.5); the released
# original code initializes Bernoulli(sigmoid(2*b)) = Bernoulli(sigmoid(0.4)).
# MISALIGNMENT: benchmark_sampling.py pools the two chains before RMSE; the
# original code computes RMSE over separate per-chain running means.
# MISALIGNMENT: its Gibbs kernel is fresh random-scan; the active original BG-1
# kernel follows a randomly permuted coordinate order until a sweep is complete.
# MISALIGNMENT: its ESS is based on recorded scalar magnetization; the original
# computes per-coordinate ESS from every transition after 10% burn-in.
# MISALIGNMENT: CUDA is not synchronized at timing boundaries, so runtime curves
# are not directly comparable to the paper or across samplers.

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

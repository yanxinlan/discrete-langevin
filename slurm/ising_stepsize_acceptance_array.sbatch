#!/usr/bin/env bash
#SBATCH --job-name=ising-fig2-left
#SBATCH --array=0-4
#SBATCH --account=prjs1237
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=/projects/prjs1237/project/tabpfgen/discrete-langevin/slurm/logs/%x-%A_%a.out
#SBATCH --error=/projects/prjs1237/project/tabpfgen/discrete-langevin/slurm/logs/%x-%A_%a.err

set -euo pipefail

# Self-contained reproduction configuration. No environment variables are read.
PROJECT_ROOT="/projects/prjs1237/project/tabpfgen/discrete-langevin"
PYTHON_BIN="python"
OUTPUT_DIR="$PROJECT_ROOT/results/ising/figure2_stepsize_acceptance"

# Explicit paper settings for Figure 2 (left):
#   target: periodic 5x5 lattice, coupling a=0.1, bias b=0.2
#   sampler: DMALA
#   alpha grid: 0.2, 0.3, 0.4, 0.5, 0.6
SIDE=5
COUPLING=0.1
BIAS=0.2
STEP_SIZES=(0.2 0.3 0.4 0.5 0.6)
STEP_SIZE="${STEP_SIZES[$SLURM_ARRAY_TASK_ID]}"

# The paper and released repository do not disclose the transition count, chain
# count, seed, burn-in, or repetition protocol used for Figure 2 (left).
# UNPUBLISHED CHOICE: use the main released Ising script defaults below.
STEPS=50000
CHAINS=2
SEED=1234567

# Record every transition so JSON means are true per-transition means. This also
# avoids the added pipeline's unequal-interval summary weighting issue.
# UNPUBLISHED CHOICE: the paper does not state its measurement frequency.
RECORD_EVERY=1
EXACT_CHUNK_SIZE=1000000

# The orange curve is interpreted as proposed Hamming distance. This is inferred
# from the reported alpha=0.6 result (about 6 proposed changes, 52% acceptance);
# the paper does not explicitly say proposed versus accepted Hamming distance.
# Read mean_acceptance and mean_proposed_hops from the output JSON. Do not use
# mean_realized_hops for the orange curve.

# Known Python-level difference:
# MISALIGNMENT: benchmark_sampling.py initializes Bernoulli(0.5), whereas the
# released original code initializes Bernoulli(sigmoid(0.4)). The long run makes
# initialization less influential, but it is not an identical protocol.
# NOTE: runtime is not a metric for this panel. record-every=1 and unsynchronized
# CUDA timing make the saved seconds unsuitable for a runtime comparison.

cd "$PROJECT_ROOT"
mkdir -p "$OUTPUT_DIR"

srun "$PYTHON_BIN" -m ising_pipeline.benchmark_sampling \
  --output "$OUTPUT_DIR/dmala_alpha${STEP_SIZE}_seed${SEED}.npz" \
  --sampler dmala \
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

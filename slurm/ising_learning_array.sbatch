#!/usr/bin/env bash
#SBATCH --job-name=ising-learn
#SBATCH --array=0-11
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/%x-%A_%a.out
#SBATCH --error=slurm/logs/%x-%A_%a.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SAMPLERS=(gibbs gwg dula dmala)
SAMPLER="${SAMPLERS[$((SLURM_ARRAY_TASK_ID % 4))]}"
REPLICATE="$((SLURM_ARRAY_TASK_ID / 4))"
SEED="$((1234600 + REPLICATE))"
DATA_PATH="${ISING_DATA_PATH:-data/ising/lattice_25_coupling_0.25.pt}"
OUTPUT_DIR="${ISING_RESULTS_ROOT:-results/ising}/learning/${SAMPLER}_seed${SEED}"
cd "$PROJECT_ROOT"
mkdir -p "$OUTPUT_DIR"

srun "$PYTHON_BIN" -m ising_pipeline.learn \
  --data "$DATA_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --sampler "$SAMPLER" \
  --sampling-steps "${ISING_LEARNING_STEPS:-40}" \
  --iterations "${ISING_LEARNING_ITERATIONS:-10000}" \
  --batch-size "${ISING_BATCH_SIZE:-256}" \
  --buffer-size "${ISING_BUFFER_SIZE:-256}" \
  --learning-rate "${ISING_LEARNING_RATE:-0.001}" \
  --l1 "${ISING_L1:-0.01}" \
  --record-every "${ISING_RECORD_EVERY:-100}" \
  --seed "$SEED" \
  --device cuda

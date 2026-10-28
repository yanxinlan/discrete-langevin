#!/bin/bash
#SBATCH --job-name=ising-data
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --output=/projects/prjs1237/project/tabpfgen/discrete-langevin/slurm/logs/%x-%j.out
#SBATCH --error=/projects/prjs1237/project/tabpfgen/discrete-langevin/slurm/logs/%x-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-python}"
cd "$PROJECT_ROOT"

srun "$PYTHON_BIN" -m ising_pipeline.generate_data \
  --output "${ISING_DATA_PATH:-data/ising/lattice_25_coupling_0.25.pt}" \
  --side "${ISING_SIDE:-25}" \
  --coupling "${ISING_COUPLING:-0.25}" \
  --bias "${ISING_BIAS:-0.0}" \
  --num-samples "${ISING_NUM_SAMPLES:-2000}" \
  --chains "${ISING_GENERATION_CHAINS:-256}" \
  --burn-in-sweeps "${ISING_BURN_IN_SWEEPS:-1000}" \
  --sweeps-between "${ISING_SWEEPS_BETWEEN:-20}" \
  --seed "${ISING_DATA_SEED:-123456}" \
  --device cuda

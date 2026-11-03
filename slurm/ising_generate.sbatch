#!/bin/bash
#SBATCH --job-name=ising-data
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=/projects/prjs1237/project/tabpfgen/discrete-langevin/slurm/logs/%x-%j.out
#SBATCH --error=/projects/prjs1237/project/tabpfgen/discrete-langevin/slurm/logs/%x-%j.err

set -euo pipefail

# The job requests node-level memory with --mem. Clear memory-mode variables
# that may be inherited from the shell that called sbatch; srun rejects mixtures.
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU

CONDA_ROOT="/home/xyan1/anaconda3"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate tabpfgen

# Self-contained Figure 5 data-generation configuration.
PROJECT_ROOT="/projects/prjs1237/project/tabpfgen/discrete-langevin"
PYTHON_BIN="$CONDA_PREFIX/bin/python"
DATA_PATH="$PROJECT_ROOT/data/ising/lattice_25_coupling_0.25_gibbs1000000.pt"
cd "$PROJECT_ROOT"

srun "$PYTHON_BIN" -m ising_pipeline.generate_data \
  --output "$DATA_PATH" \
  --side 25 \
  --coupling 0.25 \
  --bias 0.0 \
  --num-samples 2000 \
  --gibbs-steps 1000000 \
  --seed 123456 \
  --device cuda

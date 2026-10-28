#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
mkdir -p "$PROJECT_ROOT/slurm/logs" "$PROJECT_ROOT/data/ising" "$PROJECT_ROOT/results/ising"
export PROJECT_ROOT
cd "$PROJECT_ROOT"

DATA_JOB=$(sbatch --parsable slurm/ising_generate.sbatch)
SAMPLING_JOB=$(sbatch --parsable slurm/ising_sampling_array.sbatch)
LEARNING_JOB=$(sbatch --parsable --dependency="afterok:${DATA_JOB}" slurm/ising_learning_array.sbatch)

echo "data job: $DATA_JOB"
echo "sampling array: $SAMPLING_JOB"
echo "learning array (after data): $LEARNING_JOB"
echo "After completion: python -m ising_pipeline.summarize --results-root results/ising"

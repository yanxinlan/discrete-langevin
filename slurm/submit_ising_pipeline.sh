#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/projects/prjs1237/project/tabpfgen/discrete-langevin"
mkdir -p "$PROJECT_ROOT/slurm/logs" "$PROJECT_ROOT/data/ising" "$PROJECT_ROOT/results/ising"
export PROJECT_ROOT
cd "$PROJECT_ROOT"

DATA_JOB=$(sbatch --parsable slurm/ising_generate.sbatch)
LEARNING_JOB=$(sbatch --parsable --dependency="afterok:${DATA_JOB}" slurm/ising_learning_array.sbatch)

echo "data job: $DATA_JOB"
echo "learning array (after data): $LEARNING_JOB"
echo "After Figure 5 completion:"
echo "  python -m ising_pipeline.summarize --results-root results/ising/figure5_learning"
echo "  python -m ising_pipeline.plot_results --results-root results/ising/figure5_learning"

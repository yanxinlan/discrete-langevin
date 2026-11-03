#!/usr/bin/env bash
set -euo pipefail

# Submit from any directory; all paths and experimental settings live in the
# sbatch files. The log directory must exist before Slurm opens its output file.
PROJECT_ROOT="/projects/prjs1237/project/tabpfgen/discrete-langevin"
mkdir -p "$PROJECT_ROOT/slurm/logs"
cd "$PROJECT_ROOT"

SAMPLING_JOB="$(sbatch --parsable slurm/ising_sampling_array.sbatch)"
STEP_SIZE_JOB="$(sbatch --parsable slurm/ising_stepsize_acceptance_array.sbatch)"

echo "Figure 3 six-method, three-seed sampling array: $SAMPLING_JOB"
echo "Figure 2 step-size/acceptance array: $STEP_SIZE_JOB"
echo "Results root: $PROJECT_ROOT/results/ising"
echo "After Figure 3 completes:"
echo "  python -m ising_pipeline.summarize --results-root results/ising/figure3_sampling"
echo "  python -m ising_pipeline.plot_results --results-root results/ising/figure3_sampling"

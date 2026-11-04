#!/bin/bash
set -euo pipefail

PROJECT_ROOT="/projects/prjs1237/project/tabpfgen/discrete-langevin"
cd "$PROJECT_ROOT"
mkdir -p logs results/deep_ebm

for file in \
    datasets/MNIST_static/binarized_mnist_train.amat \
    datasets/MNIST_static/binarized_mnist_valid.amat \
    datasets/MNIST_static/binarized_mnist_test.amat \
    datasets/OMNIGLOT/chardata.mat; do
    if [[ ! -s "$file" ]]; then
        echo "Missing dataset file: $PROJECT_ROOT/$file" >&2
        echo "Run: bash scripts/download_deep_ebm_datasets.sh" >&2
        exit 1
    fi
done

TRAIN_JOB="$(sbatch --parsable slurm/deep_ebm_train_array.sbatch)"
# Array element i starts after the matching training element i succeeds, so a
# slower Omniglot run does not hold back finished Static MNIST evaluations.
AIS_JOB="$(sbatch --parsable --dependency="aftercorr:$TRAIN_JOB" slurm/deep_ebm_ais_array.sbatch)"

echo "Submitted deep EBM training array: $TRAIN_JOB"
echo "Submitted dependent final-AIS array: $AIS_JOB"
echo "Monitor: squeue -j $TRAIN_JOB,$AIS_JOB"

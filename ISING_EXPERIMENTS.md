# Ising sampling and learning experiments

This pipeline reproduces the two useful Ising experiments from *A Langevin-like
Sampler for Discrete Distributions*:

1. sampling a known 5 by 5 lattice target; and
2. recovering a known interaction matrix with persistent contrastive divergence.

## Data

No external dataset is involved. The learning data are synthetic binary spin
configurations generated from a periodic lattice Ising model. The saved `.pt`
file contains the samples, ground-truth coupling matrix, bias, and generation
metadata. The default learning problem uses a 25 by 25 lattice (625 variables),
matching the scale described for the learning experiment in the paper.

The model uses the original repository's matrix convention

```text
U(x) = s^T J s + b^T s,  s = 2x - 1,
```

so a symmetric `J` counts every undirected lattice edge twice.

## Environment

Only Python, NumPy, and a CUDA-enabled PyTorch installation are required. The
Slurm scripts assume the environment is already active on compute nodes. Set
`PYTHON_BIN` if the desired interpreter is not named `python`:

```bash
export PYTHON_BIN=/path/to/venv/bin/python
```

The scripts deliberately avoid `igraph` and TensorFlow Probability.

## Small local smoke test

```bash
python -m ising_pipeline.generate_data \
  --output /tmp/ising-small.pt --side 4 --num-samples 64 \
  --chains 16 --burn-in-sweeps 10 --sweeps-between 2 --device cpu

python -m ising_pipeline.benchmark_sampling \
  --output /tmp/dmala-small.npz --sampler dmala --side 3 \
  --steps 100 --chains 8 --record-every 10 --device cpu

python -m ising_pipeline.learn \
  --data /tmp/ising-small.pt --output-dir /tmp/ising-learn-small \
  --sampler dmala --iterations 10 --sampling-steps 2 \
  --batch-size 16 --buffer-size 16 --record-every 5 --device cpu
```

## Slurm: paper-focused Ising sampling

The paper-focused sampling jobs are self-contained and use the fixed cluster
root `/projects/prjs1237/project/tabpfgen/discrete-langevin`. Submit Figure 3's
supported four-method subset and Figure 2's DMALA step-size sweep together with:

```bash
bash /projects/prjs1237/project/tabpfgen/discrete-langevin/slurm/submit_ising_sampling_reproduction.sh
```

The `.sbatch` files contain all experimental values directly; they do not read
configuration from environment variables. Comments in the files distinguish
paper settings, unpublished choices, and known Python-level misalignments.

## Slurm: full sampling and learning pipeline

Create the log directory before submitting jobs. The convenience script does
this and submits data generation, sampling, and learning:

```bash
bash slurm/submit_ising_pipeline.sh
```

The sampling and learning scripts are arrays with 12 tasks: four samplers
(`gibbs`, `gwg`, `dula`, and `dmala`) times three seeds. Data generation is a
dependency of learning. Sampling is independent because it constructs its
known target directly and computes an exact reference mean by chunked
enumeration.

Common overrides can be exported before submission:

```bash
export ISING_DATA_PATH=/scratch/$USER/ising/lattice_25.pt
export ISING_RESULTS_ROOT=/scratch/$USER/ising/results
export ISING_LEARNING_ITERATIONS=10000
export ISING_SAMPLE_STEPS=50000
bash slurm/submit_ising_pipeline.sh
```

Cluster-specific account, partition, module, or container settings should be
added to the three `.sbatch` files. Every task requests one GPU. The 5 by 5
sampling reference enumerates all states in GPU-sized chunks, avoiding the
original script's multi-gigabyte all-at-once state tensor.

After the arrays finish, collect final metrics with:

```bash
python -m ising_pipeline.summarize --results-root results/ising
python -m ising_pipeline.plot_results --results-root results/ising
```

This writes `sampling_summary.csv`, `learning_summary.csv`, and sampling and
learning curves in both PDF and PNG formats.

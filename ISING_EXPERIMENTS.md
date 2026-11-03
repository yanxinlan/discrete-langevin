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

For Figure 5, data generation follows the released implementation: 2,000
parallel chains on a periodic 25 by 25 lattice with coupling 0.25 and zero
bias; every chain receives 1,000,000 cyclic single-coordinate Gibbs updates.
The resulting file is
`data/ising/lattice_25_coupling_0.25_gibbs1000000.pt`. No download is needed.

## Environment

Only Python, NumPy, Matplotlib, and PyTorch are required. The Slurm scripts
source `/home/xyan1/anaconda3` and activate the `tabpfgen` conda environment.
The Mac scripts accept `PYTHON_BIN` when the desired interpreter is not named
`python`:

```bash
export PYTHON_BIN=/path/to/venv/bin/python
```

The scripts deliberately avoid `igraph` and TensorFlow Probability.

## Small local smoke test

```bash
python -m ising_pipeline.generate_data \
  --output /tmp/ising-small.pt --side 4 --num-samples 64 \
  --gibbs-steps 20 --device cpu

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
After the Figure 3 array finishes, create the three-seed plot with:

```bash
python -m ising_pipeline.summarize --results-root results/ising/figure3_sampling
python -m ising_pipeline.plot_results --results-root results/ising/figure3_sampling
```

## Slurm: full sampling and learning pipeline

Create the log directory before submitting jobs. The convenience script does
this, submits Figure 5 data generation, and then starts learning only after the
dataset job succeeds:

```bash
bash slurm/submit_ising_pipeline.sh
```

The independent Figure 3 sampling wrapper described above submits an 18-task
array: six samplers (`Gibbs-1`, `Gibbs-4`, `GWG-1`, `GWG-4`, DULA, and DMALA)
times three seeds. The Figure 5 learning pipeline submits one data-generation
job followed by a 12-task learning array: four samplers times three seeds. All
paper settings and the cluster root are fixed inside the Slurm files; they do
not depend on exported experiment variables.

Cluster-specific account, partition, module, or container settings should be
added to the three `.sbatch` files. Every task requests one GPU. The 5 by 5
sampling reference enumerates all states in GPU-sized chunks, avoiding the
original script's multi-gigabyte all-at-once state tensor.

After the Figure 5 array finishes, collect final metrics with:

```bash
python -m ising_pipeline.summarize --results-root results/ising/figure5_learning
python -m ising_pipeline.plot_results --results-root results/ising/figure5_learning
```

This writes `sampling_summary.csv`, `learning_summary.csv`, and sampling and
learning curves in both PDF and PNG formats.

## MacBook runners

Sampling Figure 3 with six methods and three seeds, plus the Figure 2 left
step-size sweep:

```bash
PYTHON_BIN=/opt/homebrew/anaconda3/envs/tabpfgen/bin/python \
  bash scripts/run_ising_macbook.sh
```

Paper-aligned Figure 5(a,b) data generation and learning (one released-default
seed because the full Mac run is very large):

```bash
PYTHON_BIN=/opt/homebrew/anaconda3/envs/tabpfgen/bin/python \
  bash scripts/run_ising_learning_macbook.sh
```

The learning runner uses 100 PCD transitions per optimizer iteration, a buffer
and batch size of 256, 10,000 iterations, Adam at 0.001, and L1 coefficient
0.01. DULA uses alpha 0.1 and DMALA alpha 0.15. The zero bias is fixed and only
the dense coupling matrix is learned, matching `pcd.py`'s
`lattice_ising_2d` experiment.

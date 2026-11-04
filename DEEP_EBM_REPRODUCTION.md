# Deep EBM reproduction: Static MNIST and Omniglot

This workflow reproduces the DULA and DMALA rows of Table 1 for two datasets:

| Dataset | DULA step size | DMALA step size | Paper DULA | Paper DMALA |
|---|---:|---:|---:|---:|
| Static MNIST | 0.10 | 0.15 | -80.71 | -79.46 |
| Omniglot | 0.10 | 0.15 | -145.68 | -91.11 |

The log likelihoods in the last two columns are comparison targets, not hard-coded
outputs. The Gibbs and GWG entries in Table 1 were imported from earlier work, so
this workflow runs the two methods introduced in the discrete Langevin paper.

## Settings fixed to the paper

- ResNet EBM with eight residual blocks, two convolutions per block, 64 feature
  maps, and Swish activations (`resnet-64`).
- Persistent contrastive divergence with 40 sampler steps per optimizer update.
- Replay buffer of 10,000 samples.
- DULA step size 0.10 and DMALA step size 0.15.
- Adam with learning rate 0.0001 for exactly 50,000 updates.
- Validation every 5,000 updates with 10,000-step AIS; the best-validation
  checkpoint is retained.
- Final likelihood estimate uses 300,000-step AIS.

The paper does not state every implementation parameter. The following are fixed
to the released `ebm.sh`/`ais.sh` and parser defaults: batch size 100, 10,000-step
learning-rate warmup, EMA decay 0.999, mean-initialized replay buffer, zero buffer
reinitialization, base distribution enabled, seed 1234567, 100 validation-AIS
chains, and 500 final-AIS chains. The released final evaluation uses GWG inside
AIS for all trained models, which is preserved for comparability.

## Download data

From the repository root on Snellius:

```bash
bash scripts/download_deep_ebm_datasets.sh
```

This creates exactly the paths expected by `vamp_utils.py`:

- `datasets/MNIST_static/binarized_mnist_{train,valid,test}.amat`
- `datasets/OMNIGLOT/chardata.mat`

Static MNIST is downloaded from Hugo Larochelle's benchmark host. The script
validates the expected 50,000/10,000/10,000 rows and 784 columns before reporting
success.

Static MNIST is already binarized. Omniglot training examples are dynamically
binarized each epoch; validation and test examples use the original loader's
fixed NumPy seed 777.

The `tabpfgen` environment must provide `torch`, `numpy`, `scipy`, `tqdm`, and
`matplotlib`. Both Slurm jobs check these imports and CUDA before starting the
expensive run. The original code used `torchvision` only to write image grids;
the reproduction uses `ebm_image_utils.py` instead so no matching torchvision
binary is required.

## Submit

```bash
bash slurm/submit_deep_ebm_reproduction.sh
```

The training array contains four one-GPU jobs. Each final-AIS array element starts
after its corresponding training element succeeds. Results are written under
`results/deep_ebm/<dataset>/<sampler>/`. Because final AIS is expensive, the two
arrays each allow up to five days on an A100.

To summarize completed final-AIS logs:

```bash
python scripts/summarize_deep_ebm.py
```

The resulting `results/deep_ebm/summary.csv` reports the reproduced values and
their difference from Table 1.

#!/bin/bash
set -euo pipefail

# Run the paper-focused Ising experiments locally on an Apple-silicon GPU.
# The samplers use MPS. The exact 2^25-state reference is evaluated once on the
# CPU in float64 (MPS has no float64 support), cached, and reused by every run.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
RESULTS_ROOT="$PROJECT_ROOT/results/ising/macbook"
REFERENCE_CACHE="$RESULTS_ROOT/exact_reference_5x5_a0.1_b0.2.npz"

cd "$PROJECT_ROOT"
mkdir -p "$RESULTS_ROOT/sampling" "$RESULTS_ROOT/step_size_acceptance"

"$PYTHON_BIN" - <<'PY'
import platform
import sys
import torch

print(f"Python: {sys.executable}")
print(f"Platform: {platform.platform()}")
print(f"PyTorch: {torch.__version__}")
print(f"MPS built: {torch.backends.mps.is_built()}")
print(f"MPS available: {torch.backends.mps.is_available()}")
if not torch.backends.mps.is_available():
    raise SystemExit(
        "Apple MPS is unavailable in this Python environment. Run with an "
        "Apple-silicon PyTorch environment, for example: "
        "PYTHON_BIN=/path/to/env/bin/python scripts/run_ising_macbook.sh"
    )
PY

COMMON_ARGS=(
  --side 5
  --coupling 0.1
  --bias 0.2
  --steps 50000
  --chains 2
  --reference exact
  --reference-device cpu
  --reference-cache "$REFERENCE_CACHE"
  --exact-chunk-size 1000000
  --device mps
)

# Figure 3 focused reproduction with three independent seeds. Runtime is
# Mac-specific and should not be compared numerically with the paper's GPU.
SAMPLERS=(gibbs-1 gibbs-4 gwg-1 gwg-4 dula dmala)
STEP_SIZES=(1.0 1.0 1.0 1.0 0.2 0.4)
SEEDS=(1234567 1234568 1234569)
for seed in "${SEEDS[@]}"; do
  for index in "${!SAMPLERS[@]}"; do
    sampler="${SAMPLERS[$index]}"
    step_size="${STEP_SIZES[$index]}"
    echo "Running Figure 3: $sampler, seed $seed (step size $step_size)"
    "$PYTHON_BIN" -m ising_pipeline.benchmark_sampling \
      --output "$RESULTS_ROOT/sampling/${sampler}_seed${seed}.npz" \
      --sampler "$sampler" \
      --step-size "$step_size" \
      --record-every 1000 \
      --seed "$seed" \
      "${COMMON_ARGS[@]}"
  done
done

"$PYTHON_BIN" -m ising_pipeline.summarize --results-root "$RESULTS_ROOT"
"$PYTHON_BIN" -m ising_pipeline.plot_results --results-root "$RESULTS_ROOT"

# Figure 2 left: DMALA step-size and acceptance sweep. The paper does not state
# its chain count/run length; these use the released main-script defaults.
for step_size in 0.2 0.3 0.4 0.5 0.6; do
  echo "Running Figure 2 left: DMALA step size $step_size"
  "$PYTHON_BIN" -m ising_pipeline.benchmark_sampling \
    --output "$RESULTS_ROOT/step_size_acceptance/dmala_alpha${step_size}_seed1234567.npz" \
    --sampler dmala \
    --step-size "$step_size" \
    --record-every 1 \
    --seed 1234567 \
    "${COMMON_ARGS[@]}"
done

"$PYTHON_BIN" -m ising_pipeline.plot_stepsize \
  --input-dir "$RESULTS_ROOT/step_size_acceptance" \
  --output "$RESULTS_ROOT/figure2_left"

echo "MacBook experiments completed: $RESULTS_ROOT"

#!/bin/bash
set -euo pipefail

# Paper-aligned Figure 5(a,b) Ising learning on Apple silicon. This is a large
# experiment: exact data generation alone performs 1,000,000 Gibbs transitions
# on 2,000 parallel 625-dimensional chains.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_PATH="$PROJECT_ROOT/data/ising/lattice_25_coupling_0.25_gibbs1000000.pt"
RESULTS_ROOT="$PROJECT_ROOT/results/ising/figure5_learning_macbook"

cd "$PROJECT_ROOT"
mkdir -p "$(dirname "$DATA_PATH")" "$RESULTS_ROOT/learning"

"$PYTHON_BIN" - <<'PY'
import sys
import torch

print(f"Python: {sys.executable}")
print(f"PyTorch: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")
if not torch.backends.mps.is_available():
    raise SystemExit("MPS is unavailable; run with the Mac tabpfgen Python environment")
PY

if [[ ! -f "$DATA_PATH" ]]; then
  "$PYTHON_BIN" -m ising_pipeline.generate_data \
    --output "$DATA_PATH" \
    --side 25 \
    --coupling 0.25 \
    --bias 0.0 \
    --num-samples 2000 \
    --gibbs-steps 1000000 \
    --seed 123456 \
    --device mps
else
  echo "Reusing paper-aligned dataset: $DATA_PATH"
fi

# The paper does not state repetitions for Figure 5. The Mac runner uses the
# released default seed once to keep this already expensive run manageable.
for sampler in gibbs-1 gwg-1 dula dmala; do
  echo "Learning with $sampler"
  "$PYTHON_BIN" -m ising_pipeline.learn \
    --data "$DATA_PATH" \
    --output-dir "$RESULTS_ROOT/learning/${sampler}_seed123456" \
    --sampler "$sampler" \
    --sampling-steps 100 \
    --iterations 10000 \
    --batch-size 256 \
    --buffer-size 256 \
    --learning-rate 0.001 \
    --l1 0.01 \
    --record-every 1000 \
    --seed 123456 \
    --device mps
done

"$PYTHON_BIN" -m ising_pipeline.summarize --results-root "$RESULTS_ROOT"
"$PYTHON_BIN" -m ising_pipeline.plot_results --results-root "$RESULTS_ROOT"
echo "MacBook Figure 5 experiments completed: $RESULTS_ROOT"

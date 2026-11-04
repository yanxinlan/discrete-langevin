#!/bin/bash
set -euo pipefail

# Datasets used for the Static MNIST and Omniglot rows of Table 1.  The local
# directory names match vamp_utils.py exactly.
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MNIST_DIR="$PROJECT_ROOT/datasets/MNIST_static"
OMNIGLOT_DIR="$PROJECT_ROOT/datasets/OMNIGLOT"

mkdir -p "$MNIST_DIR" "$OMNIGLOT_DIR"

download() {
    local output="$1"
    shift
    if [[ -s "$output" ]]; then
        echo "Already present: $output"
        return
    fi
    rm -f "$output.part"
    local url
    for url in "$@"; do
        echo "Downloading: $output"
        echo "Source: $url"
        if curl --fail --location --retry 3 --output "$output.part" "$url"; then
            mv "$output.part" "$output"
            return
        fi
        rm -f "$output.part"
        echo "Source failed; trying the next mirror." >&2
    done
    echo "All download sources failed for: $output" >&2
    exit 1
}

# Static MNIST is Hugo Larochelle's official fixed-binarization benchmark.
MNIST_OFFICIAL="https://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist"
download "$MNIST_DIR/binarized_mnist_train.amat" \
    "$MNIST_OFFICIAL/binarized_mnist_train.amat"
download "$MNIST_DIR/binarized_mnist_valid.amat" \
    "$MNIST_OFFICIAL/binarized_mnist_valid.amat"
download "$MNIST_DIR/binarized_mnist_test.amat" \
    "$MNIST_OFFICIAL/binarized_mnist_test.amat"

# This is the 28x28 Omniglot file used by the original VampPrior/GWG loader.
download "$OMNIGLOT_DIR/chardata.mat" \
    "https://raw.githubusercontent.com/yburda/iwae/master/datasets/OMNIGLOT/chardata.mat"

validate_amat() {
    local file="$1"
    local expected_rows="$2"
    local rows columns
    rows="$(wc -l < "$file" | tr -d ' ')"
    columns="$(awk 'NR == 1 { print NF; exit }' "$file")"
    if [[ "$rows" != "$expected_rows" || "$columns" != "784" ]]; then
        echo "Invalid Static MNIST file: $file ($rows rows, $columns columns)" >&2
        exit 1
    fi
}

validate_amat "$MNIST_DIR/binarized_mnist_train.amat" 50000
validate_amat "$MNIST_DIR/binarized_mnist_valid.amat" 10000
validate_amat "$MNIST_DIR/binarized_mnist_test.amat" 10000

if [[ "$(wc -c < "$OMNIGLOT_DIR/chardata.mat" | tr -d ' ')" -lt 1000000 ]]; then
    echo "Invalid or incomplete Omniglot file: $OMNIGLOT_DIR/chardata.mat" >&2
    exit 1
fi

echo "Dataset download complete."
du -h "$MNIST_DIR"/* "$OMNIGLOT_DIR/chardata.mat"

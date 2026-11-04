#!/bin/bash
set -euo pipefail

# Datasets used for the Static MNIST and Omniglot rows of Table 1.  The local
# directory names match vamp_utils.py exactly.
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MNIST_DIR="$PROJECT_ROOT/datasets/MNIST_static"
OMNIGLOT_DIR="$PROJECT_ROOT/datasets/OMNIGLOT"

mkdir -p "$MNIST_DIR" "$OMNIGLOT_DIR"

download() {
    local url="$1"
    local output="$2"
    if [[ -s "$output" ]]; then
        echo "Already present: $output"
        return
    fi
    echo "Downloading: $output"
    curl --fail --location --retry 3 --output "$output.part" "$url"
    mv "$output.part" "$output"
}

# These are the same benchmark files used by the VampPrior data loader copied
# into this repository by the original authors.
BASE_URL="https://raw.githubusercontent.com/yburda/iwae/master/datasets"
download "$BASE_URL/BinaryMNIST/binarized_mnist_train.amat" "$MNIST_DIR/binarized_mnist_train.amat"
download "$BASE_URL/BinaryMNIST/binarized_mnist_valid.amat" "$MNIST_DIR/binarized_mnist_valid.amat"
download "$BASE_URL/BinaryMNIST/binarized_mnist_test.amat" "$MNIST_DIR/binarized_mnist_test.amat"
download "$BASE_URL/OMNIGLOT/chardata.mat" "$OMNIGLOT_DIR/chardata.mat"

echo "Dataset download complete."
du -h "$MNIST_DIR"/* "$OMNIGLOT_DIR/chardata.mat"

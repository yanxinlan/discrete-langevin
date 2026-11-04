"""Small image-grid writer for the deep EBM scripts.

The original release used torchvision only for ``save_image``.  Keeping this
local avoids coupling the experiments to a matching torchvision binary build.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch


def save_image_grid(
    values: torch.Tensor,
    path: str | Path,
    input_size: list[int] | tuple[int, int, int],
    *,
    nrow: int | None = None,
    normalize: bool = True,
) -> None:
    channels, height, width = input_size
    images = values.detach().cpu().float().reshape(-1, channels, height, width)
    if normalize:
        minimum = images.min()
        maximum = images.max()
        if maximum > minimum:
            images = (images - minimum) / (maximum - minimum)
    images = images.clamp(0.0, 1.0)

    columns = nrow or max(1, int(math.sqrt(images.shape[0])))
    rows = math.ceil(images.shape[0] / columns)
    grid = torch.zeros(channels, rows * height, columns * width)
    for index, image in enumerate(images):
        row, column = divmod(index, columns)
        grid[:, row * height : (row + 1) * height, column * width : (column + 1) * width] = image

    output = grid.permute(1, 2, 0).numpy()
    if channels == 1:
        plt.imsave(path, output[:, :, 0], cmap="gray", vmin=0.0, vmax=1.0)
    elif channels == 3:
        plt.imsave(path, output)
    else:
        raise ValueError(f"Expected 1 or 3 image channels, got {channels}")

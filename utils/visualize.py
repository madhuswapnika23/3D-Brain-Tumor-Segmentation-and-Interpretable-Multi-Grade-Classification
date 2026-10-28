"""Plot BraTS modalities and segmentation masks for loader sanity checks."""

from pathlib import Path
from typing import Mapping

import numpy as np
from torch import Tensor


_MODALITIES = ("T1", "T1ce", "T2", "FLAIR")
_SLICE_FRACTIONS = (0.3, 0.4, 0.5, 0.6)


def visualize_sample(
    sample: Mapping[str, Tensor],
    output_path: str | Path = Path("outputs/sample_check.png"),
) -> Path:
    """Save a 4-by-4 axial montage for one dataset sample.

    ``image`` must have shape ``[4, X, Y, Z]`` (or a batch of one,
    ``[1, 4, X, Y, Z]``). ``label`` must have shape ``[3, X, Y, Z]`` (or
    ``[1, 3, X, Y, Z]``), with channels ordered ET, TC, WT.
    """
    if "image" not in sample or "label" not in sample:
        raise KeyError("sample must contain 'image' and 'label' tensors")

    image = sample["image"].detach().cpu()
    label = sample["label"].detach().cpu()
    if image.ndim == 5:
        if image.shape[0] != 1:
            raise ValueError("visualize_sample accepts only a batch of one")
        image = image[0]
    if label.ndim == 5:
        if label.shape[0] != 1:
            raise ValueError("visualize_sample accepts only a batch of one")
        label = label[0]

    if image.ndim != 4 or image.shape[0] != 4:
        raise ValueError(f"image must have shape [4, D, H, W], got {tuple(image.shape)}")
    if label.ndim != 4 or label.shape[0] != 3:
        raise ValueError(f"label must have shape [3, D, H, W], got {tuple(label.shape)}")
    if image.shape[1:] != label.shape[1:]:
        raise ValueError("image and label spatial dimensions must match")
    if any(size == 0 for size in image.shape[1:]):
        raise ValueError("image spatial dimensions must be non-empty")

    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    image_array = image.numpy()
    label_array = label.numpy() > 0.5
    depth = image_array.shape[3]
    slice_indices = [int((depth - 1) * fraction) for fraction in _SLICE_FRACTIONS]

    # The masks are nested (ET within TC within WT), so assign exclusive
    # colors from inner to outer regions to keep each class visible.
    et = label_array[0]
    tc = label_array[1] & ~et
    wt = label_array[2] & ~label_array[1]
    overlay = np.zeros((*label_array.shape[1:], 4), dtype=np.float32)
    overlay[wt] = (0.0, 1.0, 0.0, 0.3)
    overlay[tc] = (1.0, 1.0, 0.0, 0.3)
    overlay[et] = (1.0, 0.0, 0.0, 0.3)

    figure, axes = plt.subplots(4, 4, figsize=(14, 14), squeeze=False)
    try:
        for row, modality in enumerate(_MODALITIES):
            for column, slice_index in enumerate(slice_indices):
                axis = axes[row, column]
                axis.imshow(image_array[row, :, :, slice_index].T, cmap="gray")
                axis.imshow(overlay[:, :, slice_index, :].transpose(1, 0, 2))
                axis.set_title(f"{modality} - {slice_index}")
                axis.axis("off")

        figure.legend(
            handles=(
                Patch(facecolor="green", alpha=0.3, label="WT"),
                Patch(facecolor="yellow", alpha=0.3, label="TC"),
                Patch(facecolor="red", alpha=0.3, label="ET"),
            ),
            loc="lower center",
            ncol=3,
        )
        figure.tight_layout(rect=(0, 0.04, 1, 1))
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=150, bbox_inches="tight")
    finally:
        plt.close(figure)

    return destination

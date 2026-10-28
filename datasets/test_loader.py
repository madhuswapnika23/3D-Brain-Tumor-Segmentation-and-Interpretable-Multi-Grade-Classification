"""Load BraTS samples with ``python -m datasets.test_loader``."""

import argparse
import random
from pathlib import Path

from .brats_dataset import BraTSDataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load BraTS samples and print patch dimensions/voxel counts."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--index", type=int, default=0,
                        help="Subject index used when --num-samples is 1 (default).")
    parser.add_argument("--num-samples", type=int, default=1,
                        help="Number of random subjects to draw (default: 1).")
    args = parser.parse_args()

    dataset = BraTSDataset(args.data_dir)

    # When drawing multiple samples, pick random indices; for a single sample
    # respect --index so existing behaviour is unchanged.
    if args.num_samples == 1:
        indices = [args.index % len(dataset)]
    else:
        indices = [random.randrange(len(dataset)) for _ in range(args.num_samples)]

    et_counts: list[int] = []

    for i, idx in enumerate(indices):
        sample = dataset[idx]
        et = int(sample["label"][0].sum().item())
        tc = int(sample["label"][1].sum().item())
        wt = int(sample["label"][2].sum().item())
        et_counts.append(et)

        print(f"--- Sample {i + 1}/{args.num_samples}  (subject index {idx}) ---")
        print(f"  Image shape : {tuple(sample['image'].shape)}")
        print(f"  Label shape : {tuple(sample['label'].shape)}")
        print(f"  ET voxels   : {et}")
        print(f"  TC voxels   : {tc}")
        print(f"  WT voxels   : {wt}")

    if args.num_samples > 1:
        print("\n--- Summary across all samples ---")
        print(f"  Min ET : {min(et_counts)}")
        print(f"  Max ET : {max(et_counts)}")
        print(f"  Mean ET: {sum(et_counts) / len(et_counts):.1f}")


if __name__ == "__main__":
    main()

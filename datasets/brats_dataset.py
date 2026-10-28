"""PyTorch dataset for BraTS 2023 GLI NIfTI volumes."""

from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


# BraTS 2023 GLI modality suffixes (appended to the folder/subject name)
_MODALITY_SUFFIXES = ("t1n", "t1c", "t2w", "t2f")


class BraTSDataset(Dataset[dict[str, Tensor]]):
    """Load BraTS 2023 GLI subjects and sample augmented 3D patches.

    Expected on-disk layout::

        <data_dir>/
            BraTS-GLI-00000-000/
                BraTS-GLI-00000-000-t1n.nii.gz
                BraTS-GLI-00000-000-t1c.nii.gz
                BraTS-GLI-00000-000-t2w.nii.gz
                BraTS-GLI-00000-000-t2f.nii.gz
                BraTS-GLI-00000-000-seg.nii.gz
            BraTS-GLI-00001-000/
                ...

    Filenames are constructed as ``{folder_name}-{suffix}.nii.gz`` so that no
    subject ID is hard-coded in this class.  Z-score normalisation is applied
    per modality over finite, non-zero voxels.

    By default, half of patch draws are centred on a randomly selected
    foreground voxel, ensuring those draws contain tumour for labelled cases.
    """

    def __init__(
        self,
        data_dir: str | Path,
        patch_size: int | tuple[int, int, int] = 128,
        foreground_sampling_prob: float = 0.5,
        augment: bool = True,
        dataset_version: str = "2023",
    ) -> None:
        if dataset_version not in ("2021", "2023"):
            raise ValueError("dataset_version must be '2021' or '2023'")
        # BraTS 2023 changed ET label from 4 (2021) to 3 (2023).
        self.et_label: int = 3 if dataset_version == "2023" else 4
        self.data_dir = Path(data_dir)
        if isinstance(patch_size, int):
            patch_size = (patch_size,) * 3
        self.patch_size = tuple(patch_size)
        if len(self.patch_size) != 3 or any(size <= 0 for size in self.patch_size):
            raise ValueError("patch_size must contain three positive dimensions")
        if not 0.0 <= foreground_sampling_prob <= 1.0:
            raise ValueError("foreground_sampling_prob must be between 0 and 1")

        self.foreground_sampling_prob = foreground_sampling_prob
        self.augment = augment

        # Discover subject folders: any immediate child directory whose name
        # starts with "BraTS-GLI-" and contains a segmentation file named
        # after itself.
        self.subject_dirs: list[Path] = sorted(
            d
            for d in self.data_dir.iterdir()
            if d.is_dir() and d.name.startswith("BraTS-GLI-")
        )

        if not self.subject_dirs:
            raise FileNotFoundError(
                f"No BraTS-GLI subject folders found under {self.data_dir}"
            )

        # Validate that every expected file exists for each subject.
        for subject_dir in self.subject_dirs:
            missing = [
                fname
                for fname in self._subject_filenames(subject_dir)
                if not (subject_dir / fname).is_file()
            ]
            if missing:
                raise FileNotFoundError(
                    f"Missing files for subject {subject_dir.name}: "
                    + ", ".join(missing)
                )

    # ------------------------------------------------------------------
    # Filename helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _subject_filenames(subject_dir: Path) -> tuple[str, ...]:
        """Return all expected filenames (relative to *subject_dir*) for one subject."""
        name = subject_dir.name
        return tuple(
            f"{name}-{suffix}.nii.gz"
            for suffix in (*_MODALITY_SUFFIXES, "seg")
        )

    @staticmethod
    def _modality_path(subject_dir: Path, suffix: str) -> Path:
        return subject_dir / f"{subject_dir.name}-{suffix}.nii.gz"

    @staticmethod
    def _seg_path(subject_dir: Path) -> Path:
        return subject_dir / f"{subject_dir.name}-seg.nii.gz"

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.subject_dirs)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        subject_dir = self.subject_dirs[index]

        image = np.stack(
            [
                self._load_and_normalize(self._modality_path(subject_dir, suffix))
                for suffix in _MODALITY_SUFFIXES
            ],
            axis=0,
        )
        segmentation = np.asarray(
            nib.load(str(self._seg_path(subject_dir))).get_fdata(),
            dtype=np.int16,
        )
        et = self.et_label
        label = np.stack(
            (
                segmentation == et,
                (segmentation == 1) | (segmentation == et),
                (segmentation == 1) | (segmentation == 2) | (segmentation == et),
            ),
            axis=0,
        ).astype(np.float32)

        if image.shape[1:] != segmentation.shape:
            raise ValueError(
                f"Modality and segmentation shapes differ for {subject_dir}"
            )

        image, label = self._pad_to_patch_size(image, label)
        image, label = self._extract_patch(image, label)
        if self.augment:
            image, label = self._augment(image, label)

        return {
            "image": torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32)),
            "label": torch.from_numpy(np.ascontiguousarray(label, dtype=np.float32)),
        }

    # ------------------------------------------------------------------
    # Internal utilities (unchanged logic)
    # ------------------------------------------------------------------

    @staticmethod
    def _load_and_normalize(path: Path) -> np.ndarray:
        volume = np.asarray(nib.load(str(path)).get_fdata(), dtype=np.float32)
        finite = np.isfinite(volume)
        nonzero = finite & (volume != 0)
        normalized = np.zeros_like(volume, dtype=np.float32)
        if np.any(nonzero):
            values = volume[nonzero]
            mean = values.mean(dtype=np.float64)
            std = values.std(dtype=np.float64)
            if std > 0:
                normalized[nonzero] = (values - mean) / std
        return normalized

    def _pad_to_patch_size(
        self, image: np.ndarray, label: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        spatial_shape = image.shape[1:]
        padding = [
            (0, max(0, patch - size))
            for size, patch in zip(spatial_shape, self.patch_size)
        ]
        if any(after for _, after in padding):
            image = np.pad(image, [(0, 0), *padding], mode="constant")
            label = np.pad(label, [(0, 0), *padding], mode="constant")
        return image, label

    def _extract_patch(
        self, image: np.ndarray, label: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        spatial_shape = image.shape[1:]
        max_starts = tuple(
            size - patch for size, patch in zip(spatial_shape, self.patch_size)
        )
        starts: list[int] = []

        use_foreground = (
            np.random.random() < self.foreground_sampling_prob
            and np.any(label[2] > 0)
        )
        if use_foreground:
            foreground = np.argwhere(label[2] > 0)
            center = foreground[np.random.randint(len(foreground))]
            for coordinate, max_start, patch in zip(
                center, max_starts, self.patch_size
            ):
                min_start = max(0, int(coordinate) - patch + 1)
                max_foreground_start = min(int(coordinate), max_start)
                starts.append(
                    np.random.randint(min_start, max_foreground_start + 1)
                )
        else:
            starts = [
                np.random.randint(max_start + 1) for max_start in max_starts
            ]

        slices = tuple(
            slice(start, start + patch) for start, patch in zip(starts, self.patch_size)
        )
        return image[(slice(None), *slices)], label[(slice(None), *slices)]

    @staticmethod
    def _augment(
        image: np.ndarray, label: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        for axis in range(1, 4):
            if np.random.random() < 0.5:
                image = np.flip(image, axis=axis)
                label = np.flip(label, axis=axis)

        rotation_axes = ((1, 2), (1, 3), (2, 3))[np.random.randint(3)]
        quarter_turns = np.random.randint(4)
        image = np.rot90(image, quarter_turns, axes=rotation_axes)
        label = np.rot90(label, quarter_turns, axes=rotation_axes)

        scales = np.random.uniform(0.9, 1.1, size=(image.shape[0], 1, 1, 1))
        image = image * scales
        image = image + np.random.normal(0.0, 0.01, size=image.shape).astype(
            np.float32
        )
        return image.astype(np.float32, copy=False), label.astype(
            np.float32, copy=False
        )

"""
loader.py
─────────
Load 3D microscopy data stored in OME-Zarr or plain Zarr format.
Returns PyTorch Datasets ready for training/inference.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _open_zarr(path: str | Path) -> zarr.Array:
    """Open a Zarr array (or group root) from *path*."""
    store = zarr.open(str(path), mode="r")
    if isinstance(store, zarr.Group):
        # OME-Zarr: pick the first array (highest resolution level)
        first_key = next(iter(store))
        store = store[first_key]
    return store  # type: ignore[return-value]


# ─────────────────────────────────────────────────────────────────────────────
# Datasets
# ─────────────────────────────────────────────────────────────────────────────

class ZarrPatchDataset(Dataset):
    """Yields random 3-D sub-volumes (patches) from a Zarr array.

    Parameters
    ----------
    zarr_path:
        Path to the `.zarr` file/directory.
    label_path:
        Optional path to a corresponding label `.zarr` (same shape).
    patch_size:
        ``(Z, Y, X)`` size of each patch.
    n_patches:
        How many patches to sample per epoch.
    transform:
        Optional callable applied to ``(image, label)`` tuples.
    seed:
        Random seed for reproducible patch sampling.
    """

    def __init__(
        self,
        zarr_path: str | Path,
        label_path: Optional[str | Path] = None,
        patch_size: Tuple[int, int, int] = (64, 128, 128),
        n_patches: int = 500,
        transform: Optional[Callable] = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.image = _open_zarr(zarr_path)
        self.label = _open_zarr(label_path) if label_path else None
        self.patch_size = patch_size
        self.n_patches = n_patches
        self.transform = transform
        self.rng = np.random.default_rng(seed)

        logger.info(
            "ZarrPatchDataset | image shape: %s | patches: %d",
            self.image.shape,
            n_patches,
        )

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self.n_patches

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> dict:
        pz, py, px = self.patch_size
        iz, iy, ix = self.image.shape[-3:]

        # Random top-left corner
        z0 = self.rng.integers(0, max(1, iz - pz))
        y0 = self.rng.integers(0, max(1, iy - py))
        x0 = self.rng.integers(0, max(1, ix - px))

        img_patch = np.array(
            self.image[..., z0 : z0 + pz, y0 : y0 + py, x0 : x0 + px],
            dtype=np.float32,
        )
        sample: dict = {"image": torch.from_numpy(img_patch).unsqueeze(0)}

        if self.label is not None:
            lbl_patch = np.array(
                self.label[..., z0 : z0 + pz, y0 : y0 + py, x0 : x0 + px],
                dtype=np.int64,
            )
            sample["label"] = torch.from_numpy(lbl_patch)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


class ZarrInferenceDataset(Dataset):
    """Sliding-window dataset for full-volume inference.

    Splits a Zarr volume into overlapping patches that can be reassembled
    after model prediction.
    """

    def __init__(
        self,
        zarr_path: str | Path,
        patch_size: Tuple[int, int, int] = (64, 128, 128),
        overlap: Tuple[int, int, int] = (8, 16, 16),
    ) -> None:
        super().__init__()
        self.image = _open_zarr(zarr_path)
        self.patch_size = patch_size
        self.overlap = overlap
        self.patches: List[Tuple[slice, slice, slice]] = []
        self._compute_patches()

    # ------------------------------------------------------------------
    def _compute_patches(self) -> None:
        """Pre-compute all patch slices for the full volume."""
        iz, iy, ix = self.image.shape[-3:]
        pz, py, px = self.patch_size
        oz, oy, ox = self.overlap

        stride_z = pz - oz
        stride_y = py - oy
        stride_x = px - ox

        for z in range(0, iz, stride_z):
            for y in range(0, iy, stride_y):
                for x in range(0, ix, stride_x):
                    sz = slice(min(z, iz - pz), min(z, iz - pz) + pz)
                    sy = slice(min(y, iy - py), min(y, iy - py) + py)
                    sx = slice(min(x, ix - px), min(x, ix - px) + px)
                    self.patches.append((sz, sy, sx))

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.patches)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> dict:
        sz, sy, sx = self.patches[idx]
        patch = np.array(
            self.image[..., sz, sy, sx], dtype=np.float32
        )
        return {
            "image": torch.from_numpy(patch).unsqueeze(0),
            "slices": (sz, sy, sx),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_dataloaders(
    train_zarr: str | Path,
    val_zarr: str | Path,
    train_labels: Optional[str | Path] = None,
    val_labels: Optional[str | Path] = None,
    patch_size: Tuple[int, int, int] = (64, 128, 128),
    n_train_patches: int = 1000,
    n_val_patches: int = 200,
    batch_size: int = 2,
    num_workers: int = 4,
    transform_train: Optional[Callable] = None,
    transform_val: Optional[Callable] = None,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Return (train_loader, val_loader) from Zarr paths."""
    train_ds = ZarrPatchDataset(
        train_zarr, train_labels, patch_size, n_train_patches, transform_train
    )
    val_ds = ZarrPatchDataset(
        val_zarr, val_labels, patch_size, n_val_patches, transform_val
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader

"""
data/utils.py
─────────────
Utility helpers for data I/O and coordinate handling.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_coords_csv(
    coords: np.ndarray,
    timepoints: Sequence[int],
    output_path: str | Path,
    col_names: Tuple[str, ...] = ("t", "z", "y", "x"),
) -> None:
    """Save cell coordinate array to a CSV file.

    Parameters
    ----------
    coords:
        Array of shape ``(N, 3)`` with (z, y, x) coordinates.
    timepoints:
        Timepoint index for each row in *coords*.
    output_path:
        Destination file path.
    """
    df = pd.DataFrame(coords, columns=list(col_names[1:]))
    df.insert(0, col_names[0], list(timepoints))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def load_coords_csv(path: str | Path) -> pd.DataFrame:
    """Load a CSV of cell coordinates back to a DataFrame."""
    return pd.read_csv(path)


# ─────────────────────────────────────────────────────────────────────────────
# Patch / volume utilities
# ─────────────────────────────────────────────────────────────────────────────

def pad_to_shape(
    volume: np.ndarray,
    target: Tuple[int, ...],
    pad_value: float = 0.0,
) -> np.ndarray:
    """Zero-pad *volume* so its last N dims match *target*."""
    pads = []
    for t, s in zip(target, volume.shape[-len(target) :]):
        diff = max(0, t - s)
        pads.append((diff // 2, diff - diff // 2))
    # prepend (0,0) for any leading dimensions
    leading = [(0, 0)] * (volume.ndim - len(target))
    return np.pad(volume, leading + pads, constant_values=pad_value)


def crop_center(
    volume: np.ndarray,
    crop_size: Tuple[int, ...],
) -> np.ndarray:
    """Crop the center region of *volume* to *crop_size*."""
    starts = [
        (s - c) // 2
        for s, c in zip(volume.shape[-len(crop_size) :], crop_size)
    ]
    slices = tuple(
        slice(st, st + c) for st, c in zip(starts, crop_size)
    )
    # preserve leading dims
    leading = (slice(None),) * (volume.ndim - len(crop_size))
    return volume[leading + slices]


# ─────────────────────────────────────────────────────────────────────────────
# Label utilities
# ─────────────────────────────────────────────────────────────────────────────

def instance_to_binary(label: np.ndarray) -> np.ndarray:
    """Convert an instance-label volume to a binary mask."""
    return (label > 0).astype(np.uint8)


def count_cells_per_frame(label_sequence: List[np.ndarray]) -> List[int]:
    """Return the number of unique cell instances per time frame."""
    return [len(np.unique(lbl)) - 1 for lbl in label_sequence]  # exclude bg

"""
preprocess.py
─────────────
Normalization, denoising, and augmentation utilities for 3D microscopy data.
"""

from __future__ import annotations

import random
from typing import Dict, Optional, Tuple

import numpy as np
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Normalization
# ─────────────────────────────────────────────────────────────────────────────

def normalize_zscore(
    volume: np.ndarray,
    mean: Optional[float] = None,
    std: Optional[float] = None,
    eps: float = 1e-8,
) -> np.ndarray:
    """Z-score normalize a 3-D volume.

    If *mean* / *std* are ``None`` they are computed from *volume* itself
    (per-sample normalization).
    """
    mean = mean if mean is not None else volume.mean()
    std = std if std is not None else volume.std()
    return (volume - mean) / (std + eps)


def normalize_minmax(
    volume: np.ndarray,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    eps: float = 1e-8,
) -> np.ndarray:
    """Min-max normalize a volume to [0, 1]."""
    vmin = vmin if vmin is not None else volume.min()
    vmax = vmax if vmax is not None else volume.max()
    return (volume - vmin) / (vmax - vmin + eps)


def clip_percentile(
    volume: np.ndarray,
    p_low: float = 0.5,
    p_high: float = 99.5,
) -> np.ndarray:
    """Clip intensity to [p_low, p_high] percentiles before normalization."""
    lo, hi = np.percentile(volume, [p_low, p_high])
    return np.clip(volume, lo, hi)


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation transforms (dict-based, compatible with ZarrPatchDataset)
# ─────────────────────────────────────────────────────────────────────────────

class RandomFlip3D:
    """Randomly flip along any spatial axis (Z, Y, X)."""

    def __init__(self, prob: float = 0.5) -> None:
        self.prob = prob

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        for axis in range(-3, 0):
            if random.random() < self.prob:
                sample["image"] = torch.flip(sample["image"], [axis])
                if "label" in sample:
                    sample["label"] = torch.flip(sample["label"], [axis])
        return sample


class RandomIntensityShift:
    """Additive + multiplicative intensity jitter."""

    def __init__(self, shift: float = 0.1, scale: float = 0.1) -> None:
        self.shift = shift
        self.scale = scale

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        img = sample["image"]
        delta = random.uniform(-self.shift, self.shift)
        factor = random.uniform(1 - self.scale, 1 + self.scale)
        sample["image"] = img * factor + delta
        return sample


class RandomGaussianNoise:
    """Add Gaussian noise to the image."""

    def __init__(self, std: float = 0.02) -> None:
        self.std = std

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        noise = torch.randn_like(sample["image"]) * self.std
        sample["image"] = sample["image"] + noise
        return sample


class Compose:
    """Chain multiple transforms sequentially."""

    def __init__(self, transforms: list) -> None:
        self.transforms = transforms

    def __call__(self, sample: dict) -> dict:
        for t in self.transforms:
            sample = t(sample)
        return sample


# ─────────────────────────────────────────────────────────────────────────────
# Factory helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_train_transforms(
    flip_prob: float = 0.5,
    intensity_shift: float = 0.1,
    intensity_scale: float = 0.1,
    noise_std: float = 0.02,
) -> Compose:
    """Return a standard augmentation pipeline for training."""
    return Compose([
        RandomFlip3D(flip_prob),
        RandomIntensityShift(intensity_shift, intensity_scale),
        RandomGaussianNoise(noise_std),
    ])


def build_val_transforms() -> Compose:
    """Return an identity (no-op) pipeline for validation."""
    return Compose([])

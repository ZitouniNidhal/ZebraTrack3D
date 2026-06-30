"""
test_data_loader.py
───────────────────
Unit tests for src/data/loader.py
"""

import numpy as np
import pytest
import zarr

from src.data.loader import ZarrPatchDataset, ZarrInferenceDataset


@pytest.fixture
def tmp_zarr(tmp_path):
    """Create a small in-memory Zarr array for testing."""
    path = tmp_path / "test.zarr"
    z = zarr.open(str(path), mode="w", shape=(1, 32, 64, 64), dtype="float32")
    z[:] = np.random.rand(1, 32, 64, 64).astype(np.float32)
    return path


class TestZarrPatchDataset:
    def test_len(self, tmp_zarr):
        ds = ZarrPatchDataset(tmp_zarr, patch_size=(16, 32, 32), n_patches=50)
        assert len(ds) == 50

    def test_item_shape(self, tmp_zarr):
        ds = ZarrPatchDataset(tmp_zarr, patch_size=(16, 32, 32), n_patches=10)
        sample = ds[0]
        assert "image" in sample
        # shape: (1, Z, Y, X) — channel dim prepended in __getitem__
        assert sample["image"].shape[-3:] == (16, 32, 32)

    def test_no_label(self, tmp_zarr):
        ds = ZarrPatchDataset(tmp_zarr, patch_size=(16, 32, 32), n_patches=5)
        sample = ds[0]
        assert "label" not in sample


class TestZarrInferenceDataset:
    def test_patch_count(self, tmp_zarr):
        ds = ZarrInferenceDataset(tmp_zarr, patch_size=(16, 32, 32), overlap=(2, 4, 4))
        assert len(ds) > 0

    def test_item_has_slices(self, tmp_zarr):
        ds = ZarrInferenceDataset(tmp_zarr, patch_size=(16, 32, 32), overlap=(2, 4, 4))
        item = ds[0]
        assert "image" in item
        assert "slices" in item
        assert len(item["slices"]) == 3

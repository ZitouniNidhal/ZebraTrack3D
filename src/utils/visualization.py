"""
visualization.py
────────────────
3D visualization utilities for cells, tracks, and lineage trees.

Backends supported
──────────────────
- **matplotlib**: static 3D scatter / line plots.
- **plotly**: interactive HTML plots.
- **napari**: live viewer (requires napari installation).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Matplotlib
# ─────────────────────────────────────────────────────────────────────────────

def plot_detections_3d(
    coords: np.ndarray,
    timepoints: Optional[np.ndarray] = None,
    title: str = "Cell Detections",
    save_path: Optional[str | Path] = None,
) -> None:
    """3D scatter plot of cell coordinates, optionally coloured by timepoint."""
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    c = timepoints if timepoints is not None else "steelblue"
    sc = ax.scatter(coords[:, 2], coords[:, 1], coords[:, 0],
                    c=c, cmap="plasma", s=8, alpha=0.7)
    if timepoints is not None:
        plt.colorbar(sc, ax=ax, label="Timepoint")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_tracks_3d(
    tracks_coords: List[np.ndarray],
    title: str = "Cell Tracks",
    save_path: Optional[str | Path] = None,
) -> None:
    """Plot each track as a 3D polyline.

    Parameters
    ----------
    tracks_coords:
        List of arrays, each of shape (T, 3) with (z, y, x) coords.
    """
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    cmap = plt.cm.get_cmap("tab20", len(tracks_coords))
    for i, coords in enumerate(tracks_coords):
        ax.plot(coords[:, 2], coords[:, 1], coords[:, 0],
                color=cmap(i), linewidth=1.0, alpha=0.8)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Plotly (interactive)
# ─────────────────────────────────────────────────────────────────────────────

def plotly_tracks(
    tracks_coords: List[np.ndarray],
    track_ids: Optional[List[int]] = None,
    save_html: Optional[str | Path] = None,
) -> None:
    """Interactive 3D track visualization with plotly."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        logger.warning("plotly not installed. Run: pip install plotly")
        return

    fig = go.Figure()
    for i, coords in enumerate(tracks_coords):
        name = f"Track {track_ids[i]}" if track_ids else f"Track {i}"
        fig.add_trace(go.Scatter3d(
            x=coords[:, 2], y=coords[:, 1], z=coords[:, 0],
            mode="lines+markers",
            marker=dict(size=3),
            line=dict(width=2),
            name=name,
        ))

    fig.update_layout(
        title="Cell Tracks (3D)",
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        template="plotly_dark",
    )
    if save_html:
        Path(save_html).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(save_html))
        logger.info("Saved interactive plot to %s", save_html)
    fig.show()


# ─────────────────────────────────────────────────────────────────────────────
# napari
# ─────────────────────────────────────────────────────────────────────────────

def napari_viewer(
    volume: Optional[np.ndarray] = None,
    cell_coords: Optional[np.ndarray] = None,
    tracks_data: Optional[np.ndarray] = None,
) -> None:
    """Launch a napari viewer with optional volume, points, and tracks layers.

    Parameters
    ----------
    volume:
        3D or 4D (T, Z, Y, X) numpy array.
    cell_coords:
        Array of shape (N, 3) or (N, 4) with [t, z, y, x] columns.
    tracks_data:
        Array compatible with napari Tracks layer:
        columns [track_id, t, z, y, x].
    """
    try:
        import napari
    except ImportError:
        logger.warning("napari not installed. Run: pip install napari[all]")
        return

    viewer = napari.Viewer()
    if volume is not None:
        viewer.add_image(volume, name="Microscopy Volume", colormap="gray")
    if cell_coords is not None:
        viewer.add_points(cell_coords, name="Cell Detections", size=5, face_color="red")
    if tracks_data is not None:
        viewer.add_tracks(tracks_data, name="Tracks")
    napari.run()

"""
division_detector.py
────────────────────
Detect cell division events (mitosis) in tracked data.

Two strategies:
1. **Heuristic** — detect tracks that suddenly have ≥2 closely spaced
   daughter cells in the next frame.
2. **CNN classifier** (stub) — classify candidate pairs as division/no-division
   using a small 3D patch-level network.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.models.tracking.graph_based import Detection, Track

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Heuristic Division Detector
# ─────────────────────────────────────────────────────────────────────────────

class HeuristicDivisionDetector:
    """Flag division events based on spatial proximity of potential daughters.

    A division is flagged when a track ends at time *t* and exactly two
    detections in frame *t+1* are within *max_daughter_dist* of its last
    position.

    Parameters
    ----------
    max_daughter_dist:
        Maximum distance (voxels) between parent's last position and each
        daughter's position.
    min_daughter_angle:
        Minimum angle (degrees) between the two daughter displacement vectors
        to reject collinear noise detections.
    """

    def __init__(
        self,
        max_daughter_dist: float = 15.0,
        min_daughter_angle: float = 30.0,
    ) -> None:
        self.max_daughter_dist = max_daughter_dist
        self.min_daughter_angle = min_daughter_angle

    def detect(
        self,
        tracks: List[Track],
        detections_per_frame: dict,
    ) -> List[Tuple[int, int, int]]:
        """Return a list of (parent_track_id, daughter1_cell_id, daughter2_cell_id)."""
        events: List[Tuple[int, int, int]] = []

        for track in tracks:
            if not track.detections:
                continue
            last_det = track.detections[-1]
            t_next = last_det.t + 1
            next_dets: List[Detection] = detections_per_frame.get(t_next, [])

            candidates = [
                d for d in next_dets
                if np.linalg.norm(d.coords - last_det.coords) < self.max_daughter_dist
            ]
            if len(candidates) == 2:
                v1 = candidates[0].coords - last_det.coords
                v2 = candidates[1].coords - last_det.coords
                cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                angle = np.degrees(np.arccos(np.clip(cos, -1, 1)))
                if angle >= self.min_daughter_angle:
                    events.append((track.track_id, candidates[0].cell_id, candidates[1].cell_id))

        logger.info("Detected %d division events.", len(events))
        return events


# ─────────────────────────────────────────────────────────────────────────────
# CNN-Based Division Classifier (stub)
# ─────────────────────────────────────────────────────────────────────────────

class DivisionClassifierCNN(nn.Module):
    """Lightweight 3D CNN to classify a patch around a candidate division.

    Input : (B, 1, Z, Y, X) intensity patch.
    Output: (B, 2) logits — [no_division, division].
    """

    def __init__(self, patch_size: int = 16) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
        )
        self.classifier = nn.Linear(64, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x).flatten(1)
        return self.classifier(x)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, patch: torch.Tensor, threshold: float = 0.5) -> bool:
        """Return True if the patch is classified as a division."""
        self.eval()
        logits = self.forward(patch.unsqueeze(0))
        prob = torch.softmax(logits, dim=1)[0, 1].item()
        return prob >= threshold

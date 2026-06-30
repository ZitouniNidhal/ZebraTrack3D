"""
metrics.py
──────────
Competition metrics for the Biohub Cell Tracking Challenge.

Metrics
───────
- **Edge Jaccard (EJ)**: IoU over the set of edges in the tracking graph.
- **Division Jaccard (DJ)**: IoU over the set of detected division events.
- **Combined score**: 0.5 * EJ + 0.5 * DJ.

Node matching uses Optimal Bipartite Matching (Hungarian algorithm) based on
spatial overlap (IoU of instance segmentation masks).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Type aliases
# ─────────────────────────────────────────────────────────────────────────────

Edge = Tuple[int, int]         # (from_node, to_node)  — integers are det IDs
DivEvent = Tuple[int, int, int]  # (parent, daughter1, daughter2)


# ─────────────────────────────────────────────────────────────────────────────
# Node matching (instance segmentation IoU)
# ─────────────────────────────────────────────────────────────────────────────

def compute_iou_matrix(
    pred_masks: Dict[int, np.ndarray],
    gt_masks: Dict[int, np.ndarray],
) -> np.ndarray:
    """Compute IoU between all pairs of predicted and GT instance masks.

    Parameters
    ----------
    pred_masks, gt_masks:
        Dicts mapping cell_id → binary boolean array (same spatial shape).

    Returns
    -------
    iou_matrix of shape (n_pred, n_gt).
    """
    pred_ids = sorted(pred_masks)
    gt_ids = sorted(gt_masks)
    iou = np.zeros((len(pred_ids), len(gt_ids)), dtype=np.float32)
    for i, pid in enumerate(pred_ids):
        p = pred_masks[pid]
        for j, gid in enumerate(gt_ids):
            g = gt_masks[gid]
            inter = (p & g).sum()
            union = (p | g).sum()
            iou[i, j] = inter / (union + 1e-8)
    return iou


def match_nodes(
    iou_matrix: np.ndarray,
    threshold: float = 0.5,
) -> Dict[int, int]:
    """Match predicted nodes to GT nodes using optimal bipartite matching.

    Returns a dict {pred_index: gt_index} for pairs above *threshold*.
    """
    # Hungarian algorithm minimizes cost → negate IoU
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    matches: Dict[int, int] = {}
    for r, c in zip(row_ind, col_ind):
        if iou_matrix[r, c] >= threshold:
            matches[r] = c
    return matches


# ─────────────────────────────────────────────────────────────────────────────
# Edge Jaccard
# ─────────────────────────────────────────────────────────────────────────────

def edge_jaccard(
    pred_edges: Set[Edge],
    gt_edges: Set[Edge],
) -> float:
    """Compute Edge Jaccard Index (IoU over edge sets).

    Parameters
    ----------
    pred_edges, gt_edges:
        Sets of (from_matched_id, to_matched_id) edges using matched node IDs.
    """
    if not pred_edges and not gt_edges:
        return 1.0
    tp = len(pred_edges & gt_edges)
    fp = len(pred_edges - gt_edges)
    fn = len(gt_edges - pred_edges)
    return tp / (tp + fp + fn + 1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# Division Jaccard
# ─────────────────────────────────────────────────────────────────────────────

def division_jaccard(
    pred_divisions: Set[DivEvent],
    gt_divisions: Set[DivEvent],
) -> float:
    """Compute Division Jaccard Index (IoU over division-event sets)."""
    if not pred_divisions and not gt_divisions:
        return 1.0
    tp = len(pred_divisions & gt_divisions)
    fp = len(pred_divisions - gt_divisions)
    fn = len(gt_divisions - pred_divisions)
    return tp / (tp + fp + fn + 1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# Combined evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    pred_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    w_edge: float = 0.5,
    w_div: float = 0.5,
) -> Dict[str, float]:
    """Compute Edge Jaccard, Division Jaccard, and combined score.

    Parameters
    ----------
    pred_df, gt_df:
        DataFrames with columns:
        ``["track_id", "parent_id", "t", "z", "y", "x"]``.

    Returns
    -------
    Dict with keys: ``edge_jaccard``, ``division_jaccard``, ``combined``.
    """
    # Build edge sets: (track_id_at_t-1, track_id_at_t)
    def _edges(df: pd.DataFrame) -> Set[Edge]:
        edges: Set[Edge] = set()
        for tid, group in df.groupby("track_id"):
            group = group.sort_values("t")
            ids = group["track_id"].tolist()
            for i in range(len(ids) - 1):
                edges.add((ids[i], ids[i + 1]))
        return edges

    # Build division sets: (parent_track_id, child1, child2)
    def _divisions(df: pd.DataFrame) -> Set[DivEvent]:
        divs: Set[DivEvent] = set()
        parent_map: Dict[int, List[int]] = {}
        for row in df.itertuples():
            if row.parent_id is not None and not np.isnan(float(row.parent_id)):
                parent_map.setdefault(int(row.parent_id), []).append(row.track_id)
        for parent, children in parent_map.items():
            if len(children) == 2:
                divs.add((parent, children[0], children[1]))
        return divs

    ej = edge_jaccard(_edges(pred_df), _edges(gt_df))
    dj = division_jaccard(_divisions(pred_df), _divisions(gt_df))
    combined = w_edge * ej + w_div * dj

    results = {"edge_jaccard": ej, "division_jaccard": dj, "combined": combined}
    logger.info("Evaluation results: %s", results)
    return results

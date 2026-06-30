# -*- coding: utf-8 -*-
"""metrics.py
────────────────────
Utility functions for evaluating cell‑tracking predictions.

The original implementation exposed only Edge Jaccard, Division Jaccard and a combined
score.  This version adds:
* type hints and exhaustive docstrings (Google style)
* input validation for the pandas DataFrames
* helper functions that clearly separate node‑matching, edge extraction and division extraction
* precision, recall and F1 for edges and divisions
* a small internal ``_compute_iou_matrix`` used by the matching routine (kept for possible future extensions)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Type aliases
# ──────────────────────────────────────────────────────────────────────────────
Edge = Tuple[int, int]  # (source_track_id, target_track_id)
DivEvent = Tuple[int, int, int]  # (parent_track_id, daughter1, daughter2)

# ──────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ──────────────────────────────────────────────────────────────────────────────

def _compute_iou_matrix(
    pred_masks: Dict[int, np.ndarray],
    gt_masks: Dict[int, np.ndarray],
) -> np.ndarray:
    """Return the IoU matrix between every predicted and ground‑truth mask.

    Parameters
    ----------
    pred_masks, gt_masks:
        Mapping ``cell_id → binary mask`` with identical spatial dimensions.

    Returns
    -------
    np.ndarray
        Shape ``(n_pred, n_gt)`` where each entry is the Intersection‑over‑Union.
    """
    pred_ids = sorted(pred_masks)
    gt_ids = sorted(gt_masks)
    iou = np.zeros((len(pred_ids), len(gt_ids)), dtype=np.float32)
    for i, pid in enumerate(pred_ids):
        p = pred_masks[pid]
        for j, gid in enumerate(gt_ids):
            g = gt_masks[gid]
            inter = np.logical_and(p, g).sum()
            union = np.logical_or(p, g).sum()
            iou[i, j] = inter / (union + 1e-8)
    return iou


def _match_nodes(iou_matrix: np.ndarray, threshold: float = 0.5) -> Dict[int, int]:
    """Perform optimal bipartite matching on an IoU matrix.

    Returns a mapping from ``pred_index`` to ``gt_index`` for all pairs whose IoU
    exceeds *threshold*.
    """
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    matches: Dict[int, int] = {}
    for r, c in zip(row_ind, col_ind):
        if iou_matrix[r, c] >= threshold:
            matches[r] = c
    return matches

# ──────────────────────────────────────────────────────────────────────────────
# Edge Jaccard and Division Jaccard – unchanged core definitions
# ──────────────────────────────────────────────────────────────────────────────

def edge_jaccard(pred_edges: Set[Edge], gt_edges: Set[Edge]) -> float:
    """Intersection‑over‑Union of two edge sets.

    An empty‑set vs empty‑set is defined as perfect (1.0).
    """
    if not pred_edges and not gt_edges:
        return 1.0
    tp = len(pred_edges & gt_edges)
    fp = len(pred_edges - gt_edges)
    fn = len(gt_edges - pred_edges)
    return tp / (tp + fp + fn + 1e-8)


def division_jaccard(pred_divs: Set[DivEvent], gt_divs: Set[DivEvent]) -> float:
    """Intersection‑over‑Union of division event sets.

    Handles the empty‑set edge case identically to ``edge_jaccard``.
    """
    if not pred_divs and not gt_divs:
        return 1.0
    tp = len(pred_divs & gt_divs)
    fp = len(pred_divs - gt_divs)
    fn = len(gt_divs - pred_divs)
    return tp / (tp + fp + fn + 1e-8)

# ──────────────────────────────────────────────────────────────────────────────
# Precision / Recall / F1 helpers
# ──────────────────────────────────────────────────────────────────────────────

def _prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """Return precision, recall and F1 given true‑positive, false‑positive and false-negative counts."""
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return precision, recall, f1


def edge_prf(pred_edges: Set[Edge], gt_edges: Set[Edge]) -> Tuple[float, float, float]:
    """Precision, recall and F1 for edge sets."""
    tp = len(pred_edges & gt_edges)
    fp = len(pred_edges - gt_edges)
    fn = len(gt_edges - pred_edges)
    return _prf(tp, fp, fn)


def division_prf(pred_divs: Set[DivEvent], gt_divs: Set[DivEvent]) -> Tuple[float, float, float]:
    """Precision, recall and F1 for division event sets."""
    tp = len(pred_divs & gt_divs)
    fp = len(pred_divs - gt_divs)
    fn = len(gt_divs - pred_divs)
    return _prf(tp, fp, fn)

# ──────────────────────────────────────────────────────────────────────────────
# Core evaluation routine
# ──────────────────────────────────────────────────────────────────────────────

def _validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate that a tracking DataFrame contains the required columns.

    Expected columns are ``["track_id", "parent_id", "t", "z", "y", "x"]``.
    ``parent_id`` may contain ``NaN`` for root nodes.
    """
    required = {"track_id", "parent_id", "t", "z", "y", "x"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")
    return df


def _edges(df: pd.DataFrame) -> Set[Edge]:
    """Extract the set of edges ``(track_id_t, track_id_t+1)`` from a DataFrame."""
    edges: Set[Edge] = set()
    for _, group in df.groupby("track_id"):
        ordered = group.sort_values("t")["track_id"].tolist()
        for i in range(len(ordered) - 1):
            edges.add((ordered[i], ordered[i + 1]))
    return edges


def _divisions(df: pd.DataFrame) -> Set[DivEvent]:
    """Extract division events ``(parent, daughter1, daughter2)``.

    Only divisions with exactly two children are considered.
    """
    parent_map: Dict[int, List[int]] = {}
    for row in df.itertuples():
        if getattr(row, "parent_id") is not None and not pd.isna(row.parent_id):
            parent_map.setdefault(int(row.parent_id), []).append(int(row.track_id))
    divisions: Set[DivEvent] = set()
    for parent, children in parent_map.items():
        if len(children) == 2:
            divisions.add((parent, children[0], children[1]))
    return divisions


def evaluate(
    pred_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    w_edge: float = 0.5,
    w_div: float = 0.5,
) -> Dict[str, float]:
    """Compute a suite of tracking metrics.

    Parameters
    ----------
    pred_df, gt_df:
        DataFrames describing the predicted and ground‑truth tracks.  They must
        contain the columns ``track_id, parent_id, t, z, y, x``.
    w_edge, w_div:
        Weights for the combined score – the default mirrors the original
        implementation (simple average).

    Returns
    -------
    dict
        ``edge_jaccard``, ``division_jaccard``, ``edge_precision``,
        ``edge_recall``, ``edge_f1``, ``division_precision``, ``division_recall``,
        ``division_f1`` and ``combined``.
    """
    pred_df = _validate_dataframe(pred_df)
    gt_df = _validate_dataframe(gt_df)

    pred_edges = _edges(pred_df)
    gt_edges = _edges(gt_df)
    pred_divs = _divisions(pred_df)
    gt_divs = _divisions(gt_df)

    ej = edge_jaccard(pred_edges, gt_edges)
    dj = division_jaccard(pred_divs, gt_divs)
    ep, er, ef1 = edge_prf(pred_edges, gt_edges)
    dp, dr, df1 = division_prf(pred_divs, gt_divs)
    combined = w_edge * ej + w_div * dj

    results = {
        "edge_jaccard": ej,
        "division_jaccard": dj,
        "edge_precision": ep,
        "edge_recall": er,
        "edge_f1": ef1,
        "division_precision": dp,
        "division_recall": dr,
        "division_f1": df1,
        "combined": combined,
    }
    logger.info("Evaluation results: %s", results)
    return results

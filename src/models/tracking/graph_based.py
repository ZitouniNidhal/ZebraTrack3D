"""
graph_based.py
──────────────
Graph-based cell tracking using Min-Cost Flow (MCF) or the Hungarian algorithm.

Workflow
────────
1. Build a directed graph:
   - Source (s) and Sink (t) nodes.
   - One "cell node" per detected cell per timepoint.
   - Edges:  s → cell, cell → cell′ (next frame), cell → t.
2. Solve for minimum-cost flow to determine optimal assignments.
3. Extract tracks from the flow solution.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Detection:
    """A single cell detection at one timepoint."""
    t: int
    cell_id: int
    z: float
    y: float
    x: float
    score: float = 1.0

    @property
    def coords(self) -> np.ndarray:
        return np.array([self.z, self.y, self.x])


@dataclass
class Track:
    """A sequence of cell detections forming a track."""
    track_id: int
    detections: List[Detection] = field(default_factory=list)
    parent_id: Optional[int] = None   # set for daughter cells after division

    def add(self, det: Detection) -> None:
        self.detections.append(det)

    @property
    def start_t(self) -> int:
        return self.detections[0].t if self.detections else -1

    @property
    def end_t(self) -> int:
        return self.detections[-1].t if self.detections else -1


# ─────────────────────────────────────────────────────────────────────────────
# Hungarian Tracker
# ─────────────────────────────────────────────────────────────────────────────

class HungarianTracker:
    """Frame-by-frame greedy tracker using the Hungarian algorithm.

    Parameters
    ----------
    max_distance:
        Maximum Euclidean distance (in voxels) to allow a link.
    max_gap:
        Maximum number of frames a cell can disappear.
    """

    def __init__(self, max_distance: float = 20.0, max_gap: int = 2) -> None:
        self.max_distance = max_distance
        self.max_gap = max_gap
        self._next_id = 0

    # ------------------------------------------------------------------
    def track(self, detections_per_frame: Dict[int, List[Detection]]) -> List[Track]:
        """Run tracking on a dict mapping timepoint → list of detections."""
        active: Dict[int, Track] = {}      # track_id → Track (still alive)
        finished: List[Track] = []
        timepoints = sorted(detections_per_frame)

        for t in timepoints:
            dets = detections_per_frame[t]
            if not dets:
                # Age out lost tracks
                to_finish = [tid for tid, tr in active.items()
                             if t - tr.end_t > self.max_gap]
                for tid in to_finish:
                    finished.append(active.pop(tid))
                continue

            if not active:
                for det in dets:
                    tid = self._new_id()
                    track = Track(track_id=tid)
                    track.add(det)
                    active[tid] = track
                continue

            # Build cost matrix
            track_list = list(active.values())
            prev_coords = np.array([tr.detections[-1].coords for tr in track_list])
            curr_coords = np.array([d.coords for d in dets])
            cost = cdist(prev_coords, curr_coords)

            # Apply distance threshold
            cost[cost > self.max_distance] = 1e9

            row_ind, col_ind = linear_sum_assignment(cost)

            matched_dets = set()
            for r, c in zip(row_ind, col_ind):
                if cost[r, c] < 1e9:
                    track_list[r].add(dets[c])
                    matched_dets.add(c)

            # Unmatched detections → new tracks
            for c, det in enumerate(dets):
                if c not in matched_dets:
                    tid = self._new_id()
                    track = Track(track_id=tid)
                    track.add(det)
                    active[tid] = track

            # Age out lost tracks
            to_finish = [tid for tid, tr in active.items()
                         if tr.end_t < t - self.max_gap]
            for tid in to_finish:
                finished.append(active.pop(tid))

        finished.extend(active.values())
        logger.info("Tracking complete. Total tracks: %d", len(finished))
        return finished

    def _new_id(self) -> int:
        tid = self._next_id
        self._next_id += 1
        return tid


# ─────────────────────────────────────────────────────────────────────────────
# Min-Cost Flow Tracker (graph-based)
# ─────────────────────────────────────────────────────────────────────────────

class MinCostFlowTracker:
    """Track cells across frames using networkx min-cost flow.

    This is a simplified formulation suitable for moderate-scale problems.
    For very large embryos consider using OR-Tools or GLPK directly.
    """

    def __init__(
        self,
        max_distance: float = 20.0,
        max_gap: int = 2,
        entry_cost: float = 5.0,
        exit_cost: float = 5.0,
    ) -> None:
        self.max_distance = max_distance
        self.max_gap = max_gap
        self.entry_cost = entry_cost
        self.exit_cost = exit_cost

    # ------------------------------------------------------------------
    def track(self, detections_per_frame: Dict[int, List[Detection]]) -> List[Track]:
        """Solve tracking as a min-cost flow problem."""
        G = nx.DiGraph()
        G.add_node("source")
        G.add_node("sink")

        all_dets: Dict[str, Detection] = {}
        timepoints = sorted(detections_per_frame)

        # Add cell nodes
        for t, dets in detections_per_frame.items():
            for det in dets:
                node = f"t{t}_c{det.cell_id}"
                all_dets[node] = det
                G.add_node(node)
                G.add_edge("source", node, capacity=1, weight=int(self.entry_cost))
                G.add_edge(node, "sink", capacity=1, weight=int(self.exit_cost))

        # Add linking edges between consecutive frames
        for i in range(len(timepoints) - 1):
            for gap in range(1, self.max_gap + 2):
                if i + gap >= len(timepoints):
                    break
                t1, t2 = timepoints[i], timepoints[i + gap]
                for det1 in detections_per_frame.get(t1, []):
                    for det2 in detections_per_frame.get(t2, []):
                        dist = float(np.linalg.norm(det1.coords - det2.coords))
                        if dist <= self.max_distance:
                            n1 = f"t{t1}_c{det1.cell_id}"
                            n2 = f"t{t2}_c{det2.cell_id}"
                            cost = int(dist)
                            G.add_edge(n1, n2, capacity=1, weight=cost)

        # Solve min-cost flow
        flow_dict = nx.min_cost_flow(G)

        # Extract tracks from flow
        tracks = self._extract_tracks(flow_dict, all_dets, timepoints)
        logger.info("MCF tracking complete. Total tracks: %d", len(tracks))
        return tracks

    # ------------------------------------------------------------------
    def _extract_tracks(
        self,
        flow_dict: dict,
        all_dets: Dict[str, Detection],
        timepoints: List[int],
    ) -> List[Track]:
        tracks: List[Track] = []
        visited: set = set()
        tid = 0

        for node, det in all_dets.items():
            if node in visited:
                continue
            # Check if this node is the start of a track
            if flow_dict.get("source", {}).get(node, 0) == 1:
                track = Track(track_id=tid)
                tid += 1
                curr = node
                while curr in all_dets:
                    visited.add(curr)
                    track.add(all_dets[curr])
                    # Follow the flow
                    nxt = None
                    for neighbor, f in flow_dict.get(curr, {}).items():
                        if f == 1 and neighbor != "sink" and neighbor in all_dets:
                            nxt = neighbor
                            break
                    curr = nxt  # type: ignore[assignment]
                    if curr is None:
                        break
                tracks.append(track)

        return tracks


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def tracks_to_dataframe(tracks: List[Track]) -> pd.DataFrame:
    """Convert a list of Track objects to a tidy DataFrame."""
    rows = []
    for tr in tracks:
        for det in tr.detections:
            rows.append(
                {
                    "track_id": tr.track_id,
                    "parent_id": tr.parent_id,
                    "t": det.t,
                    "z": det.z,
                    "y": det.y,
                    "x": det.x,
                    "score": det.score,
                }
            )
    return pd.DataFrame(rows)


def detections_from_dataframe(df: pd.DataFrame) -> Dict[int, List[Detection]]:
    """Load detections from a tidy CSV DataFrame."""
    out: Dict[int, List[Detection]] = {}
    for row in df.itertuples(index=False):
        det = Detection(t=row.t, cell_id=row.cell_id, z=row.z, y=row.y, x=row.x)
        out.setdefault(row.t, []).append(det)
    return out

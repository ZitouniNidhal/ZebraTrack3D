"""
tree_builder.py
───────────────
Construct and export cell lineage trees from tracks and division events.

A lineage tree is a directed acyclic graph (DAG) where:
  - Each node is a track (identified by its track_id).
  - Each edge parent → child represents a division event.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import pandas as pd

from src.models.tracking.graph_based import Track

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# LineageTree
# ─────────────────────────────────────────────────────────────────────────────

class LineageTree:
    """Build and query a lineage tree from tracking results.

    Parameters
    ----------
    tracks:
        All detected tracks.
    division_events:
        List of (parent_track_id, daughter1_track_id, daughter2_track_id).
    """

    def __init__(
        self,
        tracks: List[Track],
        division_events: Optional[List[Tuple[int, int, int]]] = None,
    ) -> None:
        self.tracks: Dict[int, Track] = {tr.track_id: tr for tr in tracks}
        self.division_events = division_events or []
        self.graph = nx.DiGraph()
        self._build()

    # ------------------------------------------------------------------
    def _build(self) -> None:
        """Build the lineage DAG."""
        for tid in self.tracks:
            self.graph.add_node(tid, **self._node_attrs(tid))

        for parent, d1, d2 in self.division_events:
            self.graph.add_edge(parent, d1, relation="division")
            self.graph.add_edge(parent, d2, relation="division")

        logger.info(
            "Lineage tree: %d nodes, %d division edges.",
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
        )

    # ------------------------------------------------------------------
    def _node_attrs(self, tid: int) -> dict:
        tr = self.tracks[tid]
        return {
            "start_t": tr.start_t,
            "end_t": tr.end_t,
            "length": len(tr.detections),
            "parent_id": tr.parent_id,
        }

    # ------------------------------------------------------------------
    def root_tracks(self) -> List[int]:
        """Return track IDs with no parent (founders)."""
        return [n for n in self.graph if self.graph.in_degree(n) == 0]

    def subtree(self, root_id: int) -> nx.DiGraph:
        """Return the subgraph rooted at *root_id*."""
        descendants = nx.descendants(self.graph, root_id) | {root_id}
        return self.graph.subgraph(descendants).copy()

    def generation(self, track_id: int) -> int:
        """Return the generation depth of a track (0 = founder)."""
        ancestors = nx.ancestors(self.graph, track_id)
        return len(ancestors)

    # ------------------------------------------------------------------
    def to_dataframe(self) -> pd.DataFrame:
        """Export the lineage as a tidy CSV-ready DataFrame."""
        rows = []
        for parent, child, data in self.graph.edges(data=True):
            rows.append(
                {
                    "parent_track_id": parent,
                    "child_track_id": child,
                    "relation": data.get("relation", "division"),
                }
            )
        return pd.DataFrame(rows, columns=["parent_track_id", "child_track_id", "relation"])

    def save_csv(self, path: str | Path) -> None:
        """Save the lineage edge list to CSV."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_csv(path, index=False)
        logger.info("Lineage saved to %s", path)

    def save_json(self, path: str | Path) -> None:
        """Save the lineage as a node-link JSON."""
        data = nx.node_link_data(self.graph)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Lineage JSON saved to %s", path)

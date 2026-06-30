"""
test_metrics.py
───────────────
Unit tests for src/utils/metrics.py
"""

import numpy as np
import pytest
import pandas as pd

from src.utils.metrics import edge_jaccard, division_jaccard, evaluate


class TestEdgeJaccard:
    def test_perfect_match(self):
        edges = {(0, 1), (1, 2), (2, 3)}
        assert edge_jaccard(edges, edges) == pytest.approx(1.0, abs=1e-4)

    def test_no_overlap(self):
        pred = {(0, 1), (1, 2)}
        gt = {(5, 6), (6, 7)}
        assert edge_jaccard(pred, gt) == pytest.approx(0.0, abs=1e-4)

    def test_partial_overlap(self):
        pred = {(0, 1), (1, 2), (2, 3)}
        gt = {(1, 2), (2, 3), (3, 4)}
        ej = edge_jaccard(pred, gt)
        # TP=2, FP=1, FN=1 → 2/(2+1+1)=0.5
        assert ej == pytest.approx(0.5, abs=1e-4)

    def test_empty_sets(self):
        assert edge_jaccard(set(), set()) == pytest.approx(1.0, abs=1e-4)


class TestDivisionJaccard:
    def test_perfect_match(self):
        divs = {(0, 1, 2), (3, 4, 5)}
        assert division_jaccard(divs, divs) == pytest.approx(1.0, abs=1e-4)

    def test_empty(self):
        assert division_jaccard(set(), set()) == pytest.approx(1.0, abs=1e-4)

    def test_no_overlap(self):
        pred = {(0, 1, 2)}
        gt = {(3, 4, 5)}
        assert division_jaccard(pred, gt) == pytest.approx(0.0, abs=1e-4)


class TestEvaluate:
    def _make_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "track_id":  [0, 0, 0, 1, 1],
            "parent_id": [None, None, None, None, None],
            "t":         [0, 1, 2, 0, 1],
            "z":         [0, 0, 0, 5, 5],
            "y":         [0, 0, 0, 5, 5],
            "x":         [0, 0, 0, 5, 5],
        })

    def test_perfect_score(self):
        df = self._make_df()
        results = evaluate(df, df)
        assert results["edge_jaccard"] == pytest.approx(1.0, abs=1e-4)
        assert results["combined"] == pytest.approx(1.0, abs=1e-4)

"""
evaluate.py
───────────
Compute Edge Jaccard and Division Jaccard on a prediction CSV.

Usage
─────
  python scripts/evaluate.py \\
    --pred outputs/predictions/submission.csv \\
    --gt   data/processed/ground_truth.csv
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from src.utils.metrics import evaluate

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ZebraTrack3D predictions.")
    parser.add_argument("--pred", required=True, help="Path to prediction CSV")
    parser.add_argument("--gt", required=True, help="Path to ground-truth CSV")
    parser.add_argument("--w-edge", type=float, default=0.5)
    parser.add_argument("--w-div", type=float, default=0.5)
    args = parser.parse_args()

    pred_df = pd.read_csv(args.pred)
    gt_df = pd.read_csv(args.gt)

    results = evaluate(pred_df, gt_df, w_edge=args.w_edge, w_div=args.w_div)
    print("\n=== Evaluation Results ===")
    for k, v in results.items():
        print(f"  {k:<25} {v:.4f}")
    print("=" * 30)


if __name__ == "__main__":
    main()

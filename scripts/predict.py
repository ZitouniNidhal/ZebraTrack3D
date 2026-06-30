"""
predict.py
──────────
Run full inference pipeline: detect → track → lineage → export CSV.

Usage
─────
  python scripts/predict.py --input data/raw/test.zarr \\
         --output outputs/predictions/submission.csv \\
         --checkpoint outputs/models/best_model.pth
"""

import subprocess
import sys

if __name__ == "__main__":
    args = ["python", "-m", "src.main", "predict"] + sys.argv[1:]
    subprocess.run(args, check=True)

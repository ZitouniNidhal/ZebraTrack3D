"""
train.py
────────
Standalone training script. Delegates to src/main.py train.

Usage
─────
  python scripts/train.py --config configs/params.yaml
  python scripts/train.py --config configs/params.yaml --resume outputs/models/best_model.pth
"""

import subprocess
import sys

if __name__ == "__main__":
    args = ["python", "-m", "src.main", "train"] + sys.argv[1:]
    subprocess.run(args, check=True)

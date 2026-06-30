"""
submit.py
─────────
Format predictions and submit to Kaggle.

Usage
─────
  python scripts/submit.py \\
    --file outputs/predictions/submission.csv \\
    --message "3D U-Net + MCF v1"
"""

from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit predictions to Kaggle.")
    parser.add_argument("--file", "-f", required=True,
                        help="Path to submission CSV")
    parser.add_argument("--message", "-m", default="ZebraTrack3D submission",
                        help="Submission message")
    parser.add_argument("--competition", "-c",
                        default="biohub-cell-tracking-during-development",
                        help="Kaggle competition slug")
    args = parser.parse_args()

    cmd = [
        "kaggle", "competitions", "submit",
        "-c", args.competition,
        "-f", args.file,
        "-m", args.message,
    ]
    print(f"Submitting: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print("Error:", result.stderr, file=sys.stderr)
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()

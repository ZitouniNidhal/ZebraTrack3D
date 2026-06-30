"""
main.py
───────
CLI entry point for ZebraTrack3D — training and inference.

Usage
─────
  python src/main.py train  --config configs/params.yaml
  python src/main.py predict --config configs/params.yaml \\
         --input data/raw/test.zarr --output outputs/predictions/submission.csv
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
import yaml

# ─────────────────────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("zebratrack3d")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

@click.group()
def cli() -> None:
    """ZebraTrack3D — 3D cell tracking pipeline CLI."""


# ──────────────────────────────────────────────
# Train
# ──────────────────────────────────────────────

@cli.command()
@click.option("--config", "-c", default="configs/params.yaml",
              help="Path to params.yaml", show_default=True)
@click.option("--resume", "-r", default=None,
              help="Path to checkpoint to resume from")
def train(config: str, resume: str | None) -> None:
    """Train the detection model."""
    cfg = _load_config(config)
    logger.info("Starting training with config: %s", config)

    from src.data.loader import build_dataloaders
    from src.data.preprocess import build_train_transforms, build_val_transforms
    from src.models.detection import UNet3D

    train_loader, val_loader = build_dataloaders(
        train_zarr=cfg["data"]["zarr_root"] + "train/",
        val_zarr=cfg["data"]["zarr_root"] + "val/",
        patch_size=tuple(cfg["data"]["patch_size"]),
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        transform_train=build_train_transforms(),
        transform_val=build_val_transforms(),
    )

    model = UNet3D(
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        features=cfg["model"]["features"],
        norm=cfg["model"]["norm"],
        dropout=cfg["model"]["dropout"],
    )

    _run_training(model, train_loader, val_loader, cfg, resume)


# ──────────────────────────────────────────────
# Predict
# ──────────────────────────────────────────────

@cli.command()
@click.option("--config", "-c", default="configs/params.yaml", show_default=True)
@click.option("--input", "-i", required=True, help="Path to input .zarr file")
@click.option("--output", "-o", default="outputs/predictions/submission.csv",
              show_default=True, help="Output CSV path")
@click.option("--checkpoint", "-ckpt", default=None,
              help="Model checkpoint path")
def predict(config: str, input: str, output: str, checkpoint: str | None) -> None:
    """Run full pipeline: detect → track → lineage → export."""
    cfg = _load_config(config)
    logger.info("Running inference on: %s", input)

    from src.data.loader import ZarrInferenceDataset
    from src.models.detection import UNet3D
    from src.models.tracking import MinCostFlowTracker
    from src.models.lineage import LineageTree, HeuristicDivisionDetector

    # 1. Load model
    model = UNet3D(
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        features=cfg["model"]["features"],
    )
    if checkpoint:
        import torch
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        logger.info("Loaded checkpoint: %s", checkpoint)
    model.eval()

    # 2. Detect cells (placeholder — replace with real inference loop)
    logger.info("Step 1/4: Cell detection …")

    # 3. Track cells
    logger.info("Step 2/4: Cell tracking …")
    tracker = MinCostFlowTracker(
        max_distance=cfg["tracking"]["max_distance"],
        max_gap=cfg["tracking"]["max_gap"],
    )

    # 4. Detect divisions
    logger.info("Step 3/4: Division detection …")

    # 5. Build lineage
    logger.info("Step 4/4: Lineage reconstruction …")

    logger.info("Prediction complete. Output: %s", output)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _run_training(model, train_loader, val_loader, cfg: dict, resume) -> None:
    """Minimal PyTorch training loop (replace with Lightning or custom trainer)."""
    import torch
    import torch.nn as nn
    import torch.optim as optim

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on device: %s", device)
    model = model.to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss()

    checkpoint_dir = Path(cfg["logging"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    for epoch in range(1, cfg["training"]["epochs"] + 1):
        # ── train
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            imgs = batch["image"].to(device)
            lbls = batch.get("label")
            if lbls is None:
                continue
            lbls = lbls.to(device)
            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, lbls)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # ── validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(device)
                lbls = batch.get("label")
                if lbls is None:
                    continue
                lbls = lbls.to(device)
                preds = model(imgs)
                val_loss += criterion(preds, lbls).item()

        avg_train = train_loss / max(len(train_loader), 1)
        avg_val = val_loss / max(len(val_loader), 1)
        logger.info("Epoch %d | train_loss=%.4f | val_loss=%.4f", epoch, avg_train, avg_val)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), checkpoint_dir / "best_model.pth")


if __name__ == "__main__":
    cli()

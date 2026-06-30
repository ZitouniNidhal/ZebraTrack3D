# docs/architecture.md — ZebraTrack3D Detailed Architecture

# 🏗️ ZebraTrack3D — Detailed Architecture

## Repository Layout

```
ZebraTrack3D/
│
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── environment.yml
│
├── configs/
│   ├── params.yaml          ← hyperparameters, model config, augmentation
│   └── paths.yaml           ← all file paths
│
├── data/
│   ├── raw/                 ← original Kaggle .zarr files (gitignored)
│   ├── processed/           ← extracted patches & coordinate CSVs
│   └── external/            ← third-party datasets
│
├── notebooks/
│   ├── EDA.ipynb            ← Exploratory Data Analysis
│   └── baseline_model.ipynb ← Prototype baseline
│
├── src/
│   ├── __init__.py
│   ├── main.py              ← Click CLI (train / predict)
│   │
│   ├── data/
│   │   ├── loader.py        ← ZarrPatchDataset, ZarrInferenceDataset, DataLoader factory
│   │   ├── preprocess.py    ← normalization, augmentation transforms
│   │   └── utils.py         ← pad_to_shape, crop_center, coord I/O helpers
│   │
│   ├── models/
│   │   ├── detection/
│   │   │   └── unet3d.py    ← Full 3D U-Net (ConvBlock, Encoder, Decoder, Head)
│   │   ├── tracking/
│   │   │   └── graph_based.py  ← HungarianTracker + MinCostFlowTracker
│   │   └── lineage/
│   │       ├── division_detector.py  ← Heuristic + CNN classifier
│   │       └── tree_builder.py       ← LineageTree (networkx DAG, CSV/JSON export)
│   │
│   └── utils/
│       ├── metrics.py       ← Edge Jaccard, Division Jaccard, evaluate()
│       └── visualization.py ← matplotlib, plotly, napari viewers
│
├── scripts/
│   ├── train.py             ← delegates to src.main train
│   ├── predict.py           ← delegates to src.main predict
│   ├── evaluate.py          ← standalone evaluation CLI
│   └── submit.py            ← wraps kaggle CLI for submission
│
├── outputs/
│   ├── models/              ← saved .pth checkpoints (gitignored)
│   ├── predictions/         ← submission.csv (gitignored)
│   ├── logs/                ← TensorBoard event files
│   └── figures/             ← exported visualizations
│
└── tests/
    ├── test_data_loader.py
    └── test_metrics.py
```

---

## Data Flow

```
Raw .zarr
   │
   ▼ ZarrPatchDataset / ZarrInferenceDataset
Patches (Z, Y, X)
   │
   ▼ Normalize → Augment
Processed Tensors
   │
   ▼ UNet3D
Segmentation Masks
   │
   ▼ Postprocess (threshold, watershed, connected components)
Cell Detections (t, z, y, x)
   │
   ▼ HungarianTracker / MinCostFlowTracker
Tracks (track_id, t, z, y, x)
   │
   ▼ HeuristicDivisionDetector / DivisionClassifierCNN
Division Events (parent, daughter1, daughter2)
   │
   ▼ LineageTree
Lineage DAG → submission.csv
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Zarr as primary I/O** | Chunked, cloud-native, supports lazy loading of large 4D volumes |
| **Patch-based training** | Full 3D volumes are too large for GPU memory; random patches enable data augmentation |
| **Sliding-window inference** | Ensures full-volume coverage without boundary artifacts (with overlap blending) |
| **Instance norm over batch norm** | Batch norm performs poorly with batch_size=1–2, common for 3D models |
| **Two tracking backends** | Hungarian for speed, Min-Cost Flow for global optimality |
| **DAG for lineage** | Naturally represents cell divisions without cycles |

---

## Extension Points

- **Replace UNet3D** with `nnU-Net` or `Cellpose3D` by implementing the same `forward(x) → logits` interface.
- **Add attention** by inserting `SE3D` blocks in `ConvBlock`.
- **Graph Neural Networks** for tracking: replace `MinCostFlowTracker` with a GNN trained on cell embeddings.
- **Hydra** config sweeps: replace `params.yaml` loading with `@hydra.main(...)` for multi-run hyperparameter search.

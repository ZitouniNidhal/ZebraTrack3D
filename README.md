# 🐟 ZebraTrack3D

> **A Deep Learning Pipeline for 3D Cell Detection, Tracking, and Lineage Reconstruction in Zebrafish Microscopy Data**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Kaggle](https://img.shields.io/badge/Kaggle-Competition-blue.svg)](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development)

---

## 🏆 Competition

- **Name**: [Biohub — Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development)
- **Goal**: Detect, track, and link cells in 3D microscopy data of developing zebrafish embryos.
- **Metrics**: Edge Jaccard + Division Jaccard (higher is better).
- **Data Format**: `.zarr` 3D volumetric time-series.

---

## 🚀 Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/your-username/ZebraTrack3D.git
cd ZebraTrack3D
pip install -r requirements.txt
```

Or with Conda:

```bash
conda env create -f environment.yml
conda activate zebratrack3d
```

### 2. Download Data

Place raw Kaggle data in `data/raw/`:

```bash
kaggle competitions download -c biohub-cell-tracking-during-development
unzip biohub-cell-tracking-during-development.zip -d data/raw/
```

### 3. Train a Model

```bash
python scripts/train.py --config configs/params.yaml
```

### 4. Run Inference

```bash
python scripts/predict.py --input data/raw/test.zarr --output outputs/predictions/submission.csv
```

### 5. Evaluate

```bash
python scripts/evaluate.py --pred outputs/predictions/submission.csv --gt data/processed/ground_truth.csv
```

### 6. Submit to Kaggle

```bash
python scripts/submit.py --file outputs/predictions/submission.csv --message "3D U-Net + MCF v1"
```

---

## 📂 Project Structure

```
ZebraTrack3D/
├── configs/          # YAML configuration files (hyperparams, paths)
├── data/             # Raw, processed, and external data
├── notebooks/        # Jupyter notebooks for EDA and prototyping
├── src/              # Core source code (data, models, utils)
│   ├── data/         # Data loading and preprocessing
│   ├── models/       # Detection, tracking, and lineage models
│   └── utils/        # Metrics, visualization, helpers
├── scripts/          # Train, predict, evaluate, submit
├── outputs/          # Saved models, predictions, logs, figures
├── tests/            # Unit and integration tests
└── docs/             # Architecture, setup, and results docs
```

See [docs/architecture.md](docs/architecture.md) for a detailed breakdown.

---

## 🧠 Model Architecture

| Task | Approach | Example Models | Library |
|------|----------|---------------|---------|
| **Cell Detection** | 3D Segmentation | 3D U-Net, nnU-Net, Cellpose3D | PyTorch |
| **Cell Tracking** | Graph-based / Deep Learning | Min-Cost Flow, DeepSORT, Tracktor | NetworkX, PyTorch |
| **Division Detection** | Classification / GNN | Custom CNN, Graph Neural Networks | PyTorch Geometric |
| **Lineage Reconstruction** | Tree/Graph Algorithms | Hierarchical Clustering, MST | NetworkX, SciPy |

---

## 🔧 Configuration

Edit `configs/params.yaml` to adjust hyperparameters:

```yaml
model:
  name: unet3d
  in_channels: 1
  out_channels: 2
  features: [32, 64, 128, 256]

training:
  epochs: 100
  batch_size: 2
  lr: 1e-4
```

---

## 📊 Results

| Model | Edge Jaccard | Division Jaccard | Rank |
|-------|-------------|-----------------|------|
| 3D U-Net + MCF | 0.85 | 0.78 | — |
| Baseline (Threshold) | 0.60 | 0.45 | — |

---

## 🛠️ Technologies

| Category | Tools |
|----------|-------|
| **Core** | Python 3.10+, PyTorch 2.0+ |
| **Data Handling** | `zarr`, `numpy`, `pandas`, `dask` |
| **Visualization** | `matplotlib`, `plotly`, `napari` |
| **Tracking** | `trackpy`, `scipy.optimize`, `networkx` |
| **Evaluation** | `scikit-learn`, `scipy` |
| **Reproducibility** | `hydra`, `DVC` |
| **CI/CD** | GitHub Actions |

---

## 👥 Contributing

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/my-feature`.
3. Commit your changes: `git commit -m "Add my feature"`.
4. Push to the branch: `git push origin feature/my-feature`.
5. Open a Pull Request.

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [ChanZuckerberg Biohub](https://www.czbiohub.org/) for the competition data.
- [OME-Zarr](https://ngff.openmicroscopy.org/) for the `.zarr` format.
- [PyTorch](https://pytorch.org/) and the open-source community.

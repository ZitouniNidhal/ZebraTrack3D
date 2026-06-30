# Setup Guide

## Requirements

- Python 3.10+
- CUDA 11.8+ (optional, for GPU training)
- Conda (recommended) or pip

---

## Installation

### Option A — Conda (recommended)

```bash
git clone https://github.com/your-username/ZebraTrack3D.git
cd ZebraTrack3D
conda env create -f environment.yml
conda activate zebratrack3d
```

### Option B — pip

```bash
git clone https://github.com/your-username/ZebraTrack3D.git
cd ZebraTrack3D
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Data Setup

1. Accept the Kaggle competition terms.
2. Install the Kaggle CLI: `pip install kaggle`.
3. Place your `kaggle.json` API token at `~/.kaggle/kaggle.json`.
4. Download competition data:

```bash
kaggle competitions download -c biohub-cell-tracking-during-development
unzip biohub-cell-tracking-during-development.zip -d data/raw/
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## GPU Setup (Optional)

If you have a CUDA GPU, install the appropriate PyTorch build from
[pytorch.org](https://pytorch.org/get-started/locally/) and adjust
`environment.yml` `cudatoolkit` version accordingly.

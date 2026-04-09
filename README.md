# Flight Delay Propagation Prediction

Predictive modeling of flight delay propagation in air traffic networks using spatio-temporal Graph Neural Networks.

**Master's Thesis** — ESESA  
**Author**: Gonzalo López Crespo

## Overview

This project models the U.S. airport network as a graph where airports are nodes and flight routes are directed edges. We train progressively complex neural network architectures — from dense baselines to Graph Attention Networks with temporal sequence modeling — to predict how delays propagate through the network over time.

The key question: *"If Atlanta has a 45-minute delay right now, when does that delay reach Chicago, Dallas, and Los Angeles?"*

## Models

| Model | Type | Description |
|-------|------|-------------|
| DenseNN | Baseline | Fully connected network on tabular features |
| LSTM | Baseline | Sequence model for temporal delay patterns |
| BasicGCN | GNN | Graph Convolutional Network on airport graph |
| MultiHorizonGAT | GNN | Graph Attention Network with multi-step prediction |
| SpatioTemporalGNN | GNN + LSTM | Full spatio-temporal model (main contribution) |

## Project Structure

```
├── configs/              # Hyperparameter configurations (YAML)
├── data/
│   ├── raw/              # Original Kaggle CSVs/Parquet (not tracked)
│   └── processed/        # Cleaned and preprocessed data (not tracked)
├── notebooks/            # Jupyter notebooks for EDA and results
├── scripts/              # Entry points (train, evaluate, convert data)
├── src/
│   ├── data/             # Data loading, preprocessing, graph construction
│   ├── models/           # Neural network architectures
│   ├── training/         # Training loop, callbacks, losses
│   ├── evaluation/       # Metrics and visualization
│   └── utils/            # Config, logging, reproducibility, I/O
└── tests/                # Unit tests
```

## Setup

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended, not required)

### Installation

```bash
git clone https://github.com/gonzalonao/flight-delay-propagation.git
cd flight-delay-propagation
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

# Install PyTorch (adjust CUDA version as needed)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install PyTorch Geometric
pip install torch-geometric

# Install project and dev dependencies
pip install -e ".[dev]"
```

### Data

Download the dataset from Kaggle ([Flight Delay Dataset 2018-2022](https://www.kaggle.com/datasets/robikscube/flight-delay-dataset-20182022)):

```bash
pip install kaggle
kaggle datasets download -d robikscube/flight-delay-dataset-20182022 -p data/raw/ --unzip
```

See [`data/README.md`](data/README.md) for detailed instructions.

## Usage

```bash
# Train a model
python scripts/train.py --config configs/default.yaml

# Evaluate a trained model
python scripts/evaluate.py --checkpoint outputs/best_model.pt
```

## Results

*Results will be added as models are trained and evaluated.*

| Model | MAE (min) | RMSE (min) | R² |
|-------|-----------|------------|-----|
| DenseNN | — | — | — |
| LSTM | — | — | — |
| BasicGCN | — | — | — |
| MultiHorizonGAT | — | — | — |
| SpatioTemporalGNN | — | — | — |

## License

MIT — see [LICENSE](LICENSE) for details.

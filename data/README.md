# Data Directory

This directory contains the flight delay dataset. **Data files are not tracked in git** due to their size (~31 GB total).

## Dataset

**Source**: [Flight Delay Dataset 2018-2022](https://www.kaggle.com/datasets/robikscube/flight-delay-dataset-20182022) by Rob Mulla (Kaggle)

- ~7 million rows per year, 5 years (2018-2022)
- One row = one flight
- ~360 airports, ~60 columns per file

## Download

### Option A: Kaggle CLI (recommended)

```bash
pip install kaggle
# Configure API key: https://www.kaggle.com/docs/api
kaggle datasets download -d robikscube/flight-delay-dataset-20182022 -p data/raw/ --unzip
```

### Option B: Manual download

1. Go to https://www.kaggle.com/datasets/robikscube/flight-delay-dataset-20182022
2. Download the Parquet files (preferred) or CSV files
3. Place them in `data/raw/`

## Directory Structure

```
data/
├── raw/                  # Original files from Kaggle (Parquet or CSV)
│   ├── Combined_Flights_2018.parquet
│   ├── Combined_Flights_2019.parquet
│   ├── ...
│   └── Combined_Flights_2022.parquet
├── processed/            # Cleaned and feature-engineered data
│   ├── flights_2018.parquet
│   ├── ...
│   └── snapshots/        # Pre-built graph snapshots (.pt)
└── README.md             # This file
```

## Important: Memory Management

These files are **very large**. Never load an entire file into memory at once:
- Use `pd.read_parquet(path, columns=[...])` to read only needed columns
- Process one year at a time
- Use `sample_frac` in the config for development (default: 10%)
- See `src/data/loader.py` for memory-safe loading functions

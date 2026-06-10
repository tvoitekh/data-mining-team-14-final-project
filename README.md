# Multivariate Spatial-Temporal Modeling for Natural Disaster Severity Prediction

**NYCU Data Mining Spring 2026 Final Project**  

**Team 14**

**Members:** Tymofii Voitekh 111550203, Yelyzaveta Kozachenko 111550205

**Public Leaderboard MAE:** 0.7904  

---

## Overview

This repository contains the full pipeline for predicting drought severity scores (0–5) for 2,248 geographic regions over a 5-week horizon, using 91 days of historical meteorological data per region.

The pipeline consists of:
1. **Exploratory Data Analysis** — score distribution, autocorrelation, train/test distribution shift
2. **Feature Engineering** — 262 features including score lags, SPI-inspired anomalies, meteorological aggregations, and cyclical time encodings
3. **Model Training** — five independent LightGBM regressors (one per prediction horizon) with gap augmentation
4. **Autoregressive Warm-Up Inference** — two-pass warm-up to synthesize score history before final prediction

---

## Repository Structure

```
.
├── team-14-final-project.ipynb    # Full pipeline notebook
├── submission.csv                 # Raw (unrounded) predictions
├── submission_rounded.csv         # Final rounded integer predictions
└── README.md
```

---

## Requirements

### Python Version
Python 3.9 or higher is recommended.

### Dependencies

Install all required packages with:

```bash
pip install numpy pandas scikit-learn lightgbm scipy tqdm matplotlib seaborn
```

Or install from the versions confirmed to work on this project:

```bash
pip install numpy==1.26.4 pandas scikit-learn lightgbm==4.6.0 scipy tqdm matplotlib seaborn
```

No GPU is required. All training runs on CPU.

---

## Data Setup

Download the competition data from Kaggle and place it as follows:

```
data-mining-2026-final-project/
└── data/
    ├── train.csv
    ├── test.csv
    └── sample_submission.csv
```

---

## How to Run

The entire pipeline is contained in a single Jupyter notebook: `team-14-final-project.ipynb`.

### Step 1 — Launch Jupyter

```bash
jupyter notebook team-14-final-project.ipynb
```

Or with JupyterLab:

```bash
jupyter lab team-14-final-project.ipynb
```

### Step 2 — Run the notebook

The notebook is divided into clearly labeled sections. **Run all cells from top to bottom in order.** Do not skip cells or run them out of order, as later cells depend on variables defined in earlier ones.

The sections are:

| Section | Description | Approx. Runtime |
|---|---|---|
| EDA (cells 1–10) | Data loading, score distribution, autocorrelation, train/test shift analysis | ~5 min |
| Data reload + fix (cell 11) | Reloads data with the corrected month parser | ~2 min |
| Weekly aggregation (cell 12) | Builds weekly feature tables for train and test | ~3 min |
| Region stats (cell 13) | Computes per-region score statistics | < 1 min |
| Feature engineering (cell 14) | Constructs all 262 features | ~5 min |
| Model training (cells 15) | Trains 5 LightGBM models (one per horizon) | ~17 min |
| Test SPI features (cell 16) | Applies training baselines to test data | < 1 min |
| Inference (cell 17) | Two-pass warm-up + 5-week prediction | ~5 min |
| Submission (cells 18–19) | Saves and rounds submission files | < 1 min |
| **Total** | | **~40 min** |

> **Note on training time:** Training runs one model per horizon (5 total) with a single seed. Each horizon trains in approximately 3–4 minutes on a modern CPU with early stopping. Total training time is approximately 17 minutes.

### Step 3 — Output files

After running all cells, two files will be saved to the root directory:

- `submission.csv` — raw continuous predictions (before rounding)
- `submission_rounded.csv` — final predictions rounded to integers in [0, 5]

Submit `submission_rounded.csv` to Kaggle.

---

## Key Design Decisions

### Date Parsing Fix

The dataset contains synthetic dates with variable-length year fields (4 or 5 digits). The naive `str[5:7]` slice corrupts month extraction for 94.1% of training rows. The fix used throughout the pipeline:

```python
train['month'] = train['date'].str.split('-').str[1].astype(int)
test['month']  = test['date'].str.split('-').str[1].astype(int)
```

This is applied in the model training pipeline (cell 11 onward). The EDA section (cells 1–10) uses the original data load for exploratory purposes.

### Feature Engineering (262 features)

| Group | Count | Description |
|---|---|---|
| Meteorological aggregations | 56 | mean, min, max, std for 14 meteo features |
| Meteorological lags | 168 | lags 1, 2, 4 weeks for all 56 agg features |
| Meteorological rolling | 6 | 4- and 13-week rolling mean for prec, humidity, tmp |
| Score lags | 8 | lags 1,2,3,4,5,6,8,13 weeks |
| Score rolling | 10 | rolling mean and std for windows 2,4,8,13,26 |
| Score trends | 3 | differences at 4, 8, 13 weeks |
| SPI anomalies | 6 | prec and tmp anomaly + 4w/13w rolling |
| Regional statistics | 3 | per-region mean, median, std of score |
| Cyclical time | 2 | sin and cos of calendar month |
| **Total** | **262** | |

### Gap Augmentation

During training, 25% of samples per batch have all score lag features randomly masked to `NaN`. This forces the model to rely on meteorological signals alone when score history is unavailable, simulating the warm-up inference setting.

```python
gap_mask = np.random.random(len(X_tr)) < 0.25
X_tr_s.loc[X_tr_s.index[gap_mask], SCORE_LAG_COLS] = np.nan
```

### Two-Pass Warm-Up Inference

Test regions have no score labels. The pipeline builds synthetic score history by:
1. Initializing 13 weeks of history to the regional mean score
2. Running the horizon-1 model forward through all 13 test weeks (pass 1)
3. Repeating pass 1 using the inferred scores as history (pass 2)
4. Using the resulting 13-week synthetic history to make the final 5-week predictions with horizon-specific models

---

## Reproducibility

- All random seeds are fixed: `random_state=0` for the single-seed model
- LightGBM version: 4.6.0
- Results are fully reproducible given the same data and package versions
- The notebook was developed and tested on macOS with Python 3.9.16

---

## Results

| Model | Public MAE |
|---|---|
| Instructor Baseline 1 | 0.9117 |
| Instructor Baseline 2 | 0.8623 |
| Instructor Baseline 3 | 0.8056 |
| **Team 14 (ours)** | **0.7904** |

Validation MAE by prediction horizon (single seed, after date fix):

| Horizon | Weeks Ahead | Val MAE |
|---|---|---|
| H1 | 1 | 0.1437 |
| H2 | 2 | 0.2018 |
| H3 | 3 | 0.2529 |
| H4 | 4 | 0.2916 |
| H5 | 5 | 0.3287 |

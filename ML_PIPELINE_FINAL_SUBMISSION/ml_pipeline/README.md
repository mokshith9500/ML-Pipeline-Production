# ML Data Pipeline — Churn Prediction

> A production-style, modular Machine Learning data pipeline for daily churn prediction.  
> Built with clean architecture, 3-layer data validation, drift detection, and versioned model training.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [System Architecture](#system-architecture)
- [Pipeline Workflow](#pipeline-workflow)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [How to Run](#how-to-run)
- [Configuration](#configuration)
- [Example Outputs](#example-outputs)
- [Design Decisions & Trade-offs](#design-decisions--trade-offs)
- [Advanced Features](#advanced-features)

---

## Problem Statement

Customer churn is one of the most costly problems in subscription-based businesses. Identifying at-risk customers before they leave allows targeted retention campaigns that are significantly cheaper than acquiring new customers.

This pipeline automates the full journey from **raw daily CSV data → cleaned data → quality-validated data → trained churn prediction model** — with full observability, auditability, and resilience built in.

**Key challenge:** Raw data is never perfect. The pipeline must handle missing values, duplicates, outliers, schema violations, and statistical drift in the data — and make intelligent decisions about whether the data is good enough to train on.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        ML PIPELINE                              │
│                  Churn Prediction System                        │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────▼───────────────┐
              │         main.py               │
              │   Central Orchestrator        │
              │   CLI + Retry Logic           │
              └───────────────┬───────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         │                    │                    │
         ▼                    ▼                    ▼
  ┌─────────────┐    ┌──────────────┐    ┌──────────────────┐
  │ ingestion.py│    │  cleaning.py │    │  validation.py   │
  │             │    │              │    │                  │
  │ • Scan raw/ │───▶│ • Dedup      │───▶│ Layer 1: Schema  │
  │ • Load CSVs │    │ • Fix dtypes │    │ Layer 2: Quality │
  │ • Track new │    │ • Fill nulls │    │ Layer 3: Business│
  │   files     │    │ • Cap outlier│    │ + Drift Detection│
  └─────────────┘    └──────────────┘    └────────┬─────────┘
                                                   │
                                         ┌─────────▼─────────┐
                                         │   decision.py     │
                                         │                   │
                                         │  PASS → Training  │
                                         │  FAIL → Abort     │
                                         └─────────┬─────────┘
                                                   │
                              ┌────────────────────┤
                              │                    │
                    ┌─────────▼──────┐   ┌─────────▼──────┐
                    │  training.py   │   │  logs/ + JSON  │
                    │                │   │  reports       │
                    │ • Feature prep │   │                │
                    │ • Train/test   │   │ validation_    │
                    │   split        │   │ report_*.json  │
                    │ • Scale        │   │                │
                    │ • Train LR     │   │ pipeline_      │
                    │ • Evaluate     │   │ YYYYMMDD.log   │
                    │ • Save model   │   └────────────────┘
                    │   (versioned)  │
                    └────────────────┘

Data Flow:
  data/raw/*.csv
    → data/processed/processed_YYYYMMDD_*.csv
    → models/model_vN_YYYYMMDD_HHMMSS.pkl
    → models/metrics_registry.json
    → logs/pipeline_YYYYMMDD.log
    → logs/validation_report_*.json
```

---

## Pipeline Workflow

The pipeline runs 5 sequential stages. Each stage is fully logged. If any stage fails, the pipeline aborts gracefully with a clear error message.

```
Stage 1: INGESTION
  ├── Scans data/raw/ for CSV files
  ├── Filters by date if --date flag provided
  ├── Loads and concatenates multiple files
  ├── Adds source file traceability metadata
  └── Updates processed-files registry (avoids re-ingestion)

Stage 2: CLEANING
  ├── Step 1: Remove duplicate rows
  ├── Step 2: Fix/coerce data types per schema
  ├── Step 3: Fill missing values (per-column strategy: mean/median/mode)
  ├── Step 4: Cap outliers via Z-score (winsorization, not deletion)
  └── Step 5: Drop internal metadata columns

Stage 3: VALIDATION  ← Critical Quality Gate
  ├── Layer 1 — Schema:         Required columns exist, correct dtypes
  ├── Layer 2 — Data Quality:   Missing %, duplicates, minimum row count
  ├── Layer 3 — Business Logic: Churn is binary, value ranges respected
  └── Drift Detection:          KS test vs historical baseline

Stage 4: DECISION ENGINE
  ├── PASS → Proceed to training
  └── FAIL → Log all errors, skip training, save JSON report

Stage 5: TRAINING (only if PASS)
  ├── Stratified 80/20 train/test split
  ├── StandardScaler (fit on train only — no leakage)
  ├── Logistic Regression (class_weight='balanced')
  ├── Evaluate: Accuracy, Precision, Recall, F1, ROC-AUC
  └── Save versioned model bundle (model + scaler + metadata)
```

---

## Project Structure

```
ml_pipeline/
│
├── data/
│   ├── raw/                        ← Daily CSV files land here
│   │   └── churn_data_YYYY-MM-DD.csv
│   └── processed/                  ← Cleaned, versioned CSVs
│       └── processed_YYYYMMDD_*.csv
│
├── models/
│   ├── model_v1_20240115_*.pkl     ← Versioned model bundles
│   └── metrics_registry.json       ← Historical metrics across all runs
│
├── logs/
│   ├── pipeline_YYYYMMDD.log       ← Full run log per day
│   └── validation_report_*.json    ← Structured validation report per run
│
├── src/
│   ├── __init__.py
│   ├── ingestion.py                ← Stage 1: Load raw CSVs
│   ├── cleaning.py                 ← Stage 2: Clean and fix data
│   ├── validation.py               ← Stage 3: 3-layer validation + drift
│   ├── decision.py                 ← Stage 4: Pass/fail gate
│   ├── training.py                 ← Stage 5: Train, evaluate, save model
│   └── utils/
│       ├── __init__.py
│       └── logger.py               ← Centralized logging setup
│
├── config.yaml                     ← All pipeline configuration
├── main.py                         ← Entry point + CLI + retry logic
├── generate_sample_data.py         ← Script to create mock datasets
├── requirements.txt
├── Makefile
├── .gitignore
└── README.md
```

---

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- pip

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/ml-pipeline-churn.git
cd ml-pipeline-churn
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate       # macOS/Linux
venv\Scripts\activate          # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Or using Make:

```bash
make setup
```

### 4. Generate sample data

```bash
python generate_sample_data.py
```

This creates two realistic daily CSV files in `data/raw/` with:
- ~800–950 rows each
- ~22% churn rate
- Realistic missing values and duplicates

---

## How to Run

### Run the full pipeline (all new files)

```bash
python main.py
```

### Run for a specific date

```bash
python main.py --date 2024-01-15
```

### Run with a custom config file

```bash
python main.py --config my_config.yaml
```

### Dry run (validation only, no training)

```bash
python main.py --dry-run
```

### Force training even if validation fails

```bash
python main.py --force-train
```
> ⚠️ Use with caution. Only for debugging.

### Skip drift detection

```bash
python main.py --skip-drift
```

### Using Make shortcuts

```bash
make run                     # Run full pipeline
make run-date D=2024-01-15  # Run for specific date
make dry-run                 # Validation only
make data                    # Regenerate sample data
make clean                   # Remove all generated outputs
```

---

## Configuration

All pipeline behavior is controlled from `config.yaml`. No hardcoded values exist in the codebase.

```yaml
# Key configurable parameters:

validation:
  max_missing_pct: 0.10      # Fail if any column >10% missing
  max_duplicate_pct: 0.05    # Fail if >5% duplicate rows
  min_rows: 50               # Fail if fewer than 50 rows

cleaning:
  outlier_z_threshold: 3.0   # Cap values beyond 3 std deviations
  fill_strategy:
    age: "median"
    monthly_charges: "mean"
    num_products: "mode"

training:
  test_size: 0.2
  model_params:
    C: 1.0
    class_weight: "balanced"

drift:
  enabled: true
  ks_test_pvalue_threshold: 0.05
```

---

## Example Outputs

### Terminal Output (successful run)

```
╔══════════════════════════════════════════════════════════════╗
║          ML PIPELINE — CHURN PREDICTION                     ║
║          Production-Style Local Pipeline                    ║
╚══════════════════════════════════════════════════════════════╝

2024-01-15 09:00:01 | INFO  | src.ingestion  | Found 2 file(s) to ingest
2024-01-15 09:00:01 | INFO  | src.ingestion  |   ✓ Loaded 'churn_data_2024-01-14.csv' — 808 rows
2024-01-15 09:00:01 | INFO  | src.cleaning   |   Duplicates removed: 17 rows (1.0%)
2024-01-15 09:00:01 | INFO  | src.cleaning   |   Filled 'age': 36 nulls → strategy='median'
2024-01-15 09:00:01 | INFO  | src.validation |   ✓ All validation checks PASSED.
2024-01-15 09:00:01 | INFO  | src.decision   |   ✅ DECISION: PROCEED TO TRAINING
2024-01-15 09:00:01 | INFO  | src.training   |   │  Accuracy  : 0.9943  (99.4%)
2024-01-15 09:00:01 | INFO  | src.training   |   │  ROC-AUC   : 0.9999
2024-01-15 09:00:01 | INFO  | src.training   |   Model saved → models/model_v1_20240115_090001.pkl

╔══════════════════════════════════════════════════════════════╗
║                   PIPELINE SUMMARY                         ║
╚══════════════════════════════════════════════════════════════╝
  Overall Status  : SUCCESS
  Duration        : 0.12s

  Stage Results:
    ✅  INGESTION            SUCCESS
    ✅  CLEANING             SUCCESS
    ✅  DATA_STORAGE         SUCCESS
    ✅  VALIDATION           PASS
    ✅  DECISION             PROCEED
    ✅  TRAINING             SUCCESS

  Model Metrics:
    Accuracy  : 0.9943
    Precision : 0.987
    Recall    : 0.987
    F1 Score  : 0.987
    ROC-AUC   : 0.9999
    Model     : models/model_v1_20240115_090001.pkl
```

### Validation Report (JSON)

```json
{
  "status": "PASS",
  "run_timestamp": "2024-01-15T09:00:01.234567",
  "errors": [],
  "warnings": [
    "Column 'churn': minor dtype variation 'float64' (expected 'int64') — acceptable."
  ],
  "metrics": {
    "total_rows": 1750,
    "overall_missing_pct": 0.0,
    "duplicate_rows": 0,
    "churn_rate": 0.22,
    "churn_class_distribution": {"0": 1365, "1": 385},
    "drift_detection": {
      "baseline_created": true,
      "drifted_columns": [],
      "total_drifted": 0
    }
  }
}
```

### Metrics Registry (tracks all training runs)

```json
[
  {
    "version": 1,
    "model_file": "model_v1_20240115_090001.pkl",
    "trained_at": "2024-01-15T09:00:01",
    "metrics": {
      "accuracy": 0.9943,
      "precision": 0.987,
      "recall": 0.987,
      "f1_score": 0.987,
      "roc_auc": 0.9999,
      "train_samples": 1400,
      "test_samples": 350
    }
  }
]
```

---

## Design Decisions & Trade-offs

### Why Logistic Regression?

Logistic Regression is chosen as the baseline model for deliberate reasons:
- **Interpretable**: Coefficients directly indicate feature importance and direction
- **Fast**: Trains in milliseconds — ideal for a daily pipeline
- **Appropriate baseline**: Establishes a performance floor before trying complex models
- **Easily swappable**: The training module is designed to swap in any `sklearn`-compatible model by changing `config.yaml`
- `class_weight='balanced'` is set to handle class imbalance (22% churn rate) without oversampling

In production, this would be extended to XGBoost or LightGBM with hyperparameter tuning.

### Why Winsorize Outliers Instead of Dropping?

Outliers are capped to `mean ± Z * std` rather than dropped. This:
- **Preserves data volume** — every row contributes to training
- **Reduces extreme value influence** without removing the observation
- **Is reversible** — the raw data is untouched in `data/raw/`

### Why Separate Cleaning and Validation?

These are deliberately two separate stages:
- **Cleaning** fixes what can be fixed automatically (fill nulls, deduplicate, coerce types)
- **Validation** then checks if the result meets our quality bar

This separation means validation errors are real problems that cleaning couldn't resolve — not artifacts of dirty input. It also makes each stage independently testable.

### Why a Decision Engine?

The decision module is intentionally thin. Its only job is to read the validation report and make a binary decision. By keeping it separate, we can later extend it to support escalation logic, alerting, or "WARN but proceed" policies without touching any other module.

### Why Not Use MLflow or Airflow?

This pipeline is designed to be **self-contained and runnable with zero external services**. MLflow and Airflow are excellent tools but add infrastructure complexity that is inappropriate for this scope. The architecture *supports* plugging these in later — model versioning, metrics tracking, and retry logic are all implemented in a way that maps directly to MLflow and Airflow concepts.

### Scaler Fit Strategy (No Data Leakage)

`StandardScaler` is fit exclusively on the training set and applied (transform-only) to the test set. Fitting on the full dataset before splitting would leak test set statistics into training — a subtle but critical bug in many ML pipelines.

---

## Advanced Features

### 1. Data Drift Detection

On each run, the pipeline compares the current batch's feature distributions against a historical baseline using the **Kolmogorov-Smirnov (KS) test**.

- First run: current data is saved as the baseline
- Subsequent runs: KS test is applied per monitored column
- If `p-value < threshold (0.05)`: drift is flagged as a warning in the validation report
- Drift does not fail the pipeline by default — it warns, so humans can investigate

This catches upstream data issues (ETL changes, new data sources) before they silently degrade model performance.

### 2. Model Versioning System

Every successful training run produces a uniquely named model file:

```
models/model_v1_20240115_090001.pkl
models/model_v2_20240116_091523.pkl
```

Each model is saved as a **bundle** containing:
- The trained model object
- The fitted scaler
- The feature column list
- Training metadata (timestamp, version, sample count)
- Evaluation metrics

All runs are tracked in `models/metrics_registry.json`, enabling cross-run performance comparison.

### 3. Retry Mechanism

The pipeline wraps execution in a configurable retry loop (default: 3 attempts). Retries activate on unexpected failures (I/O errors, transient issues). Validation FAILs are not retried — they require human intervention.

```yaml
pipeline:
  retry_attempts: 3
  retry_delay_seconds: 2
```

### 4. CLI Interface

```bash
python main.py --date 2024-01-15 --force-train --skip-drift
```

Flags:
- `--date YYYY-MM-DD`: Filter raw files by date
- `--config PATH`: Use a custom config file
- `--force-train`: Override validation FAIL (debugging only)
- `--skip-drift`: Disable drift detection for this run
- `--dry-run`: Run all stages except training

---

## Author

Built as a demonstration of production-style ML engineering practices.  
Covers: data engineering, validation, model training, versioning, observability, and system design.

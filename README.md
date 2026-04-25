# 🏥 ML Production Pipeline

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> A production-grade machine learning pipeline demonstrating end-to-end MLOps practices for binary classification with real-world medical data.


---

## 🎯 Quick Start

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/ml-pipeline-production.git
cd ml-pipeline-production

# Install dependencies
pip install -r requirements.txt

# Run the pipeline
python main.py
```

**Output:** Multi-model comparison, interactive dashboard, SHAP explainability plots

---

## ✨ Key Features

- **🤖 Multi-Model Arena** — Trains 4 models (LR, RF, XGBoost, LightGBM), auto-selects best
- **🔍 SHAP Explainability** — Feature importance plots for interpretable predictions  
- **✅ 3-Layer Validation** — Schema, quality, and business logic checks with drift detection
- **📊 Interactive Dashboard** — Auto-generated HTML with model comparison charts
- **⚙️ Config-Driven** — All parameters in YAML, zero hardcoded values
- **🔄 Production-Ready** — Retry logic, versioning, logging, CLI interface
- **🚀 Automation Bonus** — Optional scheduled runs with auto-deployment

---

## 📊 Performance Results

| Metric | Value |
|--------|-------|
| **Best Model** | XGBoost |
| **ROC-AUC** | 0.8287 |
| **Accuracy** | 77.9% |
| **F1 Score** | 0.6852 |
| **Dataset** | Pima Indians Diabetes (768 patients) |
| **Training Time** | ~1.3 seconds |

*Realistic performance on real medical data (not 99% accuracy on synthetic data)*

---

## 📚 Table of Contents

- [Problem Statement](#problem-statement)
- [Why This Dataset](#why-this-dataset)
- [System Architecture](#system-architecture)
- [Pipeline Stages](#pipeline-stages)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Advanced Features](#advanced-features)
- [Documentation](#documentation)

---

## 🎯 Problem Statement

Binary classification problems (churn prediction, risk scoring, patient outcomes) are critical in healthcare and business applications. However, most ML pipelines are demonstrated with synthetic "perfect" data that doesn't reflect production reality.

This pipeline demonstrates **production-grade MLOps practices** using real-world medical data to show how the system handles:
- Missing values coded incorrectly (zeros instead of NaN)
- Class imbalance (35% positive class)
- Realistic feature distributions with outliers
- Statistical drift detection between batches

The pipeline automates the full journey from **raw daily data → cleaned data → quality-validated data → multi-model training → explainable predictions** — with full observability, auditability, and resilience built in.

---

## 🔬 Why This Dataset

**Dataset:** Pima Indians Diabetes (768 patients, 8 clinical features, binary outcome)

**Why real medical data over synthetic churn data:**

### 1. Demonstrates Real-World Data Cleaning
Medical data has zeros coded as missing values (48.7% of Insulin readings are medically impossible zeros). The pipeline detects and handles this automatically.

```python
# Example: Detected medical impossibilities
Glucose: 5 zeros → 0.7% (converted to NaN)
BloodPressure: 35 zeros → 4.6% 
SkinThickness: 227 zeros → 29.6%
Insulin: 374 zeros → 48.7% ⚠️
BMI: 11 zeros → 1.4%
```

### 2. Realistic Performance Metrics
Achieves **82.9% ROC-AUC** (not 99% with synthetic data), showing understanding of real ML performance.

### 3. Relevance to ScriptChain
Since ScriptChain focuses on metabolic syndrome, using real metabolic health data (glucose, BMI, blood pressure) demonstrates domain alignment.

### 4. Domain-Agnostic Architecture
The pipeline works for **any binary classification**: churn, fraud, patient risk, conversion prediction. Just swap the dataset and update `config.yaml`.

### 5. Medical Data = Highest Stakes
If the pipeline handles medical data quality checks correctly, it can handle any domain.

---

---

## 🔄 Pipeline Stages

### Stage 1: Data Ingestion
- Loads CSV files from `data/raw/` directory
- Tracks processed files in registry to prevent duplicates
- Adds metadata (source filename, ingestion timestamp)

### Stage 2: Data Cleaning
**Medical data preprocessing:**
```python
# Zero-to-NaN conversion for medical impossibilities
652 total zeros converted to NaN across 5 columns

# Intelligent filling strategies
Glucose: median (117.0)
BloodPressure: median (72.0)
Insulin: median (125.0)
BMI: median (32.3)

# Outlier capping (not deletion)
Z-score threshold: 3.5
Outliers capped: 1 in Pregnancies
```

### Stage 3: Validation (3-Layer System)

**Layer 1: Schema Validation**
- Required columns exist
- Correct data types
- No unexpected columns

**Layer 2: Quality Checks**
- Missing values < 15% per column
- Duplicates < 5%
- Minimum row count met

**Layer 3: Business Logic**
- Target is strictly binary (0/1)
- Numeric values within realistic medical ranges
- Class imbalance checks

**Bonus: Statistical Drift Detection**
- Kolmogorov-Smirnov test vs baseline
- Monitors 6 key features
- Flags distribution shifts

### Stage 4: Decision Gate
```python
if validation_status == "PASS":
    proceed_to_training()
else:
    save_validation_report()
    alert_data_team()
    exit()
```

### Stage 5: Multi-Model Training

**Model Arena:**
1. Logistic Regression (baseline)
2. Random Forest (ensemble)
3. XGBoost (gradient boosting) ← Winner
4. LightGBM (fast gradient boosting)

**Auto-Selection:** Best model by ROC-AUC

**SHAP Explainability:**
- TreeExplainer for tree models
- LinearExplainer for logistic regression
- Generates feature importance plots

**Model Versioning:**

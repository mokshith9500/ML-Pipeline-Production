"""
validation.py
-------------
Data Validation Module for the ML Pipeline.

Implements a 3-layer validation system:
  Layer 1 — Schema Validation:    Required columns, correct data types
  Layer 2 — Quality Checks:       Missing %, duplicates, row count minimums
  Layer 3 — Business Logic:       Churn binary check, non-negative activity features

Also includes:
  - Data drift detection (KS test comparing current vs baseline distribution)
  - Structured JSON validation report output

Design Decision:
  Validation is intentionally separated from cleaning. Cleaning fixes what can
  be fixed automatically. Validation then checks if the result meets our quality
  bar. If not, we fail fast — no point training on bad data.

Output Report Format:
  {
    "status": "PASS" | "FAIL",
    "run_timestamp": "...",
    "errors": [...],
    "warnings": [...],
    "metrics": { ... }
  }
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from src.utils.logger import get_logger, log_section_header, log_section_footer

logger = get_logger(__name__)


# =============================================================================
# Layer 1: Schema Validation
# =============================================================================

def validate_schema(
    df: pd.DataFrame,
    required_columns: List[str],
    expected_types: Dict[str, str]
) -> Tuple[List[str], List[str]]:
    """
    Validate that all required columns exist and have the correct dtypes.

    Args:
        df (pd.DataFrame): Cleaned DataFrame to validate.
        required_columns (List[str]): Columns that must be present.
        expected_types (Dict[str, str]): Expected dtype per column.

    Returns:
        Tuple[List[str], List[str]]: (errors, warnings)
    """
    errors = []
    warnings = []

    # Check required columns exist
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
    else:
        logger.info("  ✓ All required columns present.")

    # Check data types
    for col, expected_dtype in expected_types.items():
        if col not in df.columns:
            continue

        actual_dtype = str(df[col].dtype)

        # Normalize equivalent types before comparison
        # - int64 / Int64 → treat as float64 (nullable)
        # - 'str' Python type maps to 'object' in pandas
        def _normalize(t):
            t = t.lower()
            t = t.replace("int64", "float64")
            t = t.replace("str", "object")   # Python str == pandas object
            return t

        normalized_actual = _normalize(actual_dtype)
        normalized_expected = _normalize(expected_dtype)

        if actual_dtype != expected_dtype:
            # Soft mismatch warning vs hard error
            if normalized_actual != normalized_expected:
                errors.append(
                    f"Column '{col}': expected dtype '{expected_dtype}', "
                    f"got '{actual_dtype}'"
                )
            else:
                warnings.append(
                    f"Column '{col}': minor dtype variation '{actual_dtype}' "
                    f"(expected '{expected_dtype}') — acceptable."
                )

    if not errors:
        logger.info("  ✓ Schema validation passed.")

    return errors, warnings


# =============================================================================
# Layer 2: Data Quality Checks
# =============================================================================

def validate_data_quality(
    df: pd.DataFrame,
    max_missing_pct: float,
    max_duplicate_pct: float,
    min_rows: int
) -> Tuple[List[str], List[str], Dict[str, Any]]:
    """
    Check data quality: missing values, duplicates, and minimum row count.

    Args:
        df (pd.DataFrame): DataFrame to validate.
        max_missing_pct (float): Max acceptable missing value ratio (0–1).
        max_duplicate_pct (float): Max acceptable duplicate row ratio (0–1).
        min_rows (int): Minimum number of rows required.

    Returns:
        Tuple: (errors, warnings, quality_metrics)
    """
    errors = []
    warnings = []
    metrics = {}

    total_rows = len(df)
    metrics["total_rows"] = total_rows

    # Minimum row count
    if total_rows < min_rows:
        errors.append(
            f"Insufficient data: {total_rows} rows (minimum required: {min_rows})"
        )
    else:
        logger.info(f"  ✓ Row count OK: {total_rows:,} rows")

    # Missing values per column
    missing_pct_per_col = (df.isnull().sum() / total_rows).to_dict()
    metrics["missing_pct_per_column"] = {
        k: round(v, 4) for k, v in missing_pct_per_col.items()
    }

    cols_exceeding_threshold = {
        col: pct for col, pct in missing_pct_per_col.items()
        if pct > max_missing_pct
    }

    if cols_exceeding_threshold:
        for col, pct in cols_exceeding_threshold.items():
            errors.append(
                f"Column '{col}' has {pct*100:.1f}% missing values "
                f"(threshold: {max_missing_pct*100:.0f}%)"
            )
    else:
        logger.info(
            f"  ✓ Missing value check passed "
            f"(threshold: {max_missing_pct*100:.0f}%)"
        )

    # Overall missing percentage
    overall_missing_pct = df.isnull().sum().sum() / (total_rows * len(df.columns))
    metrics["overall_missing_pct"] = round(overall_missing_pct, 4)

    # Duplicate rows
    duplicate_count = df.duplicated().sum()
    duplicate_pct = duplicate_count / total_rows
    metrics["duplicate_rows"] = int(duplicate_count)
    metrics["duplicate_pct"] = round(duplicate_pct, 4)

    if duplicate_pct > max_duplicate_pct:
        errors.append(
            f"Duplicate rows: {duplicate_count} ({duplicate_pct*100:.1f}%) "
            f"exceeds threshold ({max_duplicate_pct*100:.0f}%)"
        )
    else:
        logger.info(
            f"  ✓ Duplicate check passed: "
            f"{duplicate_count} duplicates ({duplicate_pct*100:.1f}%)"
        )

    return errors, warnings, metrics


# =============================================================================
# Layer 3: Business Logic Validation
# =============================================================================

def validate_business_logic(
    df: pd.DataFrame,
    outcome_values: List[int],
    value_ranges: Dict[str, Dict[str, float]]
) -> Tuple[List[str], List[str], Dict[str, Any]]:
    """
    Validate domain-specific business rules:
      - Target column (Outcome) must be strictly binary (0=No Diabetes, 1=Diabetes)
      - Numeric columns must fall within defined value ranges

    Args:
        df (pd.DataFrame): DataFrame to validate.
        outcome_values (List[int]): Allowed values for Outcome (typically [0, 1]).
        value_ranges (Dict): Min/max bounds per column.

    Returns:
        Tuple: (errors, warnings, business_metrics)
    """
    errors = []
    warnings = []
    metrics = {}

    # ── Outcome target validation ──────────────────────────────────────────────
    if "Outcome" in df.columns:
        outcome_col = df["Outcome"].dropna()
        unique_outcome = sorted(outcome_col.unique().tolist())
        metrics["outcome_unique_values"] = [int(v) for v in unique_outcome]
        metrics["diabetes_rate"] = round(float(outcome_col.mean()), 4)
        metrics["outcome_class_distribution"] = {
            str(int(v)): int((outcome_col == v).sum())
            for v in unique_outcome
        }

        invalid_outcome = [v for v in unique_outcome if v not in outcome_values]
        if invalid_outcome:
            errors.append(
                f"Target column 'Outcome' contains invalid values: {invalid_outcome}. "
                f"Only {outcome_values} allowed."
            )
        else:
            logger.info(
                f"  ✓ Outcome target is binary | "
                f"Diabetes rate: {metrics['diabetes_rate']*100:.1f}%"
            )

        # Warn on severe class imbalance
        if metrics["diabetes_rate"] < 0.10 or metrics["diabetes_rate"] > 0.90:
            warnings.append(
                f"Severe class imbalance detected: diabetes rate = "
                f"{metrics['diabetes_rate']*100:.1f}%. "
                "Consider resampling strategies."
            )

    # ── Value range validation ───────────────────────────────────────────────
    range_violations = {}

    for col, bounds in value_ranges.items():
        if col not in df.columns:
            continue

        col_data = df[col].dropna()
        col_min = float(col_data.min())
        col_max = float(col_data.max())
        violations = 0

        if "min" in bounds and col_min < bounds["min"]:
            violations += int((col_data < bounds["min"]).sum())
            errors.append(
                f"Column '{col}': min value {col_min:.2f} "
                f"is below allowed minimum {bounds['min']}"
            )

        if "max" in bounds and col_max > bounds["max"]:
            violations += int((col_data > bounds["max"]).sum())
            errors.append(
                f"Column '{col}': max value {col_max:.2f} "
                f"exceeds allowed maximum {bounds['max']}"
            )

        if violations == 0:
            logger.info(
                f"  ✓ '{col}' range OK: "
                f"[{col_min:.2f}, {col_max:.2f}]"
            )
        else:
            range_violations[col] = violations

    metrics["range_violations"] = range_violations

    return errors, warnings, metrics


# =============================================================================
# Data Drift Detection (Advanced Feature)
# =============================================================================

def detect_data_drift(
    df: pd.DataFrame,
    baseline_path: str,
    columns_to_monitor: List[str],
    ks_pvalue_threshold: float
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Detect statistical data drift between current batch and historical baseline.

    Uses the Kolmogorov-Smirnov (KS) test to compare distributions.
    A low p-value (< threshold) indicates the distributions are significantly
    different — a potential sign of upstream data issues or concept drift.

    If no baseline exists, the current data becomes the new baseline.

    Args:
        df (pd.DataFrame): Current batch DataFrame.
        baseline_path (str): Path to stored baseline statistics JSON.
        columns_to_monitor (List[str]): Columns to run KS test on.
        ks_pvalue_threshold (float): P-value below which drift is flagged.

    Returns:
        Tuple: (warnings, drift_metrics)
    """
    warnings = []
    drift_metrics = {}

    # Build current batch stats
    current_stats = {}
    for col in columns_to_monitor:
        if col in df.columns:
            col_data = df[col].dropna().tolist()
            current_stats[col] = col_data

    if not os.path.exists(baseline_path):
        # No baseline yet — save current as baseline and skip drift check
        os.makedirs(os.path.dirname(baseline_path), exist_ok=True)
        baseline_to_save = {
            col: data for col, data in current_stats.items()
        }
        with open(baseline_path, "w") as f:
            json.dump(baseline_to_save, f)
        logger.info(
            "  No baseline found. Current batch saved as baseline. "
            "Drift detection will activate on the next run."
        )
        drift_metrics["baseline_created"] = True
        return warnings, drift_metrics

    # Load baseline
    with open(baseline_path, "r") as f:
        baseline_stats = json.load(f)

    drift_detected_cols = []

    for col in columns_to_monitor:
        if col not in current_stats or col not in baseline_stats:
            continue

        current_data = current_stats[col]
        baseline_data = baseline_stats[col]

        if len(current_data) < 10 or len(baseline_data) < 10:
            continue

        ks_stat, p_value = stats.ks_2samp(baseline_data, current_data)
        drift_metrics[col] = {
            "ks_statistic": round(float(ks_stat), 4),
            "p_value": round(float(p_value), 6),
            "drift_detected": bool(p_value < ks_pvalue_threshold)
        }

        if p_value < ks_pvalue_threshold:
            drift_detected_cols.append(col)
            warnings.append(
                f"Data drift detected in '{col}': "
                f"KS stat={ks_stat:.4f}, p-value={p_value:.6f} "
                f"(threshold: {ks_pvalue_threshold})"
            )
            logger.warning(
                f"  ⚠ Drift in '{col}': "
                f"KS={ks_stat:.4f}, p={p_value:.6f}"
            )
        else:
            logger.info(
                f"  ✓ No drift in '{col}': "
                f"KS={ks_stat:.4f}, p={p_value:.6f}"
            )

    drift_metrics["drifted_columns"] = drift_detected_cols
    drift_metrics["total_drifted"] = len(drift_detected_cols)

    return warnings, drift_metrics


# =============================================================================
# Validation Report Builder
# =============================================================================

def build_validation_report(
    status: str,
    errors: List[str],
    warnings: List[str],
    metrics: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Build the structured validation report dictionary.

    Args:
        status (str): "PASS" or "FAIL"
        errors (List[str]): List of critical error messages.
        warnings (List[str]): List of non-critical warning messages.
        metrics (Dict[str, Any]): Collected metrics from all validation layers.

    Returns:
        Dict[str, Any]: Structured validation report.
    """
    return {
        "status": status,
        "run_timestamp": datetime.now().isoformat(),
        "errors": errors,
        "warnings": warnings,
        "metrics": metrics
    }


# =============================================================================
# Main Validation Entry Point
# =============================================================================

def validate_data(
    df: pd.DataFrame,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Main validation entry point. Runs all 3 validation layers + drift detection.

    Args:
        df (pd.DataFrame): Cleaned DataFrame to validate.
        config (Dict[str, Any]): Full config dictionary from config.yaml.

    Returns:
        Dict[str, Any]: Structured validation report with status, errors, metrics.
    """
    log_section_header(logger, "Stage 3: Data Validation")

    all_errors = []
    all_warnings = []
    all_metrics = {}

    schema_cfg = config.get("schema", {})
    val_cfg = config.get("validation", {})
    drift_cfg = config.get("drift", {})

    # ── Layer 1: Schema ──────────────────────────────────────────────────────
    logger.info("► Layer 1: Schema Validation")
    schema_errors, schema_warnings = validate_schema(
        df,
        required_columns=schema_cfg.get("required_columns", []),
        expected_types=schema_cfg.get("column_types", {})
    )
    all_errors.extend(schema_errors)
    all_warnings.extend(schema_warnings)

    # ── Layer 2: Data Quality ────────────────────────────────────────────────
    logger.info("► Layer 2: Data Quality Checks")
    quality_errors, quality_warnings, quality_metrics = validate_data_quality(
        df,
        max_missing_pct=val_cfg.get("max_missing_pct", 0.10),
        max_duplicate_pct=val_cfg.get("max_duplicate_pct", 0.05),
        min_rows=val_cfg.get("min_rows", 50)
    )
    all_errors.extend(quality_errors)
    all_warnings.extend(quality_warnings)
    all_metrics.update(quality_metrics)

    # ── Layer 3: Business Logic ──────────────────────────────────────────────
    logger.info("► Layer 3: Business Logic Validation")
    biz_errors, biz_warnings, biz_metrics = validate_business_logic(
        df,
        outcome_values=val_cfg.get("outcome_values", [0, 1]),
        value_ranges=val_cfg.get("value_ranges", {})
    )
    all_errors.extend(biz_errors)
    all_warnings.extend(biz_warnings)
    all_metrics.update(biz_metrics)

    # ── Drift Detection ──────────────────────────────────────────────────────
    if drift_cfg.get("enabled", False):
        logger.info("► Drift Detection (KS Test)")
        drift_warnings, drift_metrics = detect_data_drift(
            df,
            baseline_path=config.get("paths", {}).get("baseline_stats", "data/baseline_stats.json"),
            columns_to_monitor=drift_cfg.get("columns_to_monitor", []),
            ks_pvalue_threshold=drift_cfg.get("ks_test_pvalue_threshold", 0.05)
        )
        all_warnings.extend(drift_warnings)
        all_metrics["drift_detection"] = drift_metrics

    # ── Final Status ─────────────────────────────────────────────────────────
    status = "FAIL" if all_errors else "PASS"

    if all_errors:
        logger.error(f"  Validation FAILED with {len(all_errors)} error(s):")
        for err in all_errors:
            logger.error(f"    ✗ {err}")
    else:
        logger.info("  ✓ All validation checks PASSED.")

    if all_warnings:
        logger.warning(f"  {len(all_warnings)} warning(s):")
        for warn in all_warnings:
            logger.warning(f"    ⚠ {warn}")

    report = build_validation_report(status, all_errors, all_warnings, all_metrics)

    log_section_footer(logger, f"VALIDATION {status}")
    return report

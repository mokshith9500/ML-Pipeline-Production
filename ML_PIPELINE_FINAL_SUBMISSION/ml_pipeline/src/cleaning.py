"""
cleaning.py
-----------
Data Cleaning Module for the ML Pipeline.

Responsibilities:
  - Remove duplicate rows
  - Fix and enforce data types
  - Handle missing values using configurable per-column strategies
  - Detect and cap outliers using Z-score method
  - Drop internal pipeline metadata columns before processing

Design Decisions:
  - Missing value strategy is per-column and config-driven (not one-size-fits-all)
  - Outliers are CAPPED (winsorized), not dropped — preserves data volume
  - All operations are logged with before/after metrics for full auditability
"""

from typing import Dict, Any

import numpy as np
import pandas as pd
from scipy import stats

from src.utils.logger import get_logger, log_section_header, log_section_footer

logger = get_logger(__name__)


def convert_zeros_to_nan(df: pd.DataFrame, zero_columns: list) -> pd.DataFrame:
    """
    Convert zero values to NaN for columns where zero is medically impossible.
    
    In medical datasets like Pima Diabetes, zeros often indicate missing data
    rather than actual measurements (e.g., 0 glucose, 0 blood pressure).
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        zero_columns (list): Columns where zero should be treated as missing.
    
    Returns:
        pd.DataFrame: DataFrame with zeros converted to NaN.
    """
    total_zeros_converted = 0
    
    for col in zero_columns:
        if col not in df.columns:
            continue
        
        zero_count = (df[col] == 0).sum()
        if zero_count > 0:
            df[col] = df[col].replace(0, np.nan)
            total_zeros_converted += zero_count
            logger.info(
                f"  '{col}': {zero_count} zeros converted to NaN "
                f"({zero_count/len(df)*100:.1f}%)"
            )
    
    if total_zeros_converted == 0:
        logger.info("  No zeros to convert.")
    else:
        logger.info(f"  Total zeros converted to NaN: {total_zeros_converted:,}")
    
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows from the DataFrame.

    Duplicates are identified across all columns except pipeline
    metadata columns (__source_file, __ingestion_timestamp).

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with duplicates removed.
    """
    # Exclude metadata columns from duplicate detection
    cols_for_dedup = [c for c in df.columns if not c.startswith("__")]
    
    before = len(df)
    df = df.drop_duplicates(subset=cols_for_dedup)
    removed = before - len(df)

    if removed > 0:
        logger.info(f"  Duplicates removed: {removed:,} rows ({removed/before*100:.1f}%)")
    else:
        logger.info("  No duplicate rows found.")

    return df.reset_index(drop=True)


def fix_dtypes(df: pd.DataFrame, schema_types: Dict[str, str]) -> pd.DataFrame:
    """
    Coerce DataFrame columns to their expected types as defined in config.

    Columns that fail type coercion are left as-is and logged as warnings.
    This prevents a single bad column from aborting the entire pipeline.

    Args:
        df (pd.DataFrame): Input DataFrame.
        schema_types (Dict[str, str]): Mapping of column name → expected dtype.

    Returns:
        pd.DataFrame: DataFrame with corrected dtypes.
    """
    for col, expected_type in schema_types.items():
        if col not in df.columns:
            continue

        current_type = str(df[col].dtype)
        if current_type == expected_type:
            continue

        try:
            if expected_type in ("float64", "int64"):
                df[col] = pd.to_numeric(df[col], errors="coerce")
                # Don't force int64 conversion yet - wait until after outlier capping
                # Just ensure it's numeric
            elif expected_type == "object":
                df[col] = df[col].astype(str)

            logger.info(f"  dtype fix: '{col}' → {current_type} → numeric")
        except Exception as e:
            logger.warning(f"  Could not coerce '{col}' to {expected_type}: {e}")

    return df


def handle_missing_values(
    df: pd.DataFrame,
    fill_strategies: Dict[str, str]
) -> pd.DataFrame:
    """
    Fill missing values using per-column strategies from config.

    Strategies:
      - 'mean'   → fill with column mean (continuous features)
      - 'median' → fill with column median (skewed distributions)
      - 'mode'   → fill with most frequent value (categorical/discrete)
      - 'drop'   → drop rows where this column is null

    Args:
        df (pd.DataFrame): Input DataFrame.
        fill_strategies (Dict[str, str]): Column → strategy mapping.

    Returns:
        pd.DataFrame: DataFrame with missing values handled.
    """
    total_missing_before = df.isnull().sum().sum()

    if total_missing_before == 0:
        logger.info("  No missing values detected. Skipping fill step.")
        return df

    logger.info(f"  Total missing values before fill: {total_missing_before:,}")

    for col, strategy in fill_strategies.items():
        if col not in df.columns:
            continue

        missing_count = df[col].isnull().sum()
        if missing_count == 0:
            continue

        try:
            if strategy == "mean":
                fill_value = df[col].mean()
                df[col] = df[col].fillna(round(fill_value, 4))
            elif strategy == "median":
                fill_value = df[col].median()
                df[col] = df[col].fillna(fill_value)
            elif strategy == "mode":
                fill_value = df[col].mode()[0]
                df[col] = df[col].fillna(fill_value)
            elif strategy == "drop":
                df = df.dropna(subset=[col])
                logger.info(f"  Dropped rows where '{col}' is null.")
                continue

            logger.info(
                f"  Filled '{col}': {missing_count} nulls "
                f"→ strategy='{strategy}', value={fill_value}"
            )
        except Exception as e:
            logger.warning(f"  Could not fill '{col}' using '{strategy}': {e}")

    total_missing_after = df.isnull().sum().sum()
    logger.info(f"  Total missing values after fill: {total_missing_after:,}")

    return df.reset_index(drop=True)


def handle_outliers(
    df: pd.DataFrame,
    numeric_cols: list,
    z_threshold: float = 3.0
) -> pd.DataFrame:
    """
    Detect and cap outliers using the Z-score method (winsorization).

    Outliers are CAPPED to the boundary values, not dropped.
    This preserves dataset size while reducing the influence of extreme values.

    Only applied to numeric columns present in the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.
        numeric_cols (list): List of column names to check.
        z_threshold (float): Z-score threshold above which a value is an outlier.

    Returns:
        pd.DataFrame: DataFrame with outliers capped.
    """
    total_outliers = 0

    for col in numeric_cols:
        if col not in df.columns or df[col].dtype == "object":
            continue

        col_data = df[col].dropna()
        if len(col_data) < 10:
            continue

        z_scores = np.abs(stats.zscore(col_data))
        outlier_mask = z_scores > z_threshold
        outlier_count = outlier_mask.sum()

        if outlier_count > 0:
            lower = col_data.mean() - z_threshold * col_data.std()
            upper = col_data.mean() + z_threshold * col_data.std()
            df[col] = df[col].clip(lower=lower, upper=upper)
            total_outliers += outlier_count
            logger.info(
                f"  Outliers capped in '{col}': {outlier_count} values "
                f"→ range [{lower:.2f}, {upper:.2f}]"
            )

    if total_outliers == 0:
        logger.info("  No significant outliers detected.")
    else:
        logger.info(f"  Total outlier values capped: {total_outliers:,}")

    return df


def drop_metadata_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove internal pipeline metadata columns before saving/training.

    Metadata columns (prefixed with '__') are used for traceability
    during ingestion but should not be part of the cleaned dataset.

    Args:
        df (pd.DataFrame): Input DataFrame with potential metadata columns.

    Returns:
        pd.DataFrame: DataFrame with metadata columns removed.
    """
    meta_cols = [c for c in df.columns if c.startswith("__")]
    if meta_cols:
        df = df.drop(columns=meta_cols)
        logger.info(f"  Dropped metadata columns: {meta_cols}")
    return df


def clean_data(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Main cleaning entry point. Orchestrates all cleaning steps in order.

    Cleaning Pipeline:
      0. Convert zeros to NaN (for medical data where 0 = missing)
      1. Remove duplicates
      2. Fix data types
      3. Handle missing values
      4. Handle outliers
      5. Drop internal metadata columns

    Args:
        df (pd.DataFrame): Raw ingested DataFrame.
        config (Dict[str, Any]): Full config dictionary from config.yaml.

    Returns:
        pd.DataFrame: Cleaned DataFrame ready for validation.
    """
    log_section_header(logger, "Stage 2: Data Cleaning")

    original_shape = df.shape
    logger.info(f"  Input shape: {original_shape[0]:,} rows × {original_shape[1]} cols")

    # Step 0: Convert zeros to NaN for medical impossibilities
    logger.info("► Step 0: Converting zeros to NaN (medical data)...")
    zero_cols = config.get("cleaning", {}).get("zero_as_missing_columns", [])
    df = convert_zeros_to_nan(df, zero_cols)

    # Step 1: Remove duplicates
    logger.info("► Step 1: Removing duplicates...")
    df = remove_duplicates(df)

    # Step 2: Fix data types
    logger.info("► Step 2: Fixing data types...")
    schema_types = config.get("schema", {}).get("column_types", {})
    df = fix_dtypes(df, schema_types)

    # Step 3: Handle missing values
    logger.info("► Step 3: Handling missing values...")
    fill_strategies = config.get("cleaning", {}).get("fill_strategy", {})
    df = handle_missing_values(df, fill_strategies)

    # Step 4: Handle outliers
    logger.info("► Step 4: Detecting and capping outliers...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    z_threshold = config.get("cleaning", {}).get("outlier_z_threshold", 3.0)
    df = handle_outliers(df, numeric_cols, z_threshold)

    # Step 5: Drop metadata
    logger.info("► Step 5: Dropping pipeline metadata columns...")
    df = drop_metadata_columns(df)

    final_shape = df.shape
    logger.info(
        f"  Output shape: {final_shape[0]:,} rows × {final_shape[1]} cols | "
        f"Rows removed: {original_shape[0] - final_shape[0]:,}"
    )

    log_section_footer(logger, "CLEANING COMPLETE")
    return df

"""
ingestion.py
------------
Data Ingestion Module for the ML Pipeline.

Responsibilities:
  - Scan the raw data directory for CSV files
  - Filter by date if a specific run date is provided
  - Load and combine multiple daily CSV files into a single DataFrame
  - Track which files have been processed (avoid reprocessing)

Design Decision:
  File-based ingestion is used here to simulate a daily batch pipeline.
  In production, this module would be replaced or extended to pull from
  S3, a data warehouse, or a message queue — but the interface stays the same.
"""

import os
import glob
import json
from datetime import datetime
from typing import Optional

import pandas as pd

from src.utils.logger import get_logger, log_section_header, log_section_footer

logger = get_logger(__name__)

# Path to the file that tracks already-processed CSVs
PROCESSED_REGISTRY = "data/processed_files.json"


def _load_processed_registry() -> set:
    """
    Load the set of already-processed file paths from the registry.

    Returns:
        set: Set of file paths that have already been ingested.
    """
    if os.path.exists(PROCESSED_REGISTRY):
        with open(PROCESSED_REGISTRY, "r") as f:
            return set(json.load(f))
    return set()


def _save_processed_registry(processed_files: set) -> None:
    """
    Persist the updated set of processed files to disk.

    Args:
        processed_files (set): Updated set of processed file paths.
    """
    os.makedirs(os.path.dirname(PROCESSED_REGISTRY), exist_ok=True)
    with open(PROCESSED_REGISTRY, "w") as f:
        json.dump(list(processed_files), f, indent=2)


def get_raw_files(raw_dir: str, run_date: Optional[str] = None) -> list:
    """
    Scan the raw data directory and return CSV files to process.

    If run_date is provided, only files matching that date pattern
    (YYYY-MM-DD in filename) are returned. Otherwise, returns all
    CSV files not yet in the processed registry.

    Args:
        raw_dir (str): Path to the raw data directory.
        run_date (Optional[str]): Date string in YYYY-MM-DD format.

    Returns:
        list: List of file paths to process.
    """
    all_csv_files = glob.glob(os.path.join(raw_dir, "*.csv"))

    if not all_csv_files:
        logger.warning(f"No CSV files found in '{raw_dir}'")
        return []

    if run_date:
        # Filter to only files that contain the run_date in their name
        matched = [f for f in all_csv_files if run_date in os.path.basename(f)]
        if not matched:
            logger.warning(f"No files found matching date: {run_date}")
        return sorted(matched)

    # Default: return all new (unprocessed) files
    processed = _load_processed_registry()
    new_files = [f for f in all_csv_files if os.path.abspath(f) not in processed]

    if not new_files:
        logger.info("No new files to process. All files already ingested.")
    
    return sorted(new_files)


def load_data(raw_dir: str, run_date: Optional[str] = None) -> pd.DataFrame:
    """
    Main ingestion entry point. Loads CSV files and returns a combined DataFrame.

    Handles:
      - Multiple CSV files (concatenated into one DataFrame)
      - Source file tracking (adds '__source_file' column for traceability)
      - Registry update after successful load

    Args:
        raw_dir (str): Path to the raw data directory.
        run_date (Optional[str]): Date string in YYYY-MM-DD format.

    Returns:
        pd.DataFrame: Combined DataFrame from all ingested files.

    Raises:
        FileNotFoundError: If no CSV files are found.
        ValueError: If CSV files cannot be parsed.
    """
    log_section_header(logger, "Stage 1: Data Ingestion")

    files = get_raw_files(raw_dir, run_date)

    if not files:
        raise FileNotFoundError(
            f"No CSV files to process in '{raw_dir}'"
            + (f" for date {run_date}" if run_date else "")
        )

    logger.info(f"Found {len(files)} file(s) to ingest:")
    for f in files:
        logger.info(f"  → {os.path.basename(f)}")

    dataframes = []
    failed_files = []

    for filepath in files:
        try:
            df = pd.read_csv(filepath)
            df["__source_file"] = os.path.basename(filepath)
            df["__ingestion_timestamp"] = datetime.now().isoformat()
            dataframes.append(df)
            logger.info(
                f"  ✓ Loaded '{os.path.basename(filepath)}' "
                f"— {len(df):,} rows, {len(df.columns)} columns"
            )
        except Exception as e:
            logger.error(f"  ✗ Failed to load '{filepath}': {e}")
            failed_files.append(filepath)

    if not dataframes:
        raise ValueError("All files failed to load. Cannot proceed.")

    combined_df = pd.concat(dataframes, ignore_index=True)

    logger.info(f"Total rows ingested: {len(combined_df):,}")
    logger.info(f"Total columns: {len(combined_df.columns)}")

    if failed_files:
        logger.warning(f"{len(failed_files)} file(s) failed to load and were skipped.")

    # Update registry with successfully processed files
    processed = _load_processed_registry()
    for filepath in files:
        if filepath not in failed_files:
            processed.add(os.path.abspath(filepath))
    _save_processed_registry(processed)

    log_section_footer(logger, "INGESTION COMPLETE")
    return combined_df

"""
main.py
-------
Central Pipeline Orchestrator for the Churn Prediction ML Pipeline.

This is the single entry point for running the pipeline. It:
  1. Parses CLI arguments
  2. Loads configuration from config.yaml
  3. Orchestrates all pipeline stages in sequence
  4. Implements retry logic for resilience
  5. Saves processed data and logs final results

Usage:
  python main.py                          # Process all new files
  python main.py --date 2024-01-15        # Process files for specific date
  python main.py --date 2024-01-15 --force-train  # Force training even after FAIL
  python main.py --skip-drift             # Skip drift detection this run

Pipeline Stages:
  1. Ingestion    → Load raw CSV data
  2. Cleaning     → Fix, fill, cap
  3. Validation   → 3-layer quality gate
  4. Decision     → PASS → train | FAIL → abort
  5. Training     → Train, evaluate, save model
"""

import argparse
import os
import sys
import time
import traceback
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd
import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.ingestion import load_data
from src.cleaning import clean_data
from src.validation import validate_data
from src.decision import should_proceed_to_training
from src.training import train_model
from src.dashboard import generate_dashboard
from src.utils.logger import get_logger, log_section_header, log_section_footer

# Initialize logger early (uses default log dir)
logger = get_logger("main", log_dir="logs")


# =============================================================================
# Configuration Loader
# =============================================================================

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load and return the pipeline configuration from config.yaml.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        Dict[str, Any]: Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Configuration file not found: '{config_path}'. "
            "Ensure config.yaml exists in the project root."
        )

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.info(f"Configuration loaded from: {config_path}")
    return config


# =============================================================================
# Processed Data Saver
# =============================================================================

def save_processed_data(
    df: pd.DataFrame,
    processed_dir: str,
    run_date: Optional[str] = None
) -> str:
    """
    Save the cleaned DataFrame to the processed data directory.

    Uses a versioned, timestamped filename to avoid overwrites.

    Args:
        df (pd.DataFrame): Cleaned DataFrame to save.
        processed_dir (str): Output directory path.
        run_date (Optional[str]): Run date string for filename.

    Returns:
        str: File path of the saved processed data.
    """
    os.makedirs(processed_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    date_part = run_date.replace("-", "") if run_date else timestamp[:8]
    filename = f"processed_{date_part}_{timestamp}.csv"
    filepath = os.path.join(processed_dir, filename)
    df.to_csv(filepath, index=False)
    logger.info(f"  Processed data saved → {filepath}")
    logger.info(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return filepath


# =============================================================================
# CLI Argument Parser
# =============================================================================

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the pipeline.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="ML Data Pipeline — Binary Classification (Configurable for Churn, Risk Prediction, Medical Outcomes)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py
  python main.py --date 2024-01-15
  python main.py --date 2024-01-15 --force-train
  python main.py --config custom_config.yaml
        """
    )

    parser.add_argument(
        "--date",
        type=str,
        default=None,
        metavar="YYYY-MM-DD",
        help="Run pipeline for a specific date (filters raw files by date string)"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        metavar="PATH",
        help="Path to configuration YAML file (default: config.yaml)"
    )

    parser.add_argument(
        "--force-train",
        action="store_true",
        help="Force training even if validation fails (use with caution)"
    )

    parser.add_argument(
        "--skip-drift",
        action="store_true",
        help="Skip data drift detection for this run"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run all stages except training (validation only)"
    )

    return parser.parse_args()


# =============================================================================
# Core Pipeline Runner
# =============================================================================

def run_pipeline(
    config: Dict[str, Any],
    run_date: Optional[str] = None,
    force_train: bool = False,
    skip_drift: bool = False,
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    Execute the full ML pipeline from ingestion to training.

    Args:
        config (Dict[str, Any]): Loaded configuration dictionary.
        run_date (Optional[str]): Date filter for raw file selection.
        force_train (bool): If True, proceed to training even if validation fails.
        skip_drift (bool): If True, disable drift detection for this run.
        dry_run (bool): If True, skip training stage.

    Returns:
        Dict[str, Any]: Pipeline run summary with status and results.
    """
    pipeline_start = datetime.now()
    paths = config.get("paths", {})

    results = {
        "status": "RUNNING",
        "started_at": pipeline_start.isoformat(),
        "run_date": run_date,
        "stages": {}
    }

    # Optionally disable drift detection
    if skip_drift:
        config["drift"]["enabled"] = False
        logger.info("  Drift detection disabled for this run (--skip-drift flag).")

    try:
        # ── Stage 1: Ingestion ────────────────────────────────────────────────
        raw_df = load_data(
            raw_dir=paths.get("raw_data_dir", "data/raw"),
            run_date=run_date
        )
        results["stages"]["ingestion"] = {
            "status": "SUCCESS",
            "rows": len(raw_df),
            "columns": len(raw_df.columns)
        }

        # ── Stage 2: Cleaning ─────────────────────────────────────────────────
        clean_df = clean_data(raw_df, config)
        results["stages"]["cleaning"] = {
            "status": "SUCCESS",
            "rows": len(clean_df),
            "columns": len(clean_df.columns)
        }

        # ── Stage 3: Save Processed Data ──────────────────────────────────────
        log_section_header(logger, "Saving Processed Data")
        processed_path = save_processed_data(
            clean_df,
            processed_dir=paths.get("processed_data_dir", "data/processed"),
            run_date=run_date
        )
        results["stages"]["data_storage"] = {
            "status": "SUCCESS",
            "path": processed_path
        }
        log_section_footer(logger, "DATA SAVED")

        # ── Stage 4: Validation ───────────────────────────────────────────────
        validation_report = validate_data(clean_df, config)
        results["stages"]["validation"] = {
            "status": validation_report["status"],
            "errors": len(validation_report.get("errors", [])),
            "warnings": len(validation_report.get("warnings", []))
        }
        results["validation_report"] = validation_report

        # ── Stage 5: Decision ─────────────────────────────────────────────────
        proceed = should_proceed_to_training(validation_report, config)

        if force_train and not proceed:
            logger.warning(
                "  ⚠ --force-train flag set. Overriding validation FAIL. "
                "Training will proceed. Use with caution."
            )
            proceed = True

        results["stages"]["decision"] = {
            "status": "PROCEED" if proceed else "ABORTED",
            "force_train": force_train
        }

        # ── Stage 6: Training ─────────────────────────────────────────────────
        if proceed and not dry_run:
            training_results = train_model(clean_df, config)
            results["stages"]["training"] = {
                "status": "SUCCESS",
                "model_path": training_results["model_path"],
                "model_version": training_results["model_version"],
                "metrics": training_results["metrics"]
            }
            
            # ── Generate Interactive Dashboard ────────────────────────────────────
            try:
                log_section_header(logger, "Generating Dashboard")
                dashboard_path = generate_dashboard(validation_report, training_results)
                logger.info(f"  Dashboard generated → {dashboard_path}")
                logger.info("  Open in browser to view interactive results")
                results["dashboard_path"] = dashboard_path
                log_section_footer(logger, "DASHBOARD COMPLETE")
            except Exception as e:
                logger.warning(f"  Could not generate dashboard: {e}")
        
        elif dry_run:
            logger.info("  --dry-run flag: Skipping training stage.")
            results["stages"]["training"] = {"status": "SKIPPED (dry-run)"}
        else:
            results["stages"]["training"] = {
                "status": "SKIPPED",
                "reason": "Validation failed. See validation report."
            }

        results["status"] = "SUCCESS"

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        results["status"] = "FAILED"
        results["error"] = str(e)
        raise

    except Exception as e:
        logger.error(f"Pipeline failed with unexpected error: {e}")
        logger.debug(traceback.format_exc())
        results["status"] = "FAILED"
        results["error"] = str(e)
        raise

    finally:
        pipeline_end = datetime.now()
        duration = (pipeline_end - pipeline_start).total_seconds()
        results["completed_at"] = pipeline_end.isoformat()
        results["duration_seconds"] = round(duration, 2)

    return results


# =============================================================================
# Retry Wrapper
# =============================================================================

def run_pipeline_with_retry(
    config: Dict[str, Any],
    run_date: Optional[str] = None,
    force_train: bool = False,
    skip_drift: bool = False,
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    Wrap the pipeline run with configurable retry logic.

    Retries on unexpected failures (e.g. transient I/O issues).
    Validation FAILs are NOT retried — they need human intervention.

    Args:
        config (Dict[str, Any]): Pipeline configuration.
        run_date (Optional[str]): Date filter for raw files.
        force_train (bool): Override validation FAIL.
        skip_drift (bool): Skip drift detection.
        dry_run (bool): Skip training stage.

    Returns:
        Dict[str, Any]: Final pipeline run summary.
    """
    pipeline_cfg = config.get("pipeline", {})
    max_attempts = pipeline_cfg.get("retry_attempts", 3)
    retry_delay = pipeline_cfg.get("retry_delay_seconds", 2)

    last_exception = None

    for attempt in range(1, max_attempts + 1):
        try:
            if attempt > 1:
                logger.info(
                    f"  Retry attempt {attempt}/{max_attempts} "
                    f"(waiting {retry_delay}s)..."
                )
                time.sleep(retry_delay)

            results = run_pipeline(
                config=config,
                run_date=run_date,
                force_train=force_train,
                skip_drift=skip_drift,
                dry_run=dry_run
            )
            return results

        except FileNotFoundError:
            # No point retrying if files don't exist
            raise
        except Exception as e:
            last_exception = e
            if attempt < max_attempts:
                logger.warning(
                    f"  Pipeline attempt {attempt} failed: {e}. Retrying..."
                )
            else:
                logger.error(
                    f"  All {max_attempts} attempts failed. "
                    "Pipeline aborted."
                )

    raise RuntimeError(
        f"Pipeline failed after {max_attempts} attempts."
    ) from last_exception


# =============================================================================
# Entry Point
# =============================================================================

def main():
    """
    Main entry point for the ML Pipeline.

    Loads config, parses CLI args, and runs the pipeline.
    Prints a final summary to stdout on completion.
    """
    # ── Print Banner ──────────────────────────────────────────────────────────
    print("")
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║          ML PIPELINE — BINARY CLASSIFICATION                ║")
    print("║          Production-Style Data Pipeline                     ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"  Run started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("")

    # ── Parse Arguments ───────────────────────────────────────────────────────
    args = parse_args()

    if args.date:
        logger.info(f"Run date filter: {args.date}")
    if args.force_train:
        logger.warning("Force-train mode enabled (--force-train)")
    if args.skip_drift:
        logger.info("Drift detection disabled (--skip-drift)")
    if args.dry_run:
        logger.info("Dry-run mode: training will be skipped")

    # ── Load Config ───────────────────────────────────────────────────────────
    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    # ── Run Pipeline ──────────────────────────────────────────────────────────
    try:
        results = run_pipeline_with_retry(
            config=config,
            run_date=args.date,
            force_train=args.force_train,
            skip_drift=args.skip_drift,
            dry_run=args.dry_run
        )

        # ── Final Summary ──────────────────────────────────────────────────────
        print("")
        print("╔══════════════════════════════════════════════════════════════╗")
        print("║                   PIPELINE SUMMARY                         ║")
        print("╚══════════════════════════════════════════════════════════════╝")
        print(f"  Overall Status  : {results['status']}")
        print(f"  Duration        : {results.get('duration_seconds', 'N/A')}s")
        print("")
        print("  Stage Results:")
        for stage, info in results.get("stages", {}).items():
            s = str(info.get("status", ""))
            if any(k in s for k in ("SUCCESS", "PASS", "PROCEED")):
                status_icon = "✅"
            elif "SKIPPED" in s:
                status_icon = "⚠️ "
            else:
                status_icon = "❌"
            print(f"    {status_icon}  {stage.upper():<20} {info.get('status', '')}")

        # Show model metrics if training succeeded
        training_stage = results.get("stages", {}).get("training", {})
        if training_stage.get("status") == "SUCCESS":
            metrics = training_stage.get("metrics", {})
            print("")
            print("  Model Metrics:")
            print(f"    Accuracy  : {metrics.get('accuracy', 'N/A')}")
            print(f"    Precision : {metrics.get('precision', 'N/A')}")
            print(f"    Recall    : {metrics.get('recall', 'N/A')}")
            print(f"    F1 Score  : {metrics.get('f1_score', 'N/A')}")
            print(f"    ROC-AUC   : {metrics.get('roc_auc', 'N/A')}")
            print(f"    Model     : {training_stage.get('model_path', 'N/A')}")

        print("")
        print("  Check logs/ directory for detailed run logs.")
        print("═" * 64)

    except FileNotFoundError as e:
        print(f"\n  ❌ ERROR: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n  ❌ PIPELINE FAILED: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()

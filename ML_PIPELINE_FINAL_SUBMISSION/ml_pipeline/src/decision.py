"""
decision.py
-----------
Decision Engine for the ML Pipeline.

Acts as the gate between validation and model training.
Based on the validation report, determines whether the pipeline
should proceed to training or abort with a clear reason.

Design Decision:
  This module is intentionally thin — its only job is to read the
  validation report and make a binary decision. By keeping it separate,
  we can later extend it to support more complex decision logic (e.g.,
  "WARN but proceed", escalation policies, alerting) without touching
  any other module.
"""

import json
import os
from datetime import datetime
from typing import Any, Dict

from src.utils.logger import get_logger, log_section_header, log_section_footer

logger = get_logger(__name__)


def save_validation_report(
    report: Dict[str, Any],
    logs_dir: str
) -> str:
    """
    Persist the validation report as a timestamped JSON file.

    Args:
        report (Dict[str, Any]): Validation report from validation.py.
        logs_dir (str): Directory to save the report in.

    Returns:
        str: File path of the saved report.
    """
    os.makedirs(logs_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(logs_dir, f"validation_report_{timestamp}.json")

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"  Validation report saved → {report_path}")
    return report_path


def should_proceed_to_training(
    validation_report: Dict[str, Any],
    config: Dict[str, Any]
) -> bool:
    """
    Evaluate the validation report and decide whether to proceed to training.

    Decision Logic:
      - PASS with no errors → proceed to training
      - FAIL (any errors present) → skip training, log reasons

    Args:
        validation_report (Dict[str, Any]): The structured validation report.
        config (Dict[str, Any]): Full config dictionary from config.yaml.

    Returns:
        bool: True if training should proceed, False otherwise.
    """
    log_section_header(logger, "Stage 4: Decision Engine")

    # Save the report if configured to do so
    if config.get("pipeline", {}).get("save_validation_report", True):
        logs_dir = config.get("paths", {}).get("logs_dir", "logs")
        save_validation_report(validation_report, logs_dir)

    status = validation_report.get("status", "FAIL")
    errors = validation_report.get("errors", [])
    warnings = validation_report.get("warnings", [])
    metrics = validation_report.get("metrics", {})

    # Log summary
    logger.info(f"  Validation Status : {status}")
    logger.info(f"  Errors            : {len(errors)}")
    logger.info(f"  Warnings          : {len(warnings)}")

    if "total_rows" in metrics:
        logger.info(f"  Total Rows        : {metrics['total_rows']:,}")
    if "churn_rate" in metrics:
        logger.info(f"  Churn Rate        : {metrics['churn_rate']*100:.1f}%")
    if "overall_missing_pct" in metrics:
        logger.info(
            f"  Overall Missing   : "
            f"{metrics['overall_missing_pct']*100:.2f}%"
        )

    drift_info = metrics.get("drift_detection", {})
    if drift_info and "total_drifted" in drift_info:
        drifted = drift_info["total_drifted"]
        if drifted > 0:
            logger.warning(
                f"  Data Drift        : ⚠ Detected in "
                f"{drifted} column(s) — monitor closely"
            )
        else:
            logger.info("  Data Drift        : ✓ No drift detected")

    # ── Decision ─────────────────────────────────────────────────────────────
    if status == "PASS":
        logger.info("")
        logger.info("  ✅ DECISION: PROCEED TO TRAINING")
        logger.info("  Data quality meets all thresholds.")
        log_section_footer(logger, "DECISION: PROCEED")
        return True
    else:
        logger.error("")
        logger.error("  ❌ DECISION: SKIP TRAINING")
        logger.error("  Data failed validation. Reasons:")
        for i, err in enumerate(errors, 1):
            logger.error(f"    {i}. {err}")
        logger.error(
            "  Fix the data quality issues above before rerunning the pipeline."
        )
        log_section_footer(logger, "DECISION: ABORTED")
        return False

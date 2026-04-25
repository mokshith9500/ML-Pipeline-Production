"""
auto_pipeline.py
----------------
Fully Automated ML Pipeline Orchestrator

Runs the complete pipeline end-to-end with:
- Automatic data fetching from live sources
- Email alerts on failures
- Auto-deployment if model improves
- Zero manual intervention required

Usage:
  python auto_pipeline.py --source api
  python auto_pipeline.py --source database
  python auto_pipeline.py --source s3

Or schedule with cron:
  0 2 * * * cd /path/to/ml_pipeline && python auto_pipeline.py --source api
"""

import argparse
import json
import os
import smtplib
import sys
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, Optional

import pandas as pd
import requests
import yaml

from main import run_pipeline_with_retry, load_config


# =============================================================================
# Data Source Connectors
# =============================================================================

def fetch_from_api(config: Dict[str, Any]) -> pd.DataFrame:
    """
    Fetch live data from REST API.
    
    Configure in config.yaml:
      live_data:
        api:
          url: "https://api.yourcompany.com/patient-data"
          auth_token: "your_token_here"
          params:
            days: 1  # Last 1 day of data
    """
    api_config = config.get("live_data", {}).get("api", {})
    url = api_config.get("url")
    token = api_config.get("auth_token")
    params = api_config.get("params", {})
    
    if not url:
        raise ValueError("API URL not configured in config.yaml under live_data.api.url")
    
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    
    print(f"📡 Fetching data from API: {url}")
    response = requests.get(url, headers=headers, params=params, timeout=30)
    
    if response.status_code != 200:
        raise ValueError(f"API request failed: {response.status_code} - {response.text}")
    
    data = response.json()
    
    # Handle different API response formats
    if isinstance(data, dict) and "data" in data:
        df = pd.DataFrame(data["data"])
    elif isinstance(data, list):
        df = pd.DataFrame(data)
    else:
        df = pd.DataFrame([data])
    
    print(f"✓ Fetched {len(df):,} rows from API")
    return df


def fetch_from_database(config: Dict[str, Any]) -> pd.DataFrame:
    """
    Fetch live data from PostgreSQL/MySQL database.
    
    Configure in config.yaml:
      live_data:
        database:
          type: "postgresql"  # or "mysql"
          host: "db.yourcompany.com"
          port: 5432
          database: "patient_data"
          user: "ml_user"
          password: "your_password"
          query: "SELECT * FROM patient_records WHERE created_date = CURRENT_DATE"
    """
    db_config = config.get("live_data", {}).get("database", {})
    
    db_type = db_config.get("type", "postgresql")
    
    if db_type == "postgresql":
        from sqlalchemy import create_engine
        connection_string = (
            f"postgresql://{db_config['user']}:{db_config['password']}"
            f"@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        )
    elif db_type == "mysql":
        from sqlalchemy import create_engine
        connection_string = (
            f"mysql+pymysql://{db_config['user']}:{db_config['password']}"
            f"@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        )
    else:
        raise ValueError(f"Unsupported database type: {db_type}")
    
    engine = create_engine(connection_string)
    query = db_config.get("query")
    
    print(f"📊 Querying database: {db_config['host']}/{db_config['database']}")
    df = pd.read_sql(query, engine)
    print(f"✓ Fetched {len(df):,} rows from database")
    
    return df


def fetch_from_s3(config: Dict[str, Any]) -> pd.DataFrame:
    """
    Fetch live data from AWS S3 bucket.
    
    Configure in config.yaml:
      live_data:
        s3:
          bucket: "your-company-data"
          prefix: "patient-records/daily/"
          aws_access_key: "your_key"
          aws_secret_key: "your_secret"
          region: "us-east-1"
    """
    import boto3
    from io import StringIO
    
    s3_config = config.get("live_data", {}).get("s3", {})
    
    s3_client = boto3.client(
        's3',
        aws_access_key_id=s3_config.get("aws_access_key"),
        aws_secret_access_key=s3_config.get("aws_secret_key"),
        region_name=s3_config.get("region", "us-east-1")
    )
    
    bucket = s3_config.get("bucket")
    prefix = s3_config.get("prefix", "")
    
    # Get today's file
    today = datetime.now().strftime("%Y-%m-%d")
    key = f"{prefix}data_{today}.csv"
    
    print(f"☁️  Fetching from S3: s3://{bucket}/{key}")
    
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    df = pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))
    
    print(f"✓ Fetched {len(df):,} rows from S3")
    return df


# =============================================================================
# Auto-Save Fetched Data
# =============================================================================

def save_fetched_data(df: pd.DataFrame, output_dir: str = "data/raw") -> str:
    """Save fetched data as a dated CSV file."""
    os.makedirs(output_dir, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    filepath = os.path.join(output_dir, f"live_data_{today}.csv")
    df.to_csv(filepath, index=False)
    print(f"💾 Saved to: {filepath}")
    return filepath


# =============================================================================
# Email Alerts
# =============================================================================

def send_alert_email(
    subject: str,
    body: str,
    config: Dict[str, Any]
) -> None:
    """
    Send email alert on pipeline failures.
    
    Configure in config.yaml:
      alerts:
        email:
          enabled: true
          smtp_server: "smtp.gmail.com"
          smtp_port: 587
          from_email: "ml-pipeline@yourcompany.com"
          password: "your_app_password"
          to_emails:
            - "data-team@yourcompany.com"
            - "you@yourcompany.com"
    """
    email_config = config.get("alerts", {}).get("email", {})
    
    if not email_config.get("enabled", False):
        print("⚠️  Email alerts disabled in config")
        return
    
    msg = MIMEMultipart()
    msg['From'] = email_config.get("from_email")
    msg['To'] = ", ".join(email_config.get("to_emails", []))
    msg['Subject'] = subject
    
    msg.attach(MIMEText(body, 'plain'))
    
    try:
        server = smtplib.SMTP(
            email_config.get("smtp_server"),
            email_config.get("smtp_port")
        )
        server.starttls()
        server.login(
            email_config.get("from_email"),
            email_config.get("password")
        )
        server.send_message(msg)
        server.quit()
        print(f"✉️  Alert sent to {len(email_config.get('to_emails', []))} recipient(s)")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")


# =============================================================================
# Auto-Deployment Logic
# =============================================================================

def should_deploy_model(
    new_metrics: Dict[str, float],
    threshold: float = 0.80
) -> bool:
    """
    Decide whether to deploy the new model based on performance.
    
    Deploy if:
    - ROC-AUC >= threshold AND
    - F1-Score >= 0.65
    """
    roc_auc = new_metrics.get("roc_auc", 0)
    f1 = new_metrics.get("f1_score", 0)
    
    return roc_auc >= threshold and f1 >= 0.65


def deploy_model(model_path: str, config: Dict[str, Any]) -> None:
    """
    Deploy model to production endpoint.
    
    In reality, this would:
    - Copy model to production S3 bucket
    - Update API server to use new model
    - Create deployment record
    """
    deploy_config = config.get("deployment", {})
    
    if not deploy_config.get("enabled", False):
        print("⚠️  Auto-deployment disabled in config")
        return
    
    # Example: Copy to production location
    prod_path = deploy_config.get("production_model_path", "models/production/model.pkl")
    os.makedirs(os.path.dirname(prod_path), exist_ok=True)
    
    import shutil
    shutil.copy(model_path, prod_path)
    
    print(f"🚀 Model deployed to: {prod_path}")
    
    # Log deployment
    deployment_log = {
        "deployed_at": datetime.now().isoformat(),
        "model_path": model_path,
        "production_path": prod_path
    }
    
    with open("models/deployment_history.json", "a") as f:
        f.write(json.dumps(deployment_log) + "\n")


# =============================================================================
# Main Automated Pipeline
# =============================================================================

def run_automated_pipeline(data_source: str = "api") -> None:
    """
    Run the complete automated pipeline.
    
    Args:
        data_source: Where to fetch data from ('api', 'database', 's3')
    """
    print("\n" + "="*70)
    print("🤖 AUTOMATED ML PIPELINE — STARTING")
    print(f"   Data Source: {data_source.upper()}")
    print(f"   Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")
    
    # Load config
    config = load_config()
    
    try:
        # ── Step 1: Fetch Live Data ──────────────────────────────────────────
        print("\n📥 STEP 1: Fetching live data...")
        
        if data_source == "api":
            df = fetch_from_api(config)
        elif data_source == "database":
            df = fetch_from_database(config)
        elif data_source == "s3":
            df = fetch_from_s3(config)
        else:
            raise ValueError(f"Unknown data source: {data_source}")
        
        # Save to data/raw/
        data_path = save_fetched_data(df)
        
        # ── Step 2: Run Pipeline ──────────────────────────────────────────────
        print("\n🔄 STEP 2: Running ML pipeline...")
        
        results = run_pipeline_with_retry(
            config=config,
            run_date=datetime.now().strftime("%Y-%m-%d")
        )
        
        # ── Step 3: Check Results ─────────────────────────────────────────────
        print("\n📊 STEP 3: Checking results...")
        
        pipeline_status = results.get("status")
        
        if pipeline_status != "SUCCESS":
            # Pipeline failed — send alert
            error_msg = results.get("error", "Unknown error")
            
            send_alert_email(
                subject=f"🚨 ML Pipeline FAILED — {datetime.now().strftime('%Y-%m-%d')}",
                body=f"Pipeline Status: {pipeline_status}\n\nError:\n{error_msg}\n\nCheck logs for details.",
                config=config
            )
            
            print(f"\n❌ Pipeline FAILED: {error_msg}")
            sys.exit(1)
        
        # ── Step 4: Auto-Deploy if Performance Good ───────────────────────────
        training_stage = results.get("stages", {}).get("training", {})
        
        if training_stage.get("status") == "SUCCESS":
            metrics = training_stage.get("metrics", {})
            model_path = training_stage.get("model_path")
            
            print(f"\n🎯 Model Performance:")
            print(f"   ROC-AUC: {metrics.get('roc_auc', 0):.4f}")
            print(f"   F1 Score: {metrics.get('f1_score', 0):.4f}")
            
            if should_deploy_model(metrics):
                print("\n🚀 STEP 4: Auto-deploying model...")
                deploy_model(model_path, config)
                
                send_alert_email(
                    subject=f"✅ New Model Deployed — {datetime.now().strftime('%Y-%m-%d')}",
                    body=f"New model deployed successfully!\n\nMetrics:\n  ROC-AUC: {metrics.get('roc_auc'):.4f}\n  F1 Score: {metrics.get('f1_score'):.4f}\n\nModel: {model_path}",
                    config=config
                )
            else:
                print(f"\n⏸️  Model performance below deployment threshold — skipping deployment")
        
        print("\n" + "="*70)
        print("✅ AUTOMATED PIPELINE COMPLETED SUCCESSFULLY")
        print("="*70 + "\n")
    
    except Exception as e:
        print(f"\n❌ AUTOMATION FAILED: {e}")
        
        send_alert_email(
            subject=f"🚨 ML Pipeline CRASHED — {datetime.now().strftime('%Y-%m-%d')}",
            body=f"Pipeline crashed with error:\n\n{str(e)}\n\nCheck logs immediately.",
            config=config
        )
        
        raise


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Automated ML Pipeline — Zero-Touch Operation"
    )
    
    parser.add_argument(
        "--source",
        type=str,
        default="api",
        choices=["api", "database", "s3"],
        help="Live data source to fetch from"
    )
    
    args = parser.parse_args()
    
    run_automated_pipeline(data_source=args.source)

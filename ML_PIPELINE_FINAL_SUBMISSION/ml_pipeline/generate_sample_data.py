"""
generate_sample_data.py
-----------------------
Generates realistic mock CSV datasets for the ML pipeline demo.

Creates two daily CSV files simulating real churn prediction data.
Run this once to populate data/raw/ before running the pipeline.

Usage:
  python generate_sample_data.py
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta

np.random.seed(42)

def generate_churn_dataset(n_samples: int, date_str: str, churn_rate: float = 0.22) -> pd.DataFrame:
    """
    Generate a realistic churn prediction dataset.
    
    Features are correlated with churn to simulate real-world patterns:
    - Churners tend to have fewer products, more support calls, less engagement
    - Non-churners tend to have higher tenure and engagement
    """
    n_churn = int(n_samples * churn_rate)
    n_no_churn = n_samples - n_churn

    def make_segment(n, is_churn):
        """Generate one segment (churn or no-churn) with realistic distributions."""
        if is_churn:
            age = np.random.normal(38, 12, n).clip(18, 80)
            tenure = np.random.normal(12, 8, n).clip(0, 60)
            monthly = np.random.normal(75, 25, n).clip(20, 200)
            products = np.random.choice([1, 2], n, p=[0.7, 0.3])
            support_calls = np.random.poisson(4, n).clip(0, 20)
            last_login = np.random.normal(45, 20, n).clip(1, 180)
            session_dur = np.random.normal(8, 5, n).clip(0, 60)
        else:
            age = np.random.normal(42, 14, n).clip(18, 90)
            tenure = np.random.normal(36, 18, n).clip(0, 120)
            monthly = np.random.normal(60, 20, n).clip(15, 180)
            products = np.random.choice([1, 2, 3, 4], n, p=[0.3, 0.4, 0.2, 0.1])
            support_calls = np.random.poisson(1, n).clip(0, 10)
            last_login = np.random.normal(8, 5, n).clip(1, 60)
            session_dur = np.random.normal(22, 10, n).clip(5, 90)

        total_charges = (tenure * monthly * np.random.uniform(0.9, 1.1, n)).clip(0)
        churn_labels = np.ones(n, dtype=int) if is_churn else np.zeros(n, dtype=int)

        return pd.DataFrame({
            "age": np.round(age, 1),
            "tenure_months": np.round(tenure, 1),
            "monthly_charges": np.round(monthly, 2),
            "total_charges": np.round(total_charges, 2),
            "num_products": products.astype(int),
            "num_support_calls": support_calls.astype(int),
            "last_login_days": np.round(last_login, 1),
            "avg_session_duration": np.round(session_dur, 2),
            "churn": churn_labels
        })

    churners = make_segment(n_churn, is_churn=True)
    non_churners = make_segment(n_no_churn, is_churn=False)

    df = pd.concat([churners, non_churners], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Add customer IDs
    df.insert(0, "customer_id", [f"CUST_{date_str.replace('-','')}_{i:05d}" for i in range(len(df))])

    # Introduce realistic missing values (~3-4%)
    for col, pct in [("age", 0.02), ("avg_session_duration", 0.03), ("total_charges", 0.02)]:
        mask = np.random.random(len(df)) < pct
        df.loc[mask, col] = np.nan

    # Introduce a few duplicates (~1%)
    n_dupes = max(1, int(len(df) * 0.01))
    dupe_rows = df.sample(n_dupes, random_state=99)
    df = pd.concat([df, dupe_rows], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    return df


def main():
    os.makedirs("data/raw", exist_ok=True)

    today = datetime.now()
    dates = [
        (today - timedelta(days=1)).strftime("%Y-%m-%d"),
        today.strftime("%Y-%m-%d")
    ]
    sizes = [800, 950]

    for date_str, n_samples in zip(dates, sizes):
        df = generate_churn_dataset(n_samples, date_str)
        filepath = f"data/raw/churn_data_{date_str}.csv"
        df.to_csv(filepath, index=False)
        churn_rate = df["churn"].mean() * 100
        print(f"✓ Generated: {filepath}")
        print(f"  Rows: {len(df):,} | Churn rate: {churn_rate:.1f}% | Missing values: {df.isnull().sum().sum()}")

    print("\nSample data generation complete.")
    print("Run the pipeline with: python main.py")


if __name__ == "__main__":
    main()

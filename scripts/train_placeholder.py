#!/usr/bin/env python3
"""
Test: logs a run to MLflow. Replace with real training when eBay data
is collected.

Usage (from repo root):
  python scripts/train_placeholder.py

In another terminal, browse runs:
  mlflow ui --backend-store-uri file:./mlruns
"""

from __future__ import annotations

from pathlib import Path
import sys

import mlflow
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from clothing_mlops.mlflow_setup import set_experiment


def main() -> None:
    data_path = Path("ebay_historical_clothing_scraper/data/processed/ebay_historical_cleaned.csv")
    if not data_path.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found at {data_path}. "
            "Run: python scripts/clean_ebay_exports.py"
        )

    df = pd.read_csv(data_path)
    if df.empty:
        raise ValueError(f"Cleaned dataset is empty: {data_path}")

    valid_price = pd.to_numeric(df["price"], errors="coerce")
    valid_price = valid_price.dropna()

    set_experiment()
    with mlflow.start_run(run_name="placeholder-data-profile"):
        mlflow.set_tags(
            {
                "stage": "data_validation",
                "data_source": "ebay_historical_cleaned",
                "task": "item_value_prediction_ready_check",
            }
        )
        mlflow.log_params(
            {
                "profile_only": True,
                "dataset_path": str(data_path),
            }
        )
        mlflow.log_metrics(
            {
                "row_count": float(len(df)),
                "unique_brands": float(df["brand_name"].nunique(dropna=True)),
                "unique_item_types": float(df["item_type"].nunique(dropna=True)),
                "known_brand_ratio": float((df["brand_name"] != "Unknown").mean()),
                "non_null_price_ratio": float(valid_price.shape[0] / max(len(df), 1)),
                "avg_price": float(valid_price.mean()) if not valid_price.empty else 0.0,
                "median_price": float(valid_price.median()) if not valid_price.empty else 0.0,
            }
        )
    print("Logged data profile run. Start UI: mlflow ui --backend-store-uri file:./mlruns")


if __name__ == "__main__":
    main()

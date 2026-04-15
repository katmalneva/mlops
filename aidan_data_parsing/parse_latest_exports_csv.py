"""Prepare the latest eBay export CSV for step 1 of the parsing pipeline.

This script selects the newest export file, validates the required columns,
applies basic cleaning, and writes a parsing-ready intermediate CSV.

LLM-based structured extraction is intentionally not included yet and will be
added in a later pipeline step.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable

import pandas as pd


REQUIRED_COLUMNS = [
    "item_id",
    "query",
    "title",
    "price_text",
    "price_value",
    "condition_text",
    "sold_date_text",
    "item_url",
    "scraped_at_utc",
]

OUTPUT_COLUMNS = [
    "item_id",
    "query",
    "title",
    "price_value",
    "price_text",
    "condition_text",
    "sold_date_text",
    "item_url",
    "scraped_at_utc",
    "source_file",
    "row_index",
    "title_clean",
    "query_clean",
    "condition_text_clean",
    "parse_status",
    "parse_error",
    "parsed_at_utc",
]

NULLABLE_TEXT_COLUMNS = [
    "query",
    "title",
    "condition_text",
    "price_text",
    "sold_date_text",
]

TEST_ROW_LIMIT = 1


def collapse_whitespace(value: object, *, preserve_nulls: bool = False) -> object:
    """Trim a string value and collapse repeated whitespace."""
    if pd.isna(value):
        return value if preserve_nulls else pd.NA

    text = re.sub(r"\s+", " ", str(value).strip())
    if text == "":
        return value if preserve_nulls else pd.NA
    return text


def project_root() -> Path:
    """Return the repository root based on this script's location."""
    return Path(__file__).resolve().parent.parent


def exports_dir() -> Path:
    """Return the exports directory for the scraper project."""
    return project_root() / "ebay_historical_clothing_scraper" / "data" / "exports"


def outputs_dir() -> Path:
    """Return the output directory for prepared parsing files."""
    return Path(__file__).resolve().parent / "outputs"


def select_latest_export_csv(directory: Path) -> Path:
    """Find the newest export CSV by filename timestamp."""
    if not directory.exists():
        raise FileNotFoundError(f"Exports folder does not exist: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Exports path is not a directory: {directory}")

    csv_files = sorted(directory.glob("ebay_historical_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in exports folder: {directory}")

    return max(csv_files, key=lambda path: path.name)


def load_export_csv(csv_path: Path) -> pd.DataFrame:
    """Load a CSV file into a DataFrame with clear error handling."""
    try:
        return pd.read_csv(csv_path)
    except Exception as exc:  # pragma: no cover - defensive error wrapper
        raise RuntimeError(f"Failed to read CSV file '{csv_path}': {exc}") from exc


def validate_required_columns(df: pd.DataFrame, required_columns: Iterable[str]) -> None:
    """Ensure the required columns are present in the raw export."""
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns in export CSV: "
            + ", ".join(missing)
        )


def clean_nullable_text_columns(df: pd.DataFrame) -> None:
    """Trim target text columns and convert empty strings to null."""
    for column in NULLABLE_TEXT_COLUMNS:
        df[column] = df[column].map(collapse_whitespace)


def build_prepared_dataframe(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """Create the step-1 parsing dataset from the raw export."""
    prepared = df.loc[:, REQUIRED_COLUMNS].copy()
    prepared = prepared[
        [
            "item_id",
            "query",
            "title",
            "price_value",
            "price_text",
            "condition_text",
            "sold_date_text",
            "item_url",
            "scraped_at_utc",
        ]
    ]

    prepared["source_file"] = source_name
    prepared["row_index"] = df.index
    prepared["title_clean"] = prepared["title"].map(collapse_whitespace)
    prepared["query_clean"] = prepared["query"].map(collapse_whitespace)
    prepared["condition_text_clean"] = prepared["condition_text"].map(
        lambda value: collapse_whitespace(value, preserve_nulls=True)
    )
    prepared["parse_status"] = "pending"
    prepared["parse_error"] = ""
    prepared["parsed_at_utc"] = ""

    prepared = prepared.loc[:, OUTPUT_COLUMNS]

    # Temporary limit for parser testing: prepare one row at a time.
    return prepared.head(TEST_ROW_LIMIT).copy()


def print_null_summary(df: pd.DataFrame) -> None:
    """Print null counts for fields most relevant to the next parsing step."""
    summary_columns = [
        "query",
        "title",
        "price_value",
        "condition_text",
        "sold_date_text",
    ]
    print("Null summary:")
    for column in summary_columns:
        print(f"  {column}: {int(df[column].isna().sum())}")


def write_prepared_csv(df: pd.DataFrame, original_name: str, output_directory: Path) -> Path:
    """Write the prepared DataFrame to the outputs directory."""
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / f"prepared_{original_name}"
    df.to_csv(output_path, index=False)
    return output_path


def main() -> None:
    """Run step 1 of the parsing pipeline for the latest export file."""
    source_directory = exports_dir()
    latest_csv = select_latest_export_csv(source_directory)
    raw_df = load_export_csv(latest_csv)
    validate_required_columns(raw_df, REQUIRED_COLUMNS)
    clean_nullable_text_columns(raw_df)
    prepared_df = build_prepared_dataframe(raw_df, latest_csv.name)
    output_path = write_prepared_csv(prepared_df, latest_csv.name, outputs_dir())

    print(f"Selected source file: {latest_csv}")
    print(f"Row count: {len(prepared_df)}")
    print(f"Columns loaded: {list(raw_df.columns)}")
    print(f"Output file: {output_path}")
    print_null_summary(prepared_df)


if __name__ == "__main__":
    main()

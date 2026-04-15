"""Parse one pending prepared row with NVIDIA-hosted LLM inference.

This is step 2 of the parsing pipeline. It reads the latest prepared CSV from
the step-1 output folder, selects the first row with `parse_status == "pending"`,
and writes parsed structured fields back into the same CSV.

The script is intentionally limited to one row at a time for parser testing.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

from dotenv import load_dotenv
import os
import pandas as pd
import requests


NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "meta/llama-3.3-70b-instruct"
REQUEST_TIMEOUT_SECONDS = 90
YEAR_ESTIMATION_KEYS = [
    "release_year",
    "release_year_min",
    "release_year_max",
    "release_year_confidence",
]

PARSED_COLUMNS = [
    "parsed_brand_name",
    "parsed_item_type",
    "parsed_item_subtype",
    "parsed_department",
    "parsed_gender",
    "parsed_age_group",
    "parsed_size",
    "parsed_size_type",
    "parsed_us_shoe_size",
    "parsed_uk_shoe_size",
    "parsed_eu_shoe_size",
    "parsed_waist_size",
    "parsed_inseam",
    "parsed_length",
    "parsed_fit",
    "parsed_color_primary",
    "parsed_color_secondary",
    "parsed_material_primary",
    "parsed_material_secondary",
    "parsed_pattern",
    "parsed_closure",
    "parsed_occasion",
    "parsed_season",
    "parsed_style",
    "parsed_theme",
    "parsed_model_name",
    "parsed_product_line",
    "parsed_style_code",
    "parsed_sport",
    "parsed_heel_height",
    "parsed_sleeve_length",
    "parsed_neckline",
    "parsed_embellishment",
    "parsed_performance_activity",
    "parsed_made_in",
    "parsed_vintage_era",
    "parsed_release_year",
    "parsed_release_year_min",
    "parsed_release_year_max",
    "parsed_release_year_confidence",
    "parsed_condition",
    "parsed_condition_detail",
    "parsed_has_box",
    "llm_model",
    "llm_response_text",
]

FEATURE_KEYS = [
    "brand_name",
    "item_type",
    "item_subtype",
    "department",
    "gender",
    "age_group",
    "size",
    "size_type",
    "us_shoe_size",
    "uk_shoe_size",
    "eu_shoe_size",
    "waist_size",
    "inseam",
    "length",
    "fit",
    "color_primary",
    "color_secondary",
    "material_primary",
    "material_secondary",
    "pattern",
    "closure",
    "occasion",
    "season",
    "style",
    "theme",
    "model_name",
    "product_line",
    "style_code",
    "sport",
    "heel_height",
    "sleeve_length",
    "neckline",
    "embellishment",
    "performance_activity",
    "made_in",
    "vintage_era",
    "release_year",
    "release_year_min",
    "release_year_max",
    "release_year_confidence",
    "condition",
    "condition_detail",
    "has_box",
]


def script_dir() -> Path:
    """Return the directory containing this script."""
    return Path(__file__).resolve().parent


def repo_root() -> Path:
    """Return the repository root based on this script location."""
    return script_dir().parent


def outputs_dir() -> Path:
    """Return the folder holding prepared parsing CSV files."""
    return script_dir() / "outputs"


def env_file() -> Path:
    """Return the root .env path."""
    return repo_root() / ".env"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Parse one pending row from the latest prepared CSV."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Select and print the pending row without calling NVIDIA or saving changes.",
    )
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="If no pending rows exist, retry the first row with parse_status='error'.",
    )
    return parser.parse_args()


def select_latest_prepared_csv(directory: Path) -> Path:
    """Find the newest prepared CSV in the outputs directory."""
    if not directory.exists():
        raise FileNotFoundError(f"Prepared outputs folder does not exist: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Prepared outputs path is not a directory: {directory}")

    csv_files = sorted(directory.glob("prepared_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No prepared CSV files found in: {directory}")

    return max(csv_files, key=lambda path: path.name)


def load_prepared_csv(csv_path: Path) -> pd.DataFrame:
    """Load the prepared CSV while preserving empty strings."""
    try:
        return pd.read_csv(csv_path, keep_default_na=False)
    except Exception as exc:  # pragma: no cover - defensive wrapper
        raise RuntimeError(f"Failed to read prepared CSV '{csv_path}': {exc}") from exc


def ensure_parsed_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add parsed output columns if they do not already exist."""
    for column in PARSED_COLUMNS:
        if column not in df.columns:
            df[column] = ""
    return df


def first_candidate_row_index(df: pd.DataFrame, *, retry_errors: bool = False) -> int:
    """Return the first eligible row index for parsing."""
    if "parse_status" not in df.columns:
        raise ValueError("Prepared CSV is missing required column: parse_status")

    status = df["parse_status"].astype(str).str.strip()
    pending_rows = df.index[status.eq("pending")]
    if len(pending_rows) == 0:
        if retry_errors:
            error_rows = df.index[status.eq("error")]
            if len(error_rows) > 0:
                return int(error_rows[0])
            parsed_rows = df.index[status.eq("parsed")]
            if len(parsed_rows) > 0:
                return int(parsed_rows[0])
        raise ValueError("No eligible rows found. Expected parse_status='pending'.")
    return int(pending_rows[0])


def load_nvidia_config() -> tuple[str, str]:
    """Load NVIDIA credentials and model configuration from environment."""
    load_dotenv(env_file())

    api_key = os.getenv("NVIDIA_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            f"NVIDIA_API_KEY is missing. Add it to {env_file()} before running this script."
        )

    model = os.getenv("NVIDIA_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    return api_key, model


def build_messages(row: pd.Series) -> list[dict[str, str]]:
    """Build the extraction prompt for one row."""
    def json_safe(value: Any) -> Any:
        if pd.isna(value):
            return ""
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                pass
        return value

    user_payload = {
        "item_id": json_safe(row.get("item_id", "")),
        "query_clean": json_safe(row.get("query_clean", "")),
        "title_clean": json_safe(row.get("title_clean", "")),
        "condition_text_clean": json_safe(row.get("condition_text_clean", "")),
        "price_value": json_safe(row.get("price_value", "")),
        "price_text": json_safe(row.get("price_text", "")),
        "sold_date_text": json_safe(row.get("sold_date_text", "")),
    }

    system_prompt = (
        "You extract structured apparel resale features from eBay listing text. "
        "Return only valid JSON with exactly these keys: "
        + ", ".join(FEATURE_KEYS)
        + ". "
        "Every value must be a string. Use an empty string when a field is not explicit or not reliable. "
        "Do not invent missing details. "
        "Normalize item_type to a concise lowercase product category such as sneakers, boots, jeans, jacket, hoodie, sweater, shirt, pants, shorts, dress, skirt, leggings, blazer, coat, cleats, sandals, loafers, or shoes. "
        "Use item_subtype for the more specific subtype when available, such as bomber jacket, straight leg jeans, crewneck sweater, running shoes, ankle boots, maxi dress, or polo shirt. "
        "department should be one of footwear, tops, bottoms, outerwear, dresses, accessories, or ''. "
        "Normalize gender to one of: men, women, unisex, boys, girls, infant, toddler, or ''. "
        "Normalize age_group to one of: adult, kids, or ''. "
        "Normalize condition to one of: new, used, or ''. "
        "has_box must be one of: yes, no, or ''. "
        "For year fields, first use an explicit year in the title if one is present. "
        "Only estimate release_year, release_year_min, release_year_max, and release_year_confidence when the title does not explicitly provide a year. "
        "Return release_year as a single best estimated year when there is enough evidence, otherwise ''. "
        "Return release_year_min and release_year_max as the plausible production year range when you can infer a range, otherwise ''. "
        "Return release_year_confidence as a numeric string from 0.00 to 1.00 based on confidence in the year value or estimate, otherwise ''. "
        "Use title, query, and condition text together. "
        "Capture obvious tokens such as size systems, measurements, style code, product line, sport, colors, materials, fit, neckline, sleeve length, and model name when present."
    )

    user_prompt = (
        "Extract the fields from this row and return JSON only.\n"
        f"{json.dumps(user_payload, ensure_ascii=True)}"
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_year_estimation_messages(row: pd.Series, parsed: dict[str, Any]) -> list[dict[str, str]]:
    """Build a focused prompt for release-year estimation."""
    def json_safe(value: Any) -> Any:
        if pd.isna(value):
            return ""
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                pass
        return value

    user_payload = {
        "item_id": json_safe(row.get("item_id", "")),
        "query_clean": json_safe(row.get("query_clean", "")),
        "title_clean": json_safe(row.get("title_clean", "")),
        "condition_text_clean": json_safe(row.get("condition_text_clean", "")),
        "brand_name": json_safe(parsed.get("brand_name", "")),
        "item_type": json_safe(parsed.get("item_type", "")),
        "item_subtype": json_safe(parsed.get("item_subtype", "")),
        "model_name": json_safe(parsed.get("model_name", "")),
        "product_line": json_safe(parsed.get("product_line", "")),
        "style_code": json_safe(parsed.get("style_code", "")),
        "sport": json_safe(parsed.get("sport", "")),
    }

    system_prompt = (
        "You estimate the release year of apparel or footwear from eBay listing text. "
        "Return only valid JSON with exactly these keys: "
        "release_year, release_year_min, release_year_max, release_year_confidence. "
        "Every value must be a string. "
        "If the title contains an explicit 4-digit year, use it directly. "
        "Otherwise, make a best-effort estimate using brand, model name, product line, style code, sport, and title wording. "
        "Do not leave all year fields blank if there is enough evidence to make a rough estimate. "
        "release_year should be the single best estimated year. "
        "release_year_min and release_year_max should be a plausible production-year range. "
        "release_year_confidence must be a numeric string from 0.00 to 1.00. "
        "If the evidence is weak, still return a broad range and a low confidence score instead of blank values. "
        "Only return blanks if there is truly not enough information to make any useful estimate."
    )
    user_prompt = (
        "Estimate release year information for this listing and return JSON only.\n"
        f"{json.dumps(user_payload, ensure_ascii=True)}"
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def call_nvidia_chat_completion(
    *,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
) -> str:
    """Call NVIDIA's OpenAI-compatible chat completions endpoint."""
    response = requests.post(
        f"{NVIDIA_BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": messages,
            "temperature": 0.2,
            "top_p": 0.7,
            "max_tokens": 900,
            "stream": False,
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()

    payload = response.json()
    try:
        return payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"Unexpected NVIDIA response shape: {payload}") from exc


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract the first JSON object from model text output."""
    text = text.strip()

    try:
        candidate = json.loads(text)
        if isinstance(candidate, dict):
            return candidate
    except json.JSONDecodeError:
        pass

    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced_match:
        try:
            candidate = json.loads(fenced_match.group(1))
            if isinstance(candidate, dict):
                return candidate
        except json.JSONDecodeError as exc:
            raise ValueError(f"Model returned fenced JSON that could not be parsed: {exc}") from exc

    if text.startswith("```") and not text.endswith("```"):
        raise ValueError("Model response was truncated before the closing code fence.")

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"Model response did not contain a JSON object: {text}")

    candidate = json.loads(match.group(0))
    if not isinstance(candidate, dict):
        raise ValueError(f"Model JSON response is not an object: {candidate}")
    return candidate


def normalize_value(value: Any, *, lowercase: bool = False) -> str:
    """Normalize parsed values into clean strings."""
    if value is None:
        return "unknown" if lowercase else ""

    text = re.sub(r"\s+", " ", str(value).strip())
    if not text:
        return "unknown" if lowercase else ""
    return text.lower() if lowercase else text


def normalize_condition(value: Any) -> str:
    """Normalize a parsed condition into the expected label set."""
    text = normalize_value(value, lowercase=True)
    if text in {"new", "used", "unknown"}:
        return text

    if "new" in text:
        return "new"
    if text in {"pre-owned", "preowned", "used", "worn"} or "used" in text:
        return "used"
    return ""


def normalize_binary_flag(value: Any) -> str:
    """Normalize yes/no style parsed flags."""
    text = normalize_value(value, lowercase=True)
    if text in {"yes", "no"}:
        return text
    return ""


def infer_condition_from_text(condition_text: str) -> str:
    """Infer normalized condition from the raw condition text when obvious."""
    text = normalize_value(condition_text, lowercase=True)
    if not text:
        return ""
    if "new" in text:
        return "new"
    if any(token in text for token in ["pre-owned", "preowned", "used"]):
        return "used"
    return ""


def infer_gender_from_text(*values: Any) -> str:
    """Infer gender when the title or condition text makes it explicit."""
    text = " ".join(normalize_value(value, lowercase=True) for value in values if value is not None)
    if not text:
        return ""
    if re.search(r"\b(mens|men's|men|male)\b", text):
        return "men"
    if re.search(r"\b(womens|women's|women|female|ladies)\b", text):
        return "women"
    if re.search(r"\b(unisex)\b", text):
        return "unisex"
    if re.search(r"\b(boys|boy's|boys')\b", text):
        return "boys"
    if re.search(r"\b(girls|girl's|girls')\b", text):
        return "girls"
    if re.search(r"\b(toddler)\b", text):
        return "toddler"
    if re.search(r"\b(infant|baby)\b", text):
        return "infant"
    return ""


def infer_age_group(gender: str, *values: Any) -> str:
    """Infer age group from gender or title cues."""
    if gender in {"boys", "girls", "infant", "toddler"}:
        return "kids"
    text = " ".join(normalize_value(value, lowercase=True) for value in values if value is not None)
    if re.search(r"\b(kids|youth|child|children|toddler|infant|baby|boys|girls)\b", text):
        return "kids"
    if gender in {"men", "women", "unisex"}:
        return "adult"
    return ""


def infer_style_code(*values: Any) -> str:
    """Extract a likely style or SKU code from title-like text."""
    text = " ".join(normalize_value(value) for value in values if value is not None)
    match = re.search(r"\b([A-Z0-9]{2,}-[A-Z0-9]{2,}|[A-Z]{2,}\d{3,}(?:-\d{2,})?)\b", text)
    return match.group(1) if match else ""


def infer_explicit_years_from_title(title_text: str) -> tuple[str, str, str, str]:
    """Extract explicit year information from title text when present."""
    text = normalize_value(title_text)
    if not text:
        return "", "", "", ""

    years = re.findall(r"\b(19\d{2}|20\d{2}|21\d{2})\b", text)
    if not years:
        return "", "", "", ""

    unique_years = sorted({int(year) for year in years})
    if len(unique_years) == 1:
        year = str(unique_years[0])
        return year, year, year, "1.00"

    return (
        str(unique_years[0]),
        str(unique_years[0]),
        str(unique_years[-1]),
        "0.95",
    )


def normalize_year_value(value: Any) -> str:
    """Normalize a year-like value into a 4-digit year string."""
    text = normalize_value(value)
    if not text:
        return ""
    match = re.search(r"\b(19\d{2}|20\d{2}|21\d{2})\b", text)
    return match.group(1) if match else ""


def normalize_year_confidence(value: Any) -> str:
    """Normalize a confidence score into a 0.00-1.00 string."""
    text = normalize_value(value)
    if not text:
        return ""
    try:
        score = float(text)
    except ValueError:
        match = re.search(r"\d*\.?\d+", text)
        if not match:
            return ""
        score = float(match.group(0))

    score = max(0.0, min(1.0, score))
    return f"{score:.2f}"


def normalize_feature_value(key: str, value: Any) -> str:
    """Normalize a single parsed feature field."""
    text = normalize_value(value)

    if key in {
        "item_type",
        "item_subtype",
        "department",
        "gender",
        "age_group",
        "condition",
        "pattern",
        "closure",
        "occasion",
        "season",
        "style",
        "theme",
        "sport",
        "sleeve_length",
        "neckline",
        "performance_activity",
    }:
        text = text.lower()
    if key == "condition":
        return normalize_condition(text)
    if key == "has_box":
        return normalize_binary_flag(text)
    if key in {"release_year", "release_year_min", "release_year_max"}:
        return normalize_year_value(text)
    if key == "release_year_confidence":
        return normalize_year_confidence(text)
    if key in {
        "item_type",
        "item_subtype",
        "department",
        "gender",
        "age_group",
        "pattern",
        "closure",
        "occasion",
        "season",
        "style",
        "theme",
        "sport",
        "sleeve_length",
        "neckline",
        "performance_activity",
    }:
        return text
    return text


def should_estimate_year(normalized: dict[str, str], title_text: str) -> bool:
    """Return True when the focused year estimator should run."""
    explicit_year, _, _, _ = infer_explicit_years_from_title(title_text)
    if explicit_year:
        return False
    return not any(normalized.get(key, "") for key in YEAR_ESTIMATION_KEYS)


def merge_year_estimate(
    normalized: dict[str, str],
    estimated_years: dict[str, Any],
) -> dict[str, str]:
    """Merge focused year-estimation output into the normalized feature dict."""
    for key in YEAR_ESTIMATION_KEYS:
        normalized[key] = normalize_feature_value(key, estimated_years.get(key, ""))
    return normalized


def apply_parsed_result(
    df: pd.DataFrame,
    row_index: int,
    parsed: dict[str, Any],
    *,
    api_key: str,
    model: str,
    raw_response_text: str,
) -> None:
    """Write parsed fields and status updates into the DataFrame."""
    title_text = str(df.at[row_index, "title_clean"]) if "title_clean" in df.columns else ""
    condition_text = (
        str(df.at[row_index, "condition_text_clean"])
        if "condition_text_clean" in df.columns
        else ""
    )

    normalized = {key: normalize_feature_value(key, parsed.get(key, "")) for key in FEATURE_KEYS}

    inferred_gender = infer_gender_from_text(title_text, condition_text)
    if inferred_gender:
        normalized["gender"] = inferred_gender

    inferred_age_group = infer_age_group(normalized.get("gender", ""), title_text, condition_text)
    if inferred_age_group:
        normalized["age_group"] = inferred_age_group

    inferred_style_code = infer_style_code(title_text, condition_text)
    if inferred_style_code and not normalized["style_code"]:
        normalized["style_code"] = inferred_style_code

    inferred_condition = infer_condition_from_text(condition_text)
    if inferred_condition:
        normalized["condition"] = inferred_condition

    explicit_year, explicit_year_min, explicit_year_max, explicit_year_confidence = (
        infer_explicit_years_from_title(title_text)
    )
    if explicit_year:
        normalized["release_year"] = explicit_year
        normalized["release_year_min"] = explicit_year_min
        normalized["release_year_max"] = explicit_year_max
        normalized["release_year_confidence"] = explicit_year_confidence
    elif should_estimate_year(normalized, title_text):
        year_messages = build_year_estimation_messages(df.loc[row_index], parsed)
        year_response_text = call_nvidia_chat_completion(
            api_key=api_key,
            model=model,
            messages=year_messages,
        )
        year_parsed = extract_json_object(year_response_text)
        normalized = merge_year_estimate(normalized, year_parsed)
        raw_response_text = (
            raw_response_text.strip()
            + "\n\n--- YEAR ESTIMATE ---\n"
            + year_response_text.strip()
        )

    if normalized["brand_name"].lower() == "unknown":
        normalized["brand_name"] = ""

    for key, value in normalized.items():
        df.at[row_index, f"parsed_{key}"] = value

    df.at[row_index, "llm_model"] = model
    df.at[row_index, "llm_response_text"] = raw_response_text.strip()
    df.at[row_index, "parse_status"] = "parsed"
    df.at[row_index, "parse_error"] = ""
    df.at[row_index, "parsed_at_utc"] = datetime.now(timezone.utc).isoformat()


def apply_parse_error(df: pd.DataFrame, row_index: int, error_message: str) -> None:
    """Store a parse error on the selected row."""
    df.at[row_index, "parse_status"] = "error"
    df.at[row_index, "parse_error"] = error_message
    df.at[row_index, "parsed_at_utc"] = datetime.now(timezone.utc).isoformat()


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    """Persist the updated DataFrame in place."""
    df.to_csv(path, index=False)


def main() -> None:
    """Run step 2 for one pending prepared row."""
    args = parse_args()
    prepared_csv = select_latest_prepared_csv(outputs_dir())
    df = ensure_parsed_columns(load_prepared_csv(prepared_csv))
    row_index = first_candidate_row_index(df, retry_errors=args.retry_errors)
    row = df.loc[row_index]

    print(f"Selected prepared file: {prepared_csv}")
    print(f"Selected row index: {row_index}")
    print(f"Selected parse_status: {row.get('parse_status', '')}")
    print(f"Selected item_id: {row.get('item_id', '')}")
    print(f"Selected title_clean: {row.get('title_clean', '')}")

    if args.dry_run:
        print("Dry run enabled: skipping NVIDIA API call and file update.")
        return

    api_key, model = load_nvidia_config()
    messages = build_messages(row)

    try:
        raw_response_text = call_nvidia_chat_completion(
            api_key=api_key,
            model=model,
            messages=messages,
        )
        parsed = extract_json_object(raw_response_text)
        apply_parsed_result(
            df,
            row_index,
            parsed,
            api_key=api_key,
            model=model,
            raw_response_text=raw_response_text,
        )
    except Exception as exc:
        apply_parse_error(df, row_index, str(exc))
        save_dataframe(df, prepared_csv)
        raise

    save_dataframe(df, prepared_csv)
    print("Parse completed successfully.")
    print(f"Updated parse_status: {df.at[row_index, 'parse_status']}")
    print(f"Parsed brand: {df.at[row_index, 'parsed_brand_name']}")
    print(f"Parsed item_type: {df.at[row_index, 'parsed_item_type']}")
    print(f"Parsed condition: {df.at[row_index, 'parsed_condition']}")


if __name__ == "__main__":
    main()

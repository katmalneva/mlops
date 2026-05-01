"""Estimate missing `original_price` values for parsed listings via NVIDIA LLM.

Companion to ``extract_features_from_mongo.py``. Walks the parsed feature
collection, finds documents where ``original_price`` is missing, null, or an
empty/whitespace string, and asks an NVIDIA-hosted chat completions model to
make an educated guess at the item's new-retail price using the structured
features that have already been extracted (brand, item_type, model_name,
materials, etc.) plus the cleaned title/query text.

The estimate is written back to:

    * a local JSONL audit file (one line per processed doc), and
    * the same parsed document in MongoDB, populating ``original_price`` along
      with provenance fields (``original_price_source``,
      ``original_price_confidence``, ``original_price_estimated_at_utc``,
      ``original_price_llm_model``) so re-runs only target unfilled rows.

Environment variables (loaded from `.env` via python-dotenv when present):

    NVIDIA_API_KEY        Required (unless ``--dry-run``).
    NVIDIA_MODEL          Optional. Defaults to "meta/llama-3.3-70b-instruct".
    MONGODB_URI           Required.
    MONGODB_DATABASE      Optional. Defaults to "historical".
    MONGODB_PARSED_COLL   Optional. Defaults to "parsed".

Usage::

    python scripts/estimate_original_price_from_mongo.py --limit 100
    python scripts/estimate_original_price_from_mongo.py --limit 5 --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import certifi
import requests
from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne
from pymongo.errors import PyMongoError

DEFAULT_DATABASE = "historical"
DEFAULT_PARSED_COLLECTION = "parsed"
DEFAULT_MODEL = "meta/llama-3.3-70b-instruct"
DEFAULT_LIMIT = 100

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
REQUEST_TIMEOUT_SECONDS = 120
INTER_REQUEST_DELAY_SECONDS = 5

# Feature fields from the parsed doc that we feed to the LLM as context.
CONTEXT_FEATURE_KEYS = [
    "brand_name",
    "item_type",
    "item_subtype",
    "department",
    "gender",
    "age_group",
    "size",
    "size_type",
    "color_primary",
    "color_secondary",
    "material_primary",
    "material_secondary",
    "pattern",
    "fit",
    "style",
    "model_name",
    "product_line",
    "style_code",
    "sport",
    "performance_activity",
    "made_in",
    "release_year",
    "release_year_min",
    "release_year_max",
    "condition",
    "condition_detail",
    "has_box",
    "parsed_price",
]

CONTEXT_TEXT_KEYS = [
    "title_clean",
    "query_clean",
    "condition_text_clean",
]

SYSTEM_PROMPT = (
    "You estimate the original MSRP / new-retail price (in USD) of a used apparel "
    "or footwear item, given structured features extracted from an eBay resale listing. "
    "Return ONLY valid JSON with exactly these keys: "
    "original_price, original_price_confidence, reasoning. "
    "original_price must be a plain decimal string in USD with no currency symbol or "
    "thousands separators (e.g. '129.99'). It represents your best estimate of the "
    "price the item would have cost if purchased new at retail. "
    "If the brand/model is well-known, anchor on its typical retail price. "
    "If brand is generic or unknown, infer from item_type, materials, and category norms. "
    "Never leave original_price empty - always commit to a numeric estimate. "
    "original_price_confidence must be a decimal string from '0.00' to '1.00' reflecting "
    "how confident you are in the estimate (higher when brand/model are known). "
    "reasoning must be one short sentence (under 200 chars) explaining the basis of the "
    "estimate. Do not include any text outside the JSON object."
)


def _build_mongo_client(uri: str) -> MongoClient:
    """Create a MongoClient that trusts the certifi CA bundle (matches sibling script)."""
    return MongoClient(
        uri,
        tls=True,
        tlsCAFile=certifi.where(),
        serverSelectionTimeoutMS=20000,
    )


def _is_missing_price(value: Any) -> bool:
    """True when the original_price field is absent or effectively empty."""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    return False


def _json_safe(value: Any) -> Any:
    """Coerce Mongo/pandas-ish values into JSON-serializable primitives."""
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def fetch_documents_missing_price(
    *,
    uri: str,
    database: str,
    collection: str,
    limit: int,
) -> list[dict[str, Any]]:
    """Pull parsed docs whose ``original_price`` is missing/null/empty.

    We also skip docs that already carry an LLM estimate, so re-running the
    script doesn't re-bill the model on the same rows. To force a re-estimate,
    unset ``original_price_source`` on the target docs.
    """
    query = {
        "$and": [
            {
                "$or": [
                    {"original_price": {"$exists": False}},
                    {"original_price": None},
                    {"original_price": ""},
                    # Strings that are only whitespace; Mongo regex anchored.
                    {"original_price": {"$regex": r"^\s*$"}},
                ]
            },
            {
                "$or": [
                    {"original_price_source": {"$exists": False}},
                    {"original_price_source": ""},
                ]
            },
        ]
    }

    client = _build_mongo_client(uri)
    try:
        client.admin.command("ping")
        cursor = (
            client[database][collection]
            .find(query)
            .sort("_id", -1)
            .limit(limit)
        )
        return list(cursor)
    finally:
        client.close()


def build_messages(document: dict[str, Any]) -> list[dict[str, str]]:
    """Build the chat messages payload for one parsed document."""
    user_payload: dict[str, Any] = {
        key: _json_safe(document.get(key, "")) for key in CONTEXT_FEATURE_KEYS
    }
    user_payload.update(
        {key: _json_safe(document.get(key, "")) for key in CONTEXT_TEXT_KEYS}
    )

    user_prompt = (
        "Estimate the original new-retail price for the apparel/footwear item "
        "described by these features and return JSON only.\n"
        f"{json.dumps(user_payload, ensure_ascii=True)}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def call_nvidia_chat(
    *, api_key: str, model: str, messages: list[dict[str, str]]
) -> dict[str, Any]:
    """Call the NVIDIA chat/completions endpoint and return the parsed JSON body."""
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
            "max_tokens": 250,
            "stream": False,
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
_PRICE_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _normalize_price_str(value: Any) -> str:
    """Coerce an LLM price value into a clean decimal string, or '' if not parseable."""
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    text = text.replace(",", "").replace("$", "").replace("USD", "").strip()
    match = _PRICE_RE.search(text)
    if not match:
        return ""
    try:
        return f"{float(match.group(0)):.2f}"
    except ValueError:
        return ""


def _normalize_confidence_str(value: Any) -> str:
    """Coerce the confidence value into a 0.00-1.00 decimal string, or ''."""
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    match = _PRICE_RE.search(text)
    if not match:
        return ""
    try:
        confidence = float(match.group(0))
    except ValueError:
        return ""
    confidence = max(0.0, min(1.0, confidence))
    return f"{confidence:.2f}"


def parse_llm_json(content: str) -> dict[str, str]:
    """Extract a JSON object from the LLM response, stripping code fences."""
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        match = _JSON_BLOCK_RE.search(stripped)
        if not match:
            raise
        parsed = json.loads(match.group(0))

    if not isinstance(parsed, dict):
        raise ValueError("LLM response was not a JSON object")

    price = _normalize_price_str(parsed.get("original_price", ""))
    if not price:
        raise ValueError(
            f"LLM response did not contain a usable original_price: {parsed!r}"
        )

    return {
        "original_price": price,
        "original_price_confidence": _normalize_confidence_str(
            parsed.get("original_price_confidence", "")
        ),
        "original_price_reasoning": str(parsed.get("reasoning", "") or "")[:500],
    }


def write_results(path: Path, rows: list[dict[str, Any]]) -> None:
    """Persist estimate rows to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def update_parsed_with_estimates(
    *,
    uri: str,
    database: str,
    collection: str,
    rows: list[dict[str, Any]],
) -> int:
    """Write the LLM price estimates back into the parsed collection.

    Successful rows update ``original_price`` plus provenance fields so that
    subsequent runs skip them. Errored rows still record the failure on the
    source doc so we can audit them, but leave ``original_price`` untouched.
    """
    if not rows:
        return 0

    ops = []
    for row in rows:
        source_id = row.get("source_id")
        if source_id is None:
            continue

        if row.get("estimate_status") == "ok":
            ops.append(
                UpdateOne(
                    {"_id": source_id},
                    {
                        "$set": {
                            "original_price": row["original_price"],
                            "original_price_source": "llm_estimate",
                            "original_price_confidence": row.get(
                                "original_price_confidence", ""
                            ),
                            "original_price_reasoning": row.get(
                                "original_price_reasoning", ""
                            ),
                            "original_price_llm_model": row.get("llm_model", ""),
                            "original_price_estimated_at_utc": row.get(
                                "estimated_at_utc", ""
                            ),
                            "original_price_error": "",
                        }
                    },
                )
            )
        else:
            ops.append(
                UpdateOne(
                    {"_id": source_id},
                    {
                        "$set": {
                            "original_price_error": row.get("estimate_error", ""),
                            "original_price_llm_model": row.get("llm_model", ""),
                            "original_price_estimated_at_utc": row.get(
                                "estimated_at_utc", ""
                            ),
                        }
                    },
                )
            )

    if not ops:
        return 0

    client = _build_mongo_client(uri)
    try:
        result = client[database][collection].bulk_write(ops, ordered=False)
        return (result.upserted_count or 0) + (result.modified_count or 0)
    finally:
        client.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0] if __doc__ else None
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"Max parsed docs missing original_price to process (default: {DEFAULT_LIMIT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent
        / "outputs"
        / "mongo_llm_original_price_estimates.jsonl",
        help="Path to write per-row estimates as JSONL.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=INTER_REQUEST_DELAY_SECONDS,
        help="Seconds to sleep between LLM requests.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip calling the LLM and Mongo writes; just report which docs would be processed.",
    )
    parser.add_argument(
        "--no-mongo-writeback",
        action="store_true",
        help="Compute and log estimates, but do not update the parsed Mongo collection.",
    )
    return parser.parse_args()


def main() -> int:
    load_dotenv()
    args = parse_args()

    api_key = os.getenv("NVIDIA_API_KEY", "").strip()
    model = os.getenv("NVIDIA_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    mongodb_uri = (os.getenv("MONGODB_URI") or "").strip()
    database = (
        os.getenv("MONGODB_DATABASE", DEFAULT_DATABASE).strip() or DEFAULT_DATABASE
    )
    parsed_collection = (
        os.getenv("MONGODB_PARSED_COLL", DEFAULT_PARSED_COLLECTION).strip()
        or DEFAULT_PARSED_COLLECTION
    )

    if not mongodb_uri:
        print("ERROR: MONGODB_URI is not set in the environment.", file=sys.stderr)
        return 2
    if not args.dry_run and not api_key:
        print("ERROR: NVIDIA_API_KEY is not set in the environment.", file=sys.stderr)
        return 2

    print(
        f"Connecting to MongoDB database '{database}', collection '{parsed_collection}'..."
    )
    try:
        documents = fetch_documents_missing_price(
            uri=mongodb_uri,
            database=database,
            collection=parsed_collection,
            limit=args.limit,
        )
    except PyMongoError as exc:
        print(f"ERROR: MongoDB request failed: {exc}", file=sys.stderr)
        return 3

    print(
        f"Pulled {len(documents)} parsed doc(s) with missing original_price."
    )
    if not documents:
        print("Nothing to estimate. Exiting.")
        return 0

    results: list[dict[str, Any]] = []

    for idx, document in enumerate(documents, start=1):
        source_id = document.get("_id")
        item_id = document.get("item_id", "")
        base_row: dict[str, Any] = {
            "source_id": source_id,
            "item_id": _json_safe(item_id),
            "llm_model": model,
            "estimated_at_utc": datetime.now(timezone.utc).isoformat(),
            "brand_name": document.get("brand_name", ""),
            "item_type": document.get("item_type", ""),
            "model_name": document.get("model_name", ""),
            "parsed_price": document.get("parsed_price", ""),
        }

        if args.dry_run:
            results.append({**base_row, "estimate_status": "dry_run"})
            print(
                f"  [{idx}/{len(documents)}] item_id={item_id} "
                f"brand='{base_row['brand_name']}' type='{base_row['item_type']}' (dry-run)"
            )
            continue

        try:
            messages = build_messages(document)
            payload = call_nvidia_chat(
                api_key=api_key, model=model, messages=messages
            )
            content = payload["choices"][0]["message"]["content"]
            estimate = parse_llm_json(content)
            row = {
                **base_row,
                **estimate,
                "llm_response_text": content,
                "estimate_status": "ok",
                "estimate_error": "",
            }
            print(
                f"  [{idx}/{len(documents)}] item_id={item_id} "
                f"brand='{base_row['brand_name']}' type='{base_row['item_type']}' "
                f"-> original_price=${estimate['original_price']} "
                f"(conf={estimate['original_price_confidence'] or 'n/a'})"
            )
        except Exception as exc:  # noqa: BLE001 - we want to log any failure
            row = {
                **base_row,
                "estimate_status": "error",
                "estimate_error": f"{type(exc).__name__}: {exc}",
            }
            print(
                f"  [{idx}/{len(documents)}] item_id={item_id} FAILED: {exc}",
                file=sys.stderr,
            )

        results.append(row)

        if idx < len(documents) and args.delay > 0:
            time.sleep(args.delay)

    # Serialize source_id (ObjectId) for the JSONL file only; keep raw form for Mongo writeback.
    jsonl_rows = [
        {**row, "source_id": str(row["source_id"]) if row.get("source_id") is not None else None}
        for row in results
    ]
    write_results(args.output, jsonl_rows)
    print(f"Wrote {len(jsonl_rows)} row(s) to {args.output}")

    if not args.dry_run and not args.no_mongo_writeback:
        try:
            written = update_parsed_with_estimates(
                uri=mongodb_uri,
                database=database,
                collection=parsed_collection,
                rows=results,
            )
            ok_count = sum(1 for r in results if r.get("estimate_status") == "ok")
            err_count = sum(1 for r in results if r.get("estimate_status") == "error")
            print(
                f"Updated {written} parsed doc(s) "
                f"({ok_count} ok, {err_count} error) in '{database}.{parsed_collection}'."
            )
        except PyMongoError as exc:
            print(
                f"WARNING: Failed to update parsed collection with estimates: {exc}",
                file=sys.stderr,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Quick verification script for the Vertex-backed pricing adapter."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from clothing_mlops.vertex_pricing import build_pricing_backend  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one pricing estimate through the configured backend.")
    parser.add_argument(
        "--description",
        default="Patagonia Synchilla fleece pullover in navy, men's medium, lightly worn with no stains or holes.",
        help="Free-text clothing description.",
    )
    parser.add_argument(
        "--retail-price",
        type=float,
        default=139.0,
        help="Retail price in USD.",
    )
    args = parser.parse_args()

    backend = build_pricing_backend()
    result = backend.estimate(args.description, args.retail_price)

    print(
        json.dumps(
            {
                "provider": result.provider,
                "model": result.model,
                "item_summary": result.item_summary,
                "retail_price": result.retail_price,
                "prices": {
                    "like_new": result.like_new,
                    "good": result.good,
                    "used": result.used,
                },
                "confidence_notes": result.confidence_notes,
                "warning": result.warning,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

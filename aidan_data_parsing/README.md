# aidan_data_parsing

This folder contains step 1 of the parsing pipeline. The script selects the
latest CSV export from `ebay_historical_clothing_scraper/data/exports/`,
validates the expected columns, applies basic cleaning, and writes a
parsing-ready intermediate CSV. LLM-based structured extraction is not called
yet.

Run it from the repository root with:

```bash
python aidan_data_parsing/parse_latest_exports_csv.py
```

The prepared output file is written to:

`aidan_data_parsing/outputs/`

Step 2 parses one pending prepared row at a time using NVIDIA-hosted LLM
inference. It reads `NVIDIA_API_KEY` from the repository root `.env`, selects
the latest `prepared_*.csv`, parses the first `pending` row, and writes the
result back into the same CSV.

Run it from the repository root with:

```bash
venv/bin/python aidan_data_parsing/parse_one_pending_row.py
```

To test row selection without calling the API:

```bash
venv/bin/python aidan_data_parsing/parse_one_pending_row.py --dry-run
```

If a previous attempt marked the row as `error` and you want to retry it:

```bash
venv/bin/python aidan_data_parsing/parse_one_pending_row.py --retry-errors
```

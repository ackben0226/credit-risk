# Credit Risk — Probability of Default

Production-oriented PD modelling system built on the Home Credit
Default Risk dataset.

## Documents

- [Problem Statement](docs/problem_statement.md) — v2.0
- [Model Card](docs/model_card.md) — v2.0
- [Data Quality Report](docs/data_quality_report.md) — populated after inspection

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Place Home Credit CSVs in data/raw/home_credit/

python scripts/inspect_data.py
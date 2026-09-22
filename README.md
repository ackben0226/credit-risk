# Credit Risk — Probability of Default (PD) System

A production-grade **Probability of Default** modelling system built on
the public Home Credit Default Risk dataset. The system estimates a loan
applicant's probability of default at application time, converts that 
probability into a business decision, explains the decision, and monitors
itself for drift.

This is not a notebook or a Kaggle submission. It is an end-to-end system
with the data contracts, calibration, explainability, monitoring, and
governance that a real credit risk deployment requires.

---

## Key Results

| Model | Holdout AUC | Gini | KS | Brier | Calibration slope |
|---|---:|---:|---:|---:|---:|
| Champion (Elastic-Net LR, WoE) | 0.7722 | 0.5443 | 0.4136 | 0.0671 | 1.005 |
| **Challenger (LightGBM, raw)** | **0.7846** | **0.5692** | **0.4308** | **0.0658** | 0.979 |

The challenger reduces expected cost by **~2%** across cost ratios
from 5:1 to 50:1 (see [`docs/champion_challenger.md`](docs/champion_challenger.md)).

The API serves **`/score`** at under 100 ms p99, with a full
explainability endpoint producing regulator-compliant adverse-action
notices.

---

## What the System Does
<p align="center">
  <img
    src="./architecture.drawio.svg"
    alt="Home Credit Probability of Default Modelling Pipeline"
    width="100%"
  />
</p>

<p align="center">
  <em>End-to-end production-oriented Probability of Default modelling architecture.</em>
</p>

### Interactive Diagram

[Open the editable diagrams.net architecture](./home_credit_pd_pipeline.drawio)


---

## Why This Project

Three reasons this is more than a classifier:

**1 — PD is not a classification problem.** The output must be a
*calibrated probability*, not a ranking. A model with strong AUC but poor
calibration is useless for downstream provisioning, pricing, or expected
loss. Calibration is treated as a first-class requirement, not an
afterthought.

**2 — The decision is not the model.** The threshold that converts PD
into APPROVE / DECLINE is a business parameter. It lives in config, not
in the model artifact. Cost parameters change without retraining.

**3 — Regulatory requirements shape the design.** Adverse-action reasons,
audit trails, monotonicity constraints, and calibration documentation
are not optional extras. They are why the system is built the way it is.

---

## Documentation

Start here depending on what you want:

| If you want to understand… | Read |
|---|---|
| **What problem is being solved and why** | [`docs/problem_statement.md`](docs/problem_statement.md) |
| **What the model is, how it's evaluated, its limitations** | [`docs/model_card.md`](docs/model_card.md) |
| **A one-document overview for non-technical readers** | [`docs/stakeholder_brief.md`](docs/stakeholder_brief.md) |
| **Champion vs challenger comparison with cost sweep** | [`docs/champion_challenger.md`](docs/champion_challenger.md) |
| **What we found in the raw data** | [`docs/data_quality_report.md`](docs/data_quality_report.md) |
| **Running log of design decisions** | [`docs/feature_design_notes.md`](docs/feature_design_notes.md) |

---

## Quick Start

### Install

```bash
git clone https://github.com/ackben0226/credit-risk
cd credit-risk
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```
### Get the data
Download the Home Credit Default Risk dataset from Kaggle and place the
CSVs in data/raw/home_credit/:

```bash
kaggle competitions download -c home-credit-default-risk
unzip home-credit-default-risk.zip -d data/raw/home_credit/
```
Expected files: ```application_train.csv```, ```application_test.csv```,
`bureau.csv`, `bureau_balance.csv`, `previous_application.csv`,
`POS_CASH_balance.csv`, ``installments_payments.csv``,
`credit_card_balance.csv`, `plus reference files`.

### Run the pipeline
Each stage is a runnable script. Run in order:

```bash
python scripts/inspect_data.py         # ~8 min — full-data reconnaissance
python scripts/build_contracts.py      # ~5 sec — generate schema contracts
python scripts/run_validation.py       # ~3 min — validate data against contracts
python scripts/run_ingestion.py        # ~3 min — CSV → Parquet
python scripts/run_bureau_aggregation.py             # ~40 sec
python scripts/run_previous_application_aggregation.py  # ~30 sec
python scripts/run_pos_cash_aggregation.py          # ~20 sec
python scripts/run_installments_aggregation.py      # ~25 sec
python scripts/run_credit_card_aggregation.py       # ~15 sec
python scripts/run_assembly.py         # ~1 min — join into single matrix
python scripts/run_splits.py           # ~10 sec — deterministic 70/15/15
python scripts/run_binning.py          # ~60 sec — WoE + IV for champion
python scripts/run_store.py            # ~10 sec — challenger feature store
python scripts/run_champion.py         # ~3 min — Elastic-Net LR grid search
python scripts/run_challenger.py       # ~25 min — LightGBM + Optuna (30 trials)
python scripts/run_calibration.py      # ~30 sec — fit isotonic on val
python scripts/run_evaluation.py       # ~30 sec — champion vs challenger report
python scripts/run_monitoring.py       # ~10 sec — drift detection
python scripts/run_decision_demo.py    # ~10 sec — decision distribution
python scripts/run_explainability.py   # ~30 sec — SHAP + reason codes
```

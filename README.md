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
    src="./architecture.drawio (1).svg"
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

Serve the API
```bash
python scripts/run_api_server.py
```
Then in another terminal:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/readiness
curl http://localhost:8000/model-info
```
Interactive docs: http://localhost:8000/docs

Example score request:

```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "applicant_id": "demo_001",
    "features": {"EXT_SOURCE_2": 0.72, "EXT_SOURCE_3": 0.68, ...},
    "segment": "default"
  }'
```
Response:

```json
{
  "applicant_id": "demo_001",
  "pd": 0.0066,
  "decision": "APPROVE",
  "threshold_used": 0.04,
  "segment": "default",
  "override_applied": null,
  "expected_cost": 0.131,
  "model": "challenger",
  "model_version": "0.1.0",
  "config_version": "1.0.0",
  "decided_at": "2026-09-22T01:17:16Z"
}
```

Run the tests

```bash
pytest tests/test_api.py -v
```
Expected: 7 passed.

Docker
```bash
docker build -t credit-risk-api:0.1.0 .
docker run --rm -p 8000:8000 credit-risk-api:0.1.0
```
Or with compose:

```bash
docker compose up
```
Project Structure
```text
credit_risk/
├── src/credit_risk/
│   ├── data/                       Ingestion, validation
│   │   ├── ingest.py
│   │   └── validate.py
│   ├── features/                   Feature engineering
│   │   ├── aggregations/           5 relational family aggregators
│   │   ├── assembly.py             Join aggregations onto application table
│   │   ├── binning.py              WoE + IV with monotonicity constraints
│   │   ├── splits.py               Deterministic 70/15/15
│   │   └── store.py                Challenger feature store
│   ├── models/                     Training and evaluation
│   │   ├── champion.py             Elastic-Net LR scorecard
│   │   ├── challenger.py           LightGBM + Optuna
│   │   ├── calibrate.py            Isotonic / Platt calibration
│   │   └── evaluate.py             Champion vs challenger comparison
│   ├── decision/                   Decision engine
│   │   └── engine.py               APPROVE / REFER / DECLINE
│   ├── explainability/             SHAP and reason codes
│   │   ├── shap_analysis.py
│   │   └── reason_codes.py
│   ├── monitoring/                 Drift detection
│   │   └── drift.py                PSI, KS, calibration drift
│   └── api/                        FastAPI service
│       ├── main.py
│       └── schemas.py
├── scripts/                        Runnable pipeline entry points
├── configs/
│   ├── contracts/                  Generated schema contracts (8 tables, 6 joins)
│   ├── data_config.yaml
│   ├── decision.yaml               Decision thresholds, policy overrides
│   ├── monitoring_config.yaml      Drift thresholds
│   └── reason_codes.yaml           Feature → reason code mapping
├── docs/
│   ├── problem_statement.md
│   ├── model_card.md
│   ├── data_quality_report.md
│   ├── champion_challenger.md
│   ├── stakeholder_brief.md
│   └── feature_design_notes.md
├── tests/
│   └── test_api.py                 7 API tests, all passing
├── data/                           (gitignored)
│   ├── raw/home_credit/
│   ├── interim/
│   └── processed/
├── artifacts/                      (gitignored)
│   ├── models/
│   ├── calibrators/
│   ├── binning/
│   └── reports/
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

## Design Highlights
### Data contracts, not assumptions
Every table and column has a declared contract in
`configs/contracts/`. 
<br/>Contracts are __generated__ from a full-data
inspection, not hand-typed. 
<br/>If the raw data changes, re-running
inspection produces a diff for review.

Validation runs on every ingestion. If the data doesn't match the
<br/>contract, ingestion fails loudly — no silent corruption.

### Leakage discipline, enforced
Every fitted transformation — imputation, binning, WoE encoding,
<br/>categorical factorization, calibration — is fitted on the development
<br/>split only. The holdout is touched exactly once, for final evaluation.

This is enforced by the pipeline structure, not by convention.

### Two models, one decision
The champion and challenger produce calibrated PD through different
paths:

- **Champion:** WoE-encoded features → Elastic-Net logistic regression.
<br/>Naturally calibrated. Interpretable coefficients. 180 features.

- **Challenger:** raw-encoded features → LightGBM. Needs post-hoc
<br/>isotonic calibration. 362 features, 1,486 trees.

Both feed the same decision engine. The comparison report recommends
<br/>the challenger as primary (better discrimination and cost), with the
<br/>champion retained for regulatory and latency-critical scenarios.

### Explainability that satisfies regulators
Three layers:

1. __Global SHAP__ for the challenger; coefficients for the champion.
2. __Local SHAP per applicant__ with signed contributions.
3. __Deterministic reason codes__ mapping features to human-readable
<br/>adverse-action reasons via configs/reason_codes.yaml.

The reason code layer is deliberately separate from SHAP — a
<br/>regulator-friendly mapping layer that produces stable, auditable output
<br/>independent of the underlying model.

Decision engine separated from model
The threshold is a business decision, not a model output. It lives in
<br/>`configs/decision.yaml:`

```yaml
costs:
  C_FN: 20.0
  C_FP: 1.0
thresholds:
  approve_max: 0.040
  review_max: 0.200
```
Change the cost parameters, change the threshold, no retraining required.

### Monitoring with the right reference population
Drift detection compares a held-out reference population (val
split) <br/>against a monitoring population (holdout). Using the training
<br/>split as reference would inflate AUC comparisons because training
<br/>performance reflects overfitting, not drift. This distinction is
<br/>documented and enforced.

## Technology Stack
|__Category__|	__Tools__|
|:----:|:---:|
|Language|	Python 3.11+|
|Data	|pandas, numpy, pyarrow|
|Modelling|	scikit-learn, LightGBM, Optuna|
|Binning|	optbinning|
|Explainability|	SHAP|
|API|	FastAPI, Uvicorn, Pydantic v2|
|Testing|	pytest|
|Containers|	Docker, docker-compose|
|Config|	PyYAML|

## Regulatory and Ethical Posture

The system is designed with the following regulatory requirements in
mind. This is a **design alignment**, not a compliance claim.

| Requirement | How addressed |
|---|---|
| **SR 11-7** (Fed model risk) | Model card, validation reports, champion/challenger governance |
| **ECOA / Reg B** (US adverse action) | Structured reason codes with plain-language descriptions |
| **GDPR Art. 22** (EU automated decisions) | Meaningful information about the logic; right to human review stated in notices |

### Out of Scope

- **Basel IRB capital calculations** and **IFRS 9 ECL staging** require
  LGD (Loss Given Default) and EAD (Exposure at Default) models in
  addition to PD. This project models PD only.
- The PD output is suitable as an input to those frameworks, but this
  project does not implement them.
- The Home Credit dataset does not contain the recovery or
  exposure-at-default data required to model LGD or EAD.
- The project does not claim regulatory approval in any jurisdiction.

See [`docs/model_card.md`](docs/model_card.md) §17 for the full
limitations statement.

## Not claimed:

- Regulatory approval in any jurisdiction
- Fitness for production deployment without independent validation
- That the Home Credit population represents any specific lending context
- That fairness compliance has been certified (the dataset does not
<br>contain the protected attributes a complete assessment requires)

See [docs/model_card.md](docs/model_card.md) &nbsp; §10 and §17 
<br>for the full limitations statement.

## What This Project Does Not Include
Explicitly out of scope:
- Loss Given Default (LGD) modelling
- Exposure at Default (EAD) modelling
- Risk-based pricing engine
- Collections strategy
- Fraud detection
- Macroeconomic scenario generation
- Full IFRS 9 lifetime ECL engine
- Full Basel capital calculation
- Production customer data
- Legal or regulatory certification

The architecture provides interfaces for these capabilities but does
<br>not implement them.

## Verification
Check	Result
| Schema validation on raw data |	0 errors |
|:----:|:----:|
|Champion holdout AUC|	0.7722|
|Challenger holdout AUC|	0.7846|
|API tests	|7 / 7 passing|
|End-to-end scoring via API|	Verified on live requests|
|Adverse-action notice rendering|	Verified on high-risk applicant|
|Policy override behaviour	|Verified (sanctions list → DECLINE)|

## License
The code in this repository is released under the MIT License. The Home
<br>Credit Default Risk dataset is subject to its own licensing terms; see
<br>the Kaggle competition page.

## Author
Benjamin Ackah — ML Engineering / Data Science
<br>GitHub: [@ackben0226](https://github.com/ackben0226)

_Built as a reference implementation of a production-grade credit risk
<br>PD system. Every quantitative claim in the documentation is traceable
<br>to an artifact on disk, and every pipeline stage is reproducible from
<br>the raw CSVs._

```text

---

## What This README Does

The README is the entry point. It should answer three questions in the
first 30 seconds:

1. **What is this?** A production-grade credit risk PD system.
2. **What does it do?** Estimate PD, decide, explain, monitor.
3. **How do I use it?** Install, run pipeline, serve API, run tests.

It should then let a deeper reader drill into any layer:

- **Non-technical:** stakeholder brief
- **Risk manager:** model card, champion/challenger comparison
- **ML engineer:** source code, design notes
- **Regulator:** model card, problem statement, data quality report
- **Reviewer:** tests, verification results, project structure

Every claim links to an artifact on disk. Every number is real.

---

## Small Notes

### The repository URL

I've included `https://github.com/ackben0226/credit-risk` in the clone
command and the author section. If you push the repo under a different
name, update both.

### The `pyproject.toml` reference

The README mentions `pip install -e ".[dev]"`. This requires a `[dev]`
extras section in `pyproject.toml`. If yours doesn't have one yet, add:

```toml
[project.optional-dependencies]
dev = [
    "pytest>=8.3",
    "httpx>=0.27",
    "ruff>=0.7",
    "mypy>=1.12",
]
```
Or change the README to `pip install -e .` and add the dev tools
<br>separately.

### The dataset download command
The `kaggle competitions download` command requires the Kaggle CLI
<br>and API credentials `(~/.kaggle/kaggle.json)`. If a reader doesn't
<br>have that configured, the manual download from the Kaggle website
<br>works too. The README says "Download from Kaggle" but only shows the
<br>CLI form; consider adding a note about the manual option.

### Commit and Push
```powershell
cd C:\credit_risk

git add README.md
git commit -m "Rewrite README with full project overview

- Lead with key results (champion/challenger AUC, cost impact)
- Explain what the system does and why it's not a classifier
- Link to all documentation
- Quick start: install, run pipeline, serve API, test
- Full project structure
- Design highlights: contracts, leakage, two-model architecture,
  explainability, decision engine, monitoring
- Technology stack
- Regulatory posture (design alignment, not compliance claim)
- Out-of-scope list
- Verification results"
git push
```

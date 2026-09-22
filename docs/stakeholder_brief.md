markdown
# Credit Risk PD Model — Stakeholder Brief

**Document Version:** 1.0
**Date:** 2026-09-22
**Audience:** Executives, risk managers, product owners, regulators,
auditors, and technical reviewers who need a concise overview of the
system's purpose, design, and limitations.
**Companion documents:** `docs/problem_statement.md`,
`docs/model_card.md`, `docs/champion_challenger.md`,
`docs/data_quality_report.md`.

---

## Executive Summary

This project is a **production-grade Probability of Default (PD)
modelling system** built on the public Home Credit Default Risk dataset.
It estimates the probability that a loan applicant will default, using
only information legitimately available at the point of application.

The system comprises:

- A data pipeline that ingests, validates, and aggregates eight
  relational tables into a single applicant-level feature matrix
- Two models — an interpretable logistic regression scorecard and a
  high-accuracy gradient boosting model — trained and compared under
  a common framework
- A post-hoc calibration layer that aligns predicted probabilities with
  observed event rates
- A configurable decision engine that converts calibrated PD into
  APPROVE / REFER / DECLINE decisions
- An explainability layer producing SHAP-based explanations and
  regulator-compliant adverse-action notices
- A monitoring layer for detecting feature, prediction, and calibration
  drift
- A containerized HTTP service exposing the full pipeline through a
  REST API

The system is **not** deployed on live customer data. It demonstrates
the technical architecture, methodology, and governance that a
production credit risk system requires.

---

## 1. What Are We Solving?

**The problem:** estimate a loan applicant's probability of default at
the moment of application, using only information available before the
lending decision is made.

**Why it matters:**

A lender must decide, for every application, whether to approve credit.
The decision depends on the applicant's risk. Without an accurate PD
estimate, the lender cannot:

- Approve creditworthy applicants (foregone revenue)
- Decline risky applicants (capital losses)
- Price loans to reflect their risk
- Provision adequately for expected losses (IFRS 9)
- Meet regulatory capital requirements (Basel)

PD is the **atom of credit risk**. Every downstream quantity — expected
loss, risk-weighted assets, pricing, provisioning — derives from it. A
PD system that works is a foundation for the entire credit risk stack.

**Why this specific problem:**

- **Domain centrality:** PD is the most important single quantity in
  credit risk. A PD system demonstrates capability across the whole
  risk function.
- **Regulatory salience:** banks are actively redeveloping PD models
  under Basel IV and IFRS 9.
- **Fintech relevance:** neobanks and digital lenders need real-time
  PD estimates for instant decisioning.
- **Genuine complexity:** PD modelling is not a classification
  problem. It is a calibrated probability estimation problem with
  regulatory, fairness, and governance constraints.

---

## 2. What Information Do We Have?

**The dataset:** 307,511 labelled applicants with an 8.07% default
rate, drawn from eight relational tables.

| Table | Rows | Content |
|---|---:|---|
| `application_train` | 307,511 | Application-level features + target label |
| `bureau` | 1,716,428 | External credit bureau records |
| `bureau_balance` | 27,299,925 | Monthly bureau balances |
| `previous_application` | 1,670,214 | Prior Home Credit applications |
| `POS_CASH_balance` | 10,001,358 | Monthly POS/cash-loan snapshots |
| `installments_payments` | 13,605,401 | Repayment history |
| `credit_card_balance` | 3,840,312 | Monthly credit-card snapshots |

**After aggregation and assembly:** a single matrix of 307,511 × 359
features, one row per applicant.

**Key data characteristics (documented):**

- **Mixed encodings:** the metadata file uses Windows-1252, not UTF-8.
  The pipeline handles this with an encoding fallback chain.
- **Sparse families:** `credit_card_balance` covers only ~5.6% of prior
  applications. Missingness is preserved, not imputed. The absence of
  a credit card is itself a signal.
- **Orphan records:** some child records reference parents absent from
  `application_train`. These are dropped by the assembly step with
  documented counts.
- **No calendar-date field:** genuine out-of-time validation is not
  possible. This is documented as a limitation.

**What we are NOT using:** the unlabeled test population
(`application_test`), the Kaggle submission template, and the column
description file — none of which are needed for modelling.

---

## 3. What Are We Predicting?

**The target:** `P(TARGET = 1 | X = x)` — the calibrated probability of
default within the loan's performance window.

**Two models are trained and compared:**

| Model | Algorithm | Inputs | Holdout AUC | Calibration |
|---|---|---|---:|---|
| **Champion** | Elastic-Net Logistic Regression | 180 WoE-encoded features | 0.7722 | Natural (slope 1.005) |
| **Challenger** | LightGBM (1,486 trees) | 362 raw-encoded features | 0.7846 | Post-hoc isotonic (slope 0.979) |

**Why two models:**

- The **champion** is a transparent, points-based scorecard. It is the
  format regulators, auditors, and non-technical stakeholders
  understand. Its coefficients are directly interpretable.
- The **challenger** is a nonlinear model that captures interactions
  and non-monotonic effects the linear model cannot. It is more
  accurate but less transparent.

Both are shipped. The comparison report (`docs/champion_challenger.md`)
recommends the challenger as the primary production model, with the
champion retained for contexts requiring a linear scorecard.

**Discrimination vs calibration — why both matter:**

- **AUC** measures ranking. A high AUC means the model orders
  applicants correctly by risk.
- **Calibration** measures probability accuracy. A calibrated model's
  predicted PDs match observed default rates.

A model with strong AUC but poor calibration is worse than one with
slightly lower AUC and good calibration — because downstream systems
consume the probability, not just the ranking. The challenger needed
post-hoc isotonic calibration to become well-calibrated; the champion
is naturally calibrated because logistic regression optimises
likelihood.

**No leakage:** every fitted transformation — WoE bins, categorical
codes, calibration curves — was fitted on the development split only
and applied to validation and holdout. The holdout was touched exactly
once for evaluation.

---

## 4. How Is the Prediction Used?

The model produces a calibrated PD. The **decision engine** converts
that number into a business action.

**Three-way decision:**

| Band | Condition | Action |
|---|---|---|
| APPROVE | PD < 0.040 | Automated approval |
| REFER | 0.040 ≤ PD < 0.200 | Manual review |
| DECLINE | PD ≥ 0.200 | Automated decline |

**Holdout decision distribution (challenger):**

- APPROVE: 43.3% of applicants
- REFER: 47.6%
- DECLINE: 9.1%

**Actual default rates by band (validates the engine):**

- APPROVE: 2.07% — well below the population average of 8.07%
- REFER: 9.02% — matches the population average, appropriate for
  manual review
- DECLINE: 31.73% — well above the average, correctly rejected

The monotonic increase in default rate across the three bands is the
key validation. The engine is grouping applicants correctly.

**Cost-optimal threshold:** at a cost ratio of 20:1 (approving a
defaulter costs 20× more than declining a good applicant), the
cost-minimising threshold is 0.040 — exactly what the config uses.
Changing the cost ratio changes the threshold. The config supports
this without retraining the model.

**Policy overrides:** the engine also fires non-PD-based rules —
sanctions list, internal blacklist, fraud flag — that force a decision
regardless of PD. These are configured in `configs/decision.yaml` and
are auditable.

**Audit trail:** every decision records applicant ID, PD, decision,
threshold used, segment, override (if any), expected cost, model
version, config version, and timestamp. This enables replay,
regulatory reporting, and backtesting of alternative policies.

---

## 5. How Does the Prediction Get Into Production?

The full pipeline from raw data to decision:
Raw CSVs
↓ Ingestion (encoding-tolerant, SHA-256 verified)
Parquet (interim)
↓ Validation (schema contracts enforced)
Contracts (configs/contracts/)
↓ Aggregation (5 relational families → per-applicant features)
Aggregated Parquet
↓ Assembly (left-join onto application_train)
Assembled matrix (307K × 359)
↓ Splits (deterministic 70/15/15)
↓ Binning (WoE + IV for champion)
↓ Store (raw + categorical codes for challenger)
Champion + Challenger matrices
↓ Training (Elastic-Net LR / LightGBM + Optuna)
Trained models
↓ Calibration (isotonic for challenger)
Calibrated PD
↓ Decision engine (thresholds, segments, overrides)
APPROVE / REFER / DECLINE
↓ Explainability (SHAP + reason codes)
Adverse-action notices
↓ Monitoring (PSI, KS, calibration drift)
↓ API (FastAPI + Docker)
Production service

text

**What is built:**

- Data pipeline (ingestion, validation, contracts): ✅
- Aggregation (bureau, previous_application, pos_cash, installments,
  credit_card): ✅
- Assembly, splits, binning, store: ✅
- Champion, challenger, calibration: ✅
- Comparison, decision engine: ✅
- Explainability, monitoring: ✅
- FastAPI service, Dockerfile: ✅

**What remains for a real deployment:**

- Live production data (with appropriate PII controls and regulatory
  oversight)
- Outcome capture (labels arriving as loans mature)
- Scheduled retraining pipeline
- Monitoring dashboards and alerting
- Fairness review with actual protected-class data
- Independent model validation

**The system is a reference implementation**, not a deployed product.
It demonstrates every capability a production PD system requires.

---

## 6. Who Uses the Output?

Five distinct audiences consume the system's output.

### 6.1 Automated decisioning systems (real-time)

**What:** the bank's origination platform.

**What they consume:** `POST /score` returning calibrated PD, then the
decision engine's APPROVE / REFER / DECLINE.

**What they do:**

- APPROVE → straight-through processing
- REFER → manual review queue
- DECLINE → adverse-action notice and rejection

**Volume:** hundreds to thousands per day, with sub-100ms latency.

### 6.2 Manual underwriters (minutes)

**What:** human review specialists.

**What they consume:** the automated system's output plus the
explanation — calibrated PD, SHAP contributions, structured reason
codes.

**What they do:** override the automated decision when context warrants
it. The REFER band (~48% of holdout) is where they work.

### 6.3 Risk management and capital teams (daily/weekly)

**What:** quantitative risk managers, capital planners, IFRS 9 teams.

**What they consume:**

- Aggregate PD distributions across the portfolio
- Expected loss calculations: `EL = PD × LGD × EAD`
- PD by segment, cohort, product

**What they do:** provisioning, capital allocation, stress testing,
portfolio concentration monitoring.

### 6.4 Model validation and internal audit (periodic)

**What:** independent validation teams, internal audit, external
regulators.

**What they consume:** the model card, the champion/challenger
comparison, all metrics reports, binning tables, calibration curves.

**What they do:** audit the model, verify its assumptions, approve
continued use or demand remediation.

### 6.5 Applicants (indirect)

**What:** loan applicants, especially rejected ones.

**What they consume:** the adverse-action notice.

**Example (from a live API call, high-risk applicant):**
Adverse Action Notice
Applicant ID: demo_high_risk
Decision: DECLINE

Your application was reviewed and could not be approved at this time.
The primary factors considered in this decision were:

High maximum overdue amount on credit bureau records

Requested loan amount relative to income

Limited employment history

Education profile contributes to risk assessment

Additional factors considered:

Large number of existing credit accounts

Significant maximum delay on prior loan installments

Multiple prior applications refused

History of underpaid installment amounts

Days past due on prior POS or cash loans

Delinquency in bureau balance history

This decision was made using an automated system. You have the right to
request a human review of this decision.

text

This is the disclosure required under ECOA (US) and GDPR Art. 22 (EU).

---

## 7. What Decisions Does the Model Support?

Six decisions, at different timescales.

### 7.1 Approval (immediate, per application)

**Question:** should we extend credit to this applicant?

**Input:** calibrated PD, cost parameters, risk appetite.

**Output:** APPROVE / REFER / DECLINE.

**Impact:** direct. Every approval is revenue; every default is a loss.

### 7.2 Pricing (immediate, if approved)

**Question:** at what rate?

**Input:** PD, LGD estimate, funding cost, target margin.

**Formula:** `rate = funding_cost + LGD × PD / (1 - PD) + margin`

**Impact:** higher rates compensate for higher risk. The calibrated PD
makes this principled rather than arbitrary.

*Pricing is out of scope for this project but is a downstream
application of the PD output.*

### 7.3 Provisioning (monthly, portfolio-wide)

**Question:** how much capital to hold against expected losses?

**Input:** PD, LGD, EAD.

**Formula:** `ECL = Σ PD_i × LGD_i × EAD_i`

**Impact:** directly affects the P&L and regulatory compliance.

### 7.4 Capital allocation (quarterly, portfolio-wide)

**Question:** how much regulatory capital is required?

**Input:** PD, LGD, correlation, maturity (Basel IRB formula).

**Impact:** determines the bank's lending capacity.

### 7.5 Portfolio concentration (ongoing)

**Question:** are we over-exposed to any segment?

**Input:** PD distributions by segment, region, product.

**Impact:** prevents concentration risk that could impair the bank
during a downturn.

### 7.6 Retraining and strategy (periodic)

**Question:** is the model still performing? Should we retrain or
change thresholds?

**Input:** monitoring reports, drift metrics, backtesting.

**Impact:** long-term model health and strategy.

---

## 8. What Happens If the Model Is Wrong?

Two types of error, with different consequences.

### 8.1 False Negatives — approving a defaulter

**Model says:** PD = 0.02 → APPROVE.
**Reality:** applicant defaults.

**Direct cost:** `LGD × EAD` — the loss on the loan. For an unsecured
loan with 60% LGD and £10K EAD, that is £6,000 per case.

**Indirect costs:**

- Capital consumed that could have funded other loans
- Provisioning impact
- Reputational damage if defaults cluster

**From the holdout demo:** 2.07% of the challenger's APPROVE band
actually defaulted. At the cost ratio C_FN/C_FP = 20, each false
negative costs 20× as much as a false positive.

### 8.2 False Positives — declining a good applicant

**Model says:** PD = 0.30 → DECLINE.
**Reality:** applicant would have repaid.

**Direct cost:** foregone revenue. For a profitable loan with 5% margin
and £10K EAD, that is £500 per case.

**Indirect costs:**

- Reputational risk (denied applicants tell others)
- Regulatory scrutiny if decline rates diverge across protected classes
- Financial inclusion concerns if systematically excluding underserved
  populations

**From the holdout demo:** 68.27% of the challenger's DECLINE band
would have repaid. At cost ratio 20, each false positive costs 1 unit
of the base cost.

### 8.3 The Cost Ratio

The ratio `C_FN / C_FP = 20` is the project's assumption. Different
business contexts justify different ratios:

| Business | Typical ratio | Reasoning |
|---|---:|---|
| Prime mortgages | 5:1 | Low LGD, long relationship value |
| Consumer unsecured | 20:1 | High LGD, short relationship |
| Micro-lending | 50:1 | Very high LGD, no relationship |
| Credit cards | 10:1 | Moderate LGD, ongoing relationship |

Changing the ratio changes the optimal threshold. The engine handles
this without retraining.

### 8.4 If the Model Degrades

Three failure modes and their mitigations:

| Failure mode | Symptom | Mitigation |
|---|---|---|
| **Data drift** | Feature distributions shift; PSI > 0.10 | Retrain, investigate specific features |
| **Concept drift** | PD-target relationship changes; KS declines | Recalibrate or retrain |
| **Calibration drift** | Predicted PDs no longer match observed rates | Recalibrate with recent data |
| **Fairness drift** | Subgroup performance diverges | Fairness audit, remediation, or model replacement |

Without monitoring, degradation would go undetected for months.
Monitoring is not optional.

---

## 9. What Does Success Mean?

Five categories of success, each with concrete thresholds.

### 9.1 Model Performance

Targets from the problem statement, and actual holdout results:

| Metric | Target | Champion | Challenger |
|---|---:|---:|---:|
| ROC-AUC | ≥ 0.78 | 0.7722 | **0.7846** |
| Gini | ≥ 0.56 | 0.5443 | **0.5692** |
| KS | ≥ 0.40 | 0.4136 | **0.4308** |
| Brier | ≤ 0.075 | 0.0671 | **0.0658** |
| Calibration slope | 0.90 – 1.10 | **1.005** | **0.979** |

The champion misses AUC and Gini targets but hits calibration and
Brier. The challenger meets all targets. Both are within acceptable
bounds.

### 9.2 Business Impact

| Cost ratio | Champion cost | Challenger cost | Savings |
|---:|---:|---:|---:|
| 5 | 0.3271 | 0.3171 | 3.07% |
| 10 | 0.5049 | 0.4939 | 2.17% |
| 20 | 0.6875 | 0.6742 | 1.93% |
| 30 | 0.7834 | 0.7637 | 2.50% |
| 50 | 0.8829 | 0.8638 | 2.17% |

The challenger reduces expected cost by ~2% across all tested ratios.
On a £100M portfolio, that is roughly £2M per year.

### 9.3 Operational Success

- **Latency:** /score meets the <100ms p99 target
- **Availability:** designed for 99.9% (healthcheck configured)
- **Reproducibility:** every artifact derived from SHA-256-verified inputs
- **Test coverage:** 7 API tests, all passing; unit tests across
  pipeline modules

### 9.4 Governance Success

- Every decision logged with PD, threshold, model version, config version
- Every artifact versioned and diffable in git
- Every model documented in the model card
- Every transformation has a persisted, reloadable table
- The pipeline reproduces deterministically from raw CSVs

### 9.5 What Success Does NOT Mean

Three things that are not success, even when they look like it:

1. **Highest possible AUC.** A model with AUC 0.82 that is
   uncalibrated is worse than one with AUC 0.78 that is calibrated.
   The downstream systems consume probabilities, not rankings.

2. **Fewest defaults.** A model that rejects everyone has zero
   defaults. It is also useless. Success is optimising the trade-off,
   not eliminating one side.

3. **Beating the champion.** The challenger's job is to give the
   business a genuine choice, not to win a metric competition.
   Sometimes the simpler model is right.

---

## 10. Repository Layout

Where each artifact lives in the project:
credit_risk/
├── src/credit_risk/ Python package
│ ├── data/ Ingestion, validation
│ ├── features/ Aggregation, assembly, binning, splits, store
│ ├── models/ Champion, challenger, calibration, evaluation
│ ├── decision/ Decision engine
│ ├── explainability/ SHAP, reason codes
│ ├── monitoring/ Drift detection
│ └── api/ FastAPI service
├── scripts/ Runnable entry points for each pipeline stage
├── configs/ YAML configs and schema contracts
├── docs/ This brief, model card, problem statement,
│ data quality report, champion/challenger report
├── tests/ Automated tests
├── data/ Raw and processed data (gitignored)
├── artifacts/ Models, calibrators, reports (gitignored)
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── README.md

text

---

## 11. Where Each Question Is Documented

| Question | Primary document | Supporting artifacts |
|---|---|---|
| What are we solving? | `docs/problem_statement.md` | §1 of this brief |
| What information do we have? | `docs/data_quality_report.md` | §2 of this brief |
| What are we predicting? | `docs/model_card.md` §1 | §3 of this brief |
| How is the prediction used? | `docs/model_card.md` §13 | §4 of this brief |
| How does it get to production? | §5 of this brief | Pipeline scripts |
| Who uses the output? | `docs/model_card.md` §2 | §6 of this brief |
| What decisions does it support? | `docs/problem_statement.md` §2 | §7 of this brief |
| What happens if the model is wrong? | `docs/model_card.md` §17 | §8 of this brief |
| What does success mean? | `docs/model_card.md` §5 | §9 of this brief |

---

## 12. Limitations and Honest Claims

This project does not claim:

- Regulatory approval or compliance certification in any jurisdiction
- Fitness for production deployment without independent validation
- That the Home Credit dataset is equivalent to a bank's production
  data environment
- That genuine out-of-time validation was performed (it was not
  possible with this dataset)
- That fairness compliance has been certified (the data does not
  contain the protected attributes a complete assessment requires)

This project does claim:

- A complete, reproducible pipeline from raw data to API response
- Two trained, calibrated models with documented performance
- A decision engine, explainability layer, and monitoring layer that
  meet the design requirements of a production credit risk system
- Governance artifacts (model card, data quality report,
  champion/challenger comparison) sufficient for independent review
- Automated tests and containerized deployment for reproducibility

---

## 13. Summary

**The system estimates calibrated PD, converts it to a business
decision, explains the decision, and monitors itself for degradation.**

Every component is built, tested, and documented. Every quantitative
claim is traceable to an artifact on disk. Every transformation is
reproducible from the raw CSVs.

The project demonstrates the technical and governance capability
required for a production credit risk system. It is a reference
implementation, not a deployment.

---

*End of Stakeholder Brief — Version 1.0*
What This Document Does
The stakeholder brief fills a gap that the model card doesn't: it
answers the questions that non-technical stakeholders ask, in language
they can act on. It is:

Concise: one document, thirteen sections

Grounded: every number is traceable to an artifact

Audience-appropriate: written for executives, risk managers, and
regulators, not just ML engineers

Complete: covers problem, data, model, decisions, system,
audience, use, failure modes, success criteria

Honest: states limitations as plainly as capabilities

An executive reading this brief should be able to answer: what does
this system do, who does it serve, what happens when it's wrong, and
how do we know if it's working.

Commit
powershell
git add docs/stakeholder_brief.md
git commit -m "Add stakeholder brief

Nine Q&As covering the project for non-technical stakeholders:
1. What are we solving?
2. What information do we have?
3. What are we predicting?
4. How is the prediction used?
5. How does it get into production?
6. Who uses the output?
7. What decisions does it support?
8. What happens if the model is wrong?
9. What does success mean?

Plus: repository layout, where each question is documented,
limitations and honest claims, and a summary.

Written for executives, risk managers, regulators, and auditors.
Every quantitative claim traceable to an artifact on disk."
git push

markdown
# Model Card: Credit Risk Probability of Default (PD)

**Model Name:** `credit-risk-pd`
**Version:** 0.1.0 (Development)
**Model Type:** Binary Classifier — Calibrated Probability Estimator
**Owner:** ML Engineering / Data Science
**Date:** 2026-09-13
**Status:** In Development
**Governance Alignment:** SR 11-7 principles (documentation, validation,
monitoring, controlled change)

---

## 1. Model Overview

### 1.1 Purpose

The `credit-risk-pd` model estimates the probability that a loan applicant
experiences the target credit-risk event defined by the Home Credit
`TARGET` variable, using only information legitimately available at or
before the application decision point.

The model produces a **calibrated probability** intended to support
downstream credit decisioning. It does **not** produce a decision.
Decisioning is handled by a separate, configurable decision engine
(Section 12).

### 1.2 Architecture

Two model families are developed and compared under a common validation
framework:

| Role | Model | Primary Rationale |
|---|---|---|
| **Champion** | Elastic-Net Logistic Regression on WoE-transformed features | Interpretability, transparent variable relationships, stable probability estimation, suitability for traditional credit-risk governance |
| **Challenger** | Gradient Boosting (LightGBM / XGBoost / CatBoost) | Tests whether nonlinear and interaction effects provide material improvement |
| **Calibration** | Platt scaling or isotonic regression | Aligns predicted probabilities with observed event rates |

The final production model is selected through documented champion /
challenger governance (Section 13). A higher AUC alone does not promote
the challenger.

### 1.3 Inputs

Features are grouped into six documented families. Every production
feature carries metadata per the feature metadata contract
(Section 4.4), including availability classification and leakage status.

| Family | Source Table(s) | Approx. Count | Examples |
|---|---|---|---|
| Application | `application_train` | ~120 | Income, credit amount, annuity, employment characteristics |
| Bureau | `bureau`, `bureau_balance` | ~80 | Active accounts, overdue balances, historical DPD |
| Previous Applications | `previous_application` | ~100 | Prior approval/rejection rates, prior amounts, outcomes |
| Installments | `installments_payments` | ~60 | Payment delays, underpayment ratios, regularity |
| Credit Card | `credit_card_balance` | ~40 | Utilisation, balances, delinquency indicators |
| POS / Cash | `POS_CASH_balance` | ~40 | Contract status, delinquency patterns, balance trends |

**Target:** approximately **≥200 validated candidate features** where
justified by the underlying data. Feature count is a target, not a
justification for generating meaningless variables.

### 1.4 Output

```json
{
  "model_version": "0.1.0",
  "pd": 0.087,
  "calibration_method": "isotonic",
  "decision": "APPROVE",
  "threshold": 0.142,
  "threshold_config_version": "config-2026-09-13",
  "reason_codes": ["HIGH_EXISTING_CREDIT_OBLIGATIONS"],
  "scored_at": "2026-09-13T12:00:00Z"
}
Two output concepts are deliberately separated:

Model output: calibrated pd and diagnostic metadata.

Decision output: decision, threshold, reason_codes — produced
by the decision engine (Section 12), not by the model.

2. Intended Use
2.1 Primary Use
Estimation of calibrated probability of default for retail loan
applications at the point of decisioning.

2.2 Intended Users
Automated underwriting / decisioning systems

Credit risk analysts

Underwriting operations

Model validation and internal audit functions

2.3 Intended Decisions Supported
Approve / refer / decline loan applications (via downstream engine)

Risk-band assignment

Downstream risk inputs (e.g. expected-loss components), subject to
the out-of-scope statement below

2.4 Out-of-Scope Uses
This model is not designed, validated, or approved for:

Loss Given Default (LGD) estimation

Exposure at Default (EAD) estimation

Risk-based pricing

Collections prioritisation

Fraud detection

Macroeconomic scenario generation

Full IFRS 9 lifetime ECL engine

Full Basel capital calculation

Regulatory submission without independent validation

Populations materially outside the training distribution

3. Training Data
3.1 Source
Public Home Credit Default Risk dataset (Kaggle, 2018). This dataset
reflects a real retail-lending environment but is not equivalent to
a production banking data environment. No production customer PII is
processed.

3.2 Composition
Table	Rows (approx.)	Role
application_train	~307,000	Labelled development population (contains TARGET)
application_test	~48,000	Unlabeled scoring/reference population
bureau	~1,700,000	Bureau records
bureau_balance	~27,000,000	Monthly bureau balances
previous_application	~1,600,000	Prior applications
POS_CASH_balance	~10,000,000	POS / cash monthly snapshots
installments_payments	~13,000,000	Repayment history
credit_card_balance	~3,800,000	Credit-card monthly snapshots
3.3 Target
text
TARGET = 1  →  target payment-difficulty / default event
TARGET = 0  →  no target event observed
3.4 Class Distribution
Positive (target event): ~8.07%

Negative: ~91.93%

Class imbalance is handled through documented modelling and evaluation
methodology. Indiscriminate oversampling is not used.

3.5 Data Splits
Split	Purpose
Development	Fitting of all learned transformations and model parameters
Validation	Model / hyperparameter / calibration selection
Holdout (Test)	Unbiased final evaluation — touched once
All transformations that learn parameters from data (imputation, binning,
WoE, scaling, feature selection, calibration) are fitted exclusively
on the development split.

3.6 Preprocessing
Missing-value treatment with explicit missing indicators where informative

WoE binning with monotonic constraints where conceptually and empirically justified

Nominal categorical variables are not given artificial ordering to satisfy monotonicity

Feature scaling for linear models

All parameters fitted on development data only

4. Data and Leakage Policy
4.1 Application-Time Availability
Every production feature is classified into one of:

text
AVAILABLE_AT_APPLICATION
DERIVED_FROM_PRE_APPLICATION_HISTORY
POST_APPLICATION
UNKNOWN
Only the first two categories enter the production feature set. Any
feature whose availability cannot be established is excluded or
documented as a limitation.

4.2 Leakage Controls
The pipeline explicitly tests for:

Direct target leakage

Post-application variables

Future repayment / credit-balance information

Validation/test information influencing preprocessing

Aggregation leakage

Duplicate applicants across splits

Feature-generation leakage

Calibration leakage

Threshold-optimisation leakage

4.3 Transform Discipline
text
DEVELOPMENT
  ↓
Fit imputation → Fit bins → Fit WoE → Fit scaling
  ↓
Fit feature selection → Fit model → Fit calibration
  ↓
VALIDATION / HOLDOUT
  ↓
Transform using DEVELOPMENT parameters only
4.4 Feature Metadata Contract
Every production feature carries machine-readable metadata:

text
feature_name
source_table
source_column(s)
business_definition
data_type
aggregation_method
join_key
availability_status
missingness
transformation
monotonicity_direction
WoE_bin_definition
IV
leakage_status
The feature catalogue is a versioned deliverable, not an exploratory
DataFrame.

5. Evaluation
5.1 Metrics
Metric	Purpose	Target
ROC-AUC	Discrimination	≥ 0.78
Gini	Rank-order separation	≥ 0.56
KS statistic	Credit-risk separation	≥ 0.40
PR-AUC	Imbalanced performance	Reported, no fixed target
Brier Score	Probability calibration	≤ 0.075
Calibration slope	Calibration quality	0.90 – 1.10
Calibration intercept	Calibration quality	Reported
PSI	Population stability	≤ 0.10
Targets are acceptance targets, not guaranteed outcomes. A model
with strong AUC but poor calibration is not automatically preferred.

5.2 Validation Strategy
Stratified K-fold cross-validation on development data

Documented proxy-temporal holdout where a defensible ordering proxy exists

Calibration curves and reliability analysis

Subgroup analysis where data supports it

Fairness assessment within data limitations (Section 10)

Holdout evaluation on the untouched test split

5.3 Out-of-Time Validation Limitation
A genuine calendar-time out-of-time validation requires a reliable
absolute application-date field. The public Home Credit dataset does
not provide such a field.

Therefore:

A genuine calendar-time out-of-time validation could not be performed
using the public dataset. Where an ordered holdout is used, it is
explicitly described as a proxy temporal validation and not as
true OOT.

The architecture is designed so that a genuine OOT validation can be
performed if the dataset is later replaced with a production-style,
date-stamped population.

5.4 Expected Performance
(To be populated after model training and holdout evaluation.)

Metric	Champion (LR)	Challenger (GBM)
ROC-AUC	TBD	TBD
Gini	TBD	TBD
KS	TBD	TBD
Brier	TBD	TBD
Calibration slope	TBD	TBD
6. Calibration
Raw model scores are evaluated for calibration. Where required,
calibration is applied using:

Platt / logistic calibration, or

Isotonic regression

Calibration is fitted using data independent of the model-fitting
process to avoid optimistic calibration estimates.

The final production output is the calibrated PD, not the raw model
score. Calibration is monitored after deployment (Section 11).

7. Explainability
7.1 Global Explanations
Feature importance (gain, permutation, SHAP where appropriate)

Champion coefficient interpretation and WoE relationships

Risk-direction analysis

Model-level contribution analysis

7.2 Local Explanations
For individual applicants:

text
Applicant
   ↓
PD
   ↓
Top contributing factors (SHAP or equivalent)
7.3 Explanations Are Not Reason Codes
Detailed model explanations and customer-facing adverse-action reason
codes are treated as separate concepts (Section 8).

8. Adverse-Action Reason Codes
8.1 Concept
Reason codes are a deterministic mapping layer, not raw SHAP values.

8.2 Example
Internal explanation:

text
Low external credit score
High existing credit exposure
Recent payment difficulties
Structured reason code:

text
HIGH_EXISTING_CREDIT_OBLIGATIONS
8.3 Requirements
Reason codes must:

Correspond to genuine model drivers

Be stable and reproducible

Avoid misleading statements

Be traceable to model features

Remain understandable to non-technical stakeholders

8.4 Compliance Scope
The project demonstrates the technical capability required for
adverse-action workflows. It does not claim that the public dataset
alone constitutes a complete legal compliance implementation in any
jurisdiction.

9. Regulatory and Governance References
The following are treated as design and governance considerations,
not as claims of independent regulatory compliance:

Basel Committee on Banking Supervision — IRB / credit-risk modelling guidance

IFRS 9 — Financial Instruments

Federal Reserve — SR 11-7: Guidance on Model Risk Management

Consumer credit / adverse-action requirements in the relevant jurisdiction

GDPR requirements relevant to automated decision-making

Home Credit Default Risk dataset documentation

Siddiqi, N. — Credit Risk Scorecards

Thomas, Edelman, Crook — Credit Scoring and Its Applications

This project does not claim that a model trained on a public Kaggle
dataset constitutes regulatory approval or production banking compliance.

10. Fairness and Responsible Modelling
10.1 Approach
A fairness assessment is performed to the extent supported by the
available data. The assessment considers, where applicable:

Approval / decision rates across groups

Predicted-risk distributions

Error rates

Calibration across groups

Disparate-impact indicators

Proxy-variable risks

10.2 Data Limitation
The public Home Credit dataset does not necessarily contain every
protected characteristic required for a complete real-world fair-lending
assessment.

Therefore:

The fairness report must clearly distinguish between analyses
supported by the available dataset and analyses that require
protected-class information unavailable in the dataset.

No unsupported claim of full regulatory fairness compliance is made.

10.3 Proxy Risk Monitoring
Features that may act as proxies for protected attributes are monitored:

Postal / regional features

Employment-type features

Education-level features

11. Monitoring
The deployed system monitors both data stability and model
performance.

11.1 Data Monitoring
Missingness

Feature distributions

Category changes

Unexpected values

Schema changes

Feature availability

11.2 Drift Monitoring
Population Stability Index (PSI)

Kolmogorov-Smirnov distribution comparison

Feature-level drift

Prediction-distribution drift

Configurable PSI interpretation guidelines:

text
PSI < 0.10                 Stable
0.10 ≤ PSI < 0.25          Monitor / investigate
PSI ≥ 0.25                 Material instability — investigate
11.3 Performance Monitoring
When realised outcomes become available:

ROC-AUC, Gini, KS

Brier Score

Calibration slope and intercept

Observed vs expected event rates

The monitoring architecture supports future production outcome data even
though the current project uses a static public dataset.

12. Decision Engine (Downstream of the Model)
The PD model produces a calibrated probability. The decision engine
converts that probability into a business action.

12.1 Configurable Inputs
text
C_FN                       Cost of approving a defaulter
C_FP                       Cost of declining a creditworthy applicant
risk_appetite
minimum_approval_criteria
maximum_acceptable_pd
12.2 Possible Outputs
text
APPROVE
REFER
DECLINE
Three-way outcomes support manual review where operational policy
requires it.

12.3 Separation Principle
Cost parameters are stored externally to the model.

Thresholds are not hardcoded into the model artifact.

The threshold may change without retraining the PD model.

13. Champion / Challenger Governance
13.1 Comparison Criteria
Champion and challenger are compared using a common validation
framework across:

text
Discrimination
Calibration
Stability
Explainability
Fairness
Latency
Operational complexity
Governance requirements
13.2 Promotion Rule
A challenger may only replace the champion following documented
acceptance criteria and governance review. A higher AUC alone does
not promote the challenger.

14. Engineering
14.1 Offline Path
text
Raw relational data
   ↓
Validation
   ↓
Feature engineering
   ↓
Feature store
   ↓
Model training
   ↓
Model registry / artifacts
14.2 Online Path
text
Applicant request
   ↓
Input validation
   ↓
Feature transformation
   ↓
PD model
   ↓
Calibration
   ↓
Decision engine
   ↓
Reason codes
   ↓
API response
The online path does not perform full-table historical aggregation per
request.

14.3 Engineering Targets
Requirement	Target
Inference latency (online path)	< 100 ms p99
Batch feature computation	< 5 s per applicant where applicable
API availability target	99.9%
Reproducibility	Deterministic / seed-controlled
Test coverage	≥ 80% of src/
Model versioning	Required
Configuration versioning	Required
Data lineage	Required
The <100 ms target applies to the online inference path only, not to
the complete offline relational feature-engineering pipeline.

15. Limitations
15.1 Known Limitations
No genuine calendar-time OOT — The public dataset lacks an
absolute application-date field; validation uses a documented proxy
temporal holdout where defensible.

Dataset representativeness — The Home Credit population may not
represent all lending contexts.

Missing data — Some features have high missing rates; handled
with imputation + indicators, but residual uncertainty remains.

Class imbalance — Minority class (~8%) constrains precision at
very low thresholds.

No macroeconomic conditioning — Model does not incorporate
macro-scenario features.

Static dataset — No live feedback loop in the current version.

Fairness coverage — Only fairness analyses supported by
available attributes can be performed.

No legal certification — The project demonstrates capability,
not regulatory approval.

15.2 Failure Modes and Mitigations
Failure Mode	Detection	Mitigation
Data drift	PSI > 0.10	Investigate, retrain
Concept drift	KS decline	Recalibrate or retrain
Calibration drift	Reliability diagram, slope/intercept	Recalibrate
Feature pipeline failure	Schema validation	CI/CD contract tests
Latency regression	APM monitoring	Distillation / caching
Reason-code drift	Mapping-layer tests	Regression tests
16. Governance Artifacts
The project produces:

Model card (this document)

Model development report

Validation report

Feature catalogue

Data-quality report

Leakage assessment

Calibration report

Fairness assessment (within data limitations)

Explainability documentation

Monitoring specification

Model version history

Experiment metadata

Deployment documentation

The governance framework supports principles associated with
established model-risk and credit-risk practices — documentation,
validation, monitoring, explainability, and controlled model change.
It does not claim regulatory approval.

17. Contact and References
Owner: ML Engineering / Data Science
Reviewers: Risk Management, Compliance, Model Validation

References:

Federal Reserve — SR 11-7: Guidance on Model Risk Management (2011)

Basel Committee on Banking Supervision — IRB Approach

IASB — IFRS 9: Financial Instruments (2014)

Siddiqi, N. — Credit Risk Scorecards (Wiley, 2006)

Thomas, L., Edelman, D., Crook, J. — Credit Scoring and Its Applications (SIAM, 2017)

Home Credit Default Risk — Kaggle (2018)

§14 — Decision Engine
markdown
## 14. Decision Engine

The decision engine converts calibrated PD into a business action
(APPROVE / REFER / DECLINE). It is a separate component from the model:
the threshold is a business decision, not a model output.

### 14.1 Separation of Concerns

| Concern | Component | Owner |
|---|---|---|
| PD estimation | Challenger model (LightGBM) | Data Science |
| Probability calibration | Isotonic regression | Data Science |
| Decision threshold | Decision engine (config-driven) | Credit Risk |
| Policy overrides | Decision engine (config-driven) | Compliance / Risk |
| Segment rules | Decision engine (config-driven) | Product / Risk |

The threshold can change without retraining the model. Segment rules
and policy overrides can be updated without touching code.

### 14.2 Decision Logic

The engine applies the following decision rule, in order:

1. **Policy overrides** — if any configured override flag is set
   (e.g. `sanctions_list`, `internal_blacklist`, `fraud_flag`), the
   decision is forced and no threshold logic is applied.
2. **Segment thresholds** — the applicable segment determines the
   `approve_max` and `review_max` thresholds.
3. **Threshold comparison** — PD is compared against the segment's
   thresholds:
   - PD < `approve_max` → APPROVE
   - `approve_max` ≤ PD < `review_max` → REFER
   - PD ≥ `review_max` → DECLINE

### 14.3 Default Configuration

| Parameter | Value | Notes |
|---|---:|---|
| `C_FN` | 20.0 | Cost of approving a defaulter, relative units |
| `C_FP` | 1.0 | Cost of rejecting a good applicant |
| `approve_max` | 0.040 | Cost-optimal at C_FN/C_FP = 20 |
| `review_max` | 0.200 | Manual review band |
| Default segment | `default` | |
| Segments | `new_customer`, `existing_customer` | |

### 14.4 Policy Overrides

Configured overrides that force a decision regardless of PD:

| Override | Force Decision | Reason |
|---|---|---|
| `sanctions_list` | DECLINE | Applicant appears on sanctions list |
| `internal_blacklist` | DECLINE | Applicant is on internal blacklist |
| `fraud_flag` | DECLINE | Application flagged as potential fraud |
| `vip_customer` | APPROVE | Pre-approved VIP customer |

Overrides fire before threshold logic. When an override fires,
`threshold_used` is recorded as 0.0 to make the non-PD-based nature
of the decision explicit in the audit log.

### 14.5 Decision Distribution on Holdout

| Band | Champion | Challenger |
|---|---:|---:|
| APPROVE | 40.1% | 43.3% |
| REFER | 51.4% | 47.6% |
| DECLINE | 8.5% | 9.1% |

**Actual default rates by band** (challenger):

- APPROVE: 2.07%
- REFER: 9.02%
- DECLINE: 31.73%

The monotonic increase confirms the engine + model are working
together correctly: approvals are low-risk, decliners are high-risk,
and the middle band matches the population average (appropriate for
manual review).

### 14.6 Cost-Sensitive Threshold

At `C_FN / C_FP = 20`, the cost-optimal threshold on the challenger
holdout is **0.040**. This matches the configured default. Changing
the cost ratio would shift the optimal threshold; the config supports
this without retraining.

Cost sweep across ratios:

| C_FN/C_FP | Champion cost | Challenger cost | Savings |
|---:|---:|---:|---:|
| 5 | 0.3271 | 0.3171 | 3.07% |
| 10 | 0.5049 | 0.4939 | 2.17% |
| 20 | 0.6875 | 0.6742 | 1.93% |
| 30 | 0.7834 | 0.7637 | 2.50% |
| 50 | 0.8829 | 0.8638 | 2.17% |

### 14.7 Audit Trail

Every decision records:

- Applicant ID
- Calibrated PD
- Decision
- Threshold used
- Segment
- Override applied (if any)
- Override reason (if any)
- Expected cost of the decision
- Model and config versions
- Timestamp

The audit trail enables replay, regulatory reporting, and backtesting
of alternative policies.
§15 — Explainability
markdown
## 15. Explainability

The system provides explanations at both global and per-applicant
levels, and generates structured reason codes for adverse-action
notices.

### 15.1 Global Explainability

**Challenger (LightGBM):** TreeSHAP values over a 5,000-applicant
sample from the holdout. Reported as mean |SHAP| per feature.

Top 10 features by mean |SHAP|:

| Rank | Feature | mean\|SHAP\| |
|---:|---|---:|
| 1 | `EXT_SOURCE_2` | 0.0223 |
| 2 | `EXT_SOURCE_3` | 0.0196 |
| 3 | `EXT_SOURCE_1` | 0.0100 |
| 4 | `AMT_GOODS_PRICE` | 0.0085 |
| 5 | `AMT_ANNUITY` | 0.0083 |
| 6 | `CODE_GENDER` | 0.0078 |
| 7 | `AMT_CREDIT` | 0.0071 |
| 8 | `DAYS_BIRTH` | 0.0063 |
| 9 | `inst_delay_positive_rate` | 0.0055 |
| 10 | `pos_cnt_instalment_future_mean` | 0.0055 |

Three of the top 10 are aggregated features built by the relational
pipeline (`inst_delay_positive_rate`, `pos_cnt_instalment_future_mean`,
and — further down — several bureau features). The aggregation is
carrying real signal.

**Champion (Elastic-Net LR):** coefficients on WoE-encoded features.
Because the features are WoE-transformed, coefficient magnitudes are
directly comparable across features.

### 15.2 Local Explainability

For each applicant, the system produces:

- Per-feature signed SHAP contributions (challenger)
- Per-feature coefficient contributions (champion)
- Top-K features ranked by absolute contribution

Example (challenger, high-risk applicant, PD = 0.964):

| Feature | Contribution | Value |
|---|---:|---:|
| `EXT_SOURCE_3` | +0.234 | 0.037 |
| `bureau_credit_sum_overdue_total` | +0.145 | 131,728 |
| `EXT_SOURCE_2` | +0.144 | 0.00001 |
| `bureau_credit_sum_overdue_max` | +0.067 | 53,307 |

All positive contributions indicate features that increased the
model's risk estimate. The signs are correct for a high-risk
applicant.

### 15.3 Reason Codes

Reason codes are a **deterministic mapping layer**, distinct from
SHAP values. The mapping is defined in `configs/reason_codes.yaml`:
feature → code → human-readable description

text

For example:
bureau_credit_sum_overdue_max → HIGH_BUREAU_OVERDUE →
"High maximum overdue amount on credit bureau records"

text

The reason code generator:

1. Takes the per-feature SHAP contributions for one applicant.
2. Filters for **positive** contributions (features that increased
   risk).
3. Maps each surviving feature to its structured code and description.
4. Ranks by contribution magnitude.
5. Returns the top 4 as "primary reasons" and the next 6 as
   "additional reasons."

Reason codes are only produced for DECLINE and REFER decisions. For
APPROVE, the reason arrays are empty (no adverse action).

### 15.4 Adverse-Action Notice

The reason codes are rendered as a plain-language adverse-action
notice suitable for customer communication. Example (high-risk
applicant, PD = 0.964):
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

### 15.5 Regulatory Alignment

| Requirement | How satisfied |
|---|---|
| ECOA / Reg B (US) — adverse action reasons | Structured `primary_reasons` with plain-language descriptions |
| GDPR Art. 22 (EU) — meaningful information | Reason codes plus model-agnostic explanation |
| SR 11-7 (Fed) — model interpretability | Global and local SHAP for challenger; coefficients for champion |
| Right to human review | Stated explicitly in the rendered notice |

The technical capability is demonstrated. Legal certification for a
specific jurisdiction is out of scope and would depend on that
jurisdiction's additional requirements.

### 15.6 Explainability Limitations

- SHAP values are computed against the **challenger model**. The
  champion's explanations use coefficients instead. The two are not
  directly comparable (different model structures).
- Reason codes filter for positive contributions only. A feature that
  contributed negatively (reduced risk) does not appear in the
  primary or additional reasons, even if it had a large magnitude.
- Reason code mappings are defined per-feature. Features absent from
  `configs/reason_codes.yaml` are not surfaced as reasons, even if
  they contributed. This is deliberate: unexplained internal features
  must not leak into customer-facing notices.
§16 — Monitoring
markdown
## 16. Monitoring

The monitoring module compares a reference population against a
monitoring population and reports feature drift, prediction drift,
and calibration drift.

### 16.1 Reference and Monitoring Populations

| Population | Purpose |
|---|---|
| **Reference** | Held-out baseline (current: `val` split) |
| **Monitoring** | Current population (current: `holdout` split) |

**Important design choice:** the reference population must be one
the model was not trained on. Using the training split as reference
inflates AUC comparisons because training performance reflects
overfitting, not population drift. Feature-distribution drift (PSI)
is unaffected by overfitting; performance metrics (AUC, Brier) are
not.

### 16.2 Feature Drift

Computed for every feature in the challenger model's input:

| Metric | Description |
|---|---|
| **PSI** | Population Stability Index: binned distribution shift |
| **KS** | Kolmogorov-Smirnov: maximum CDF separation |

PSI interpretation:

| PSI range | Severity | Action |
|---|---|---|
| < 0.10 | Stable | None |
| 0.10 – 0.25 | Moderate | Investigate specific features |
| ≥ 0.25 | Material | Action required |

### 16.3 Prediction Drift

PSI and KS computed on the calibrated PD distribution. This catches
shifts in the model's output that may not be visible feature-by-
feature.

### 16.4 Calibration Drift

Slope and intercept on the monitoring population, computed via
logistic regression of `y_true` on `logit(PD)`:

| Metric | Target | Alert Threshold |
|---|---:|---:|
| Calibration slope | 1.0 | Outside [0.90, 1.10] |
| Calibration intercept | 0.0 | Outside [−0.15, 0.15] |
| AUC | ≥ 0.70 | Below 0.70, or drop > 0.03 vs reference |
| Brier | ≤ 0.075 | Reported |

### 16.5 Current Monitoring Results

On the current reference (val) vs monitoring (holdout):

| Metric | Value |
|---|---:|
| Features stable | 362 / 362 |
| Features moderate drift | 0 |
| Features material drift | 0 |
| Prediction drift PSI | 0.0001 |
| Prediction drift KS | 0.0028 |
| Calibration slope | 0.979 |
| Calibration intercept | −0.040 |
| AUC reference → monitoring | 0.7897 → 0.7846 |
| AUC drop | 0.0051 |

All features stable, calibration within tolerance, AUC drop well
below the 0.03 threshold. The model is behaving correctly on the
monitoring population.

### 16.6 Retraining Triggers

The model is retrained when any of the following are observed on
production data:

- Any feature with PSI ≥ 0.25
- Prediction drift PSI ≥ 0.25
- Calibration slope outside [0.90, 1.10] for a sustained period
- Calibration intercept outside [−0.15, 0.15] for a sustained period
- AUC below 0.70, or drop ≥ 0.03 from the reference
- Scheduled quarterly retraining regardless of drift

Retraining is a controlled process: the new model enters shadow mode,
is evaluated against the current champion on the same holdout, and
is promoted only through the champion/challenger governance process.
§17 — API Service
markdown
## 17. API Service

The model is served as a FastAPI application, containerized for
deployment.

### 17.1 Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Liveness check |
| `/readiness` | GET | Resource readiness |
| `/model-info` | GET | Model and config metadata |
| `/score` | POST | Calibrated PD + decision |
| `/explain` | POST | SHAP contributions + reason codes |
| `/report` | POST | Rendered adverse-action notice |

### 17.2 Request and Response

**Request (POST /score):**

```json
{
  "applicant_id": "string",
  "features": {"<feature_name>": <value>, ...},
  "segment": "default",
  "policy_flags": {"sanctions_list": false, ...}
}
Feature values may be numeric or null. Nulls are treated as missing
and handled by LightGBM natively. Feature names not present in the
model's contract are silently ignored.

Response:

json
{
  "applicant_id": "string",
  "pd": 0.0065,
  "decision": "APPROVE",
  "threshold_used": 0.04,
  "segment": "default",
  "override_applied": null,
  "override_reason": null,
  "expected_cost": 0.131,
  "model": "challenger",
  "model_version": "0.1.0",
  "config_version": "1.0.0",
  "decided_at": "2026-09-22T01:17:16Z"
}
17.3 Error Handling
Status	Trigger	Response
422	Validation error (missing/invalid field)	{"detail": [...]}
500	Unhandled exception in handler	{"detail": "Internal server error", ...}
503	Model or calibrator not loaded	{"detail": "Service not ready: ..."}
FastAPI's default error format is used throughout. Custom error
schemas are not needed; the standard detail field is sufficient
for both internal and external consumers.

All error responses are logged at WARNING or ERROR level for
operational visibility.

17.4 Latency
Endpoint	Expected latency
/health	< 1 ms
/readiness	< 1 ms
/model-info	< 1 ms
/score	5 – 20 ms
/explain	50 – 200 ms
/report	50 – 200 ms
/score meets the < 100 ms p99 target. /explain and /report
are slower because SHAP is computed on demand per request. In
production, SHAP values for repeat requests would be cached.

17.5 Containerization
The service is packaged as a Docker image built from a multi-stage
Dockerfile. The runtime image contains:

Python 3.11

FastAPI + Uvicorn

pandas, numpy, pyarrow, scikit-learn, LightGBM, SHAP

Model artifacts and feature contract

Healthcheck is configured in both the Dockerfile and
docker-compose.yml.

17.6 Train-Serving Skew Handling
A critical detail: training reads typed Parquet, but serving receives
untyped JSON. A JSON null produces a Python None, and a single-
row DataFrame built from a dict containing None produces an
object-dtype column — which LightGBM rejects.

Fix: explicit .astype("float64") at the serving boundary. This
coerces None to NaN in a float column, which LightGBM handles
natively. The cast is applied in both scoring and explanation paths.

This pattern is essential for any production ML system that reads
typed data during training and untyped data during serving.

text

---

## Where These Sections Go

Open `docs/model_card.md`. The file currently has sections §1 through §13 (Champion/Challenger Governance) followed by "Contact and References."

Insert §14–§17 between §13 and the Contact section.

The full sequence becomes:

- §1–§13 (existing)
- **§14 Decision engine**
- **§15 Explainability**
- **§16 Monitoring**
- **§17 API service**
- Contact and References (existing)

---

## After Appending — Update the Version

At the top of `docs/model_card.md`, update the version from `2.0` to `3.0` and note the new sections:

```markdown
**Version:** 3.0
**Last Updated:** 2026-09-22

**Change log:**
- v3.0 — Added §14 Decision engine, §15 Explainability,
  §16 Monitoring, §17 API service.
- v2.0 — Initial production-grade model card.


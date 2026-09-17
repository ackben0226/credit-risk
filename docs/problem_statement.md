# Credit Risk Probability of Default (PD) Model

## Problem Statement

**Document Version:** 2.0
**Status:** Approved — Technical Revision
**Owner:** ML Engineering / Data Science
**Last Updated:** 2026-09-13
**Classification:** Internal — Model Development

---

# 1. Executive Summary

This project defines and implements a production-oriented **Probability of Default (PD) modelling system** for retail credit risk.

The system will estimate the probability that a loan applicant experiences the target credit-risk event during the relevant loan performance window. The resulting PD will form the analytical foundation for a downstream credit decisioning system.

The project is designed around four principles:

1. **Risk discrimination** — applicants with materially different levels of credit risk must be distinguishable.
2. **Probability calibration** — predicted PDs must correspond closely to observed event rates.
3. **Explainability and governance** — predictions must be interpretable, reproducible, auditable, and capable of supporting adverse-action reasoning.
4. **Production readiness** — the model must be deployable through a low-latency API with appropriate monitoring, testing, versioning, and operational controls.

The project uses the **Home Credit Default Risk** dataset, which represents a relational retail-lending environment containing an application dataset together with historical bureau, previous-application, and repayment/balance information.

The modelling system will therefore go beyond a conventional machine-learning classification notebook. It will demonstrate an end-to-end credit-risk workflow covering:

```text
Raw Data
    ↓
Data Validation
    ↓
Leakage Controls
    ↓
Relational Feature Engineering
    ↓
WoE / IV Transformation
    ↓
Champion + Challenger Models
    ↓
Probability Calibration
    ↓
Model Validation
    ↓
Cost-Sensitive Decisioning
    ↓
Explainability / Reason Codes
    ↓
API Deployment
    ↓
Monitoring
    ↓
Governance & CI/CD
```

The primary output of the modelling layer is:

```text
PD(x) = P(TARGET = 1 | X = x)
```

where `TARGET = 1` represents the target payment-difficulty/default event defined by the Home Credit dataset.

The project does **not** attempt to model Loss Given Default (LGD), Exposure at Default (EAD), fraud, collections, or risk-based pricing.

---

# 2. Business Context

## 2.1 The Lending Decision Problem

At the point of a credit application, a lender must determine whether extending credit to an applicant is consistent with its risk appetite and commercial objectives.

The overall lending problem can be separated into three related questions:

1. **Risk estimation** — What is the probability that the applicant will default?
2. **Decisioning** — Given the estimated risk and the lender's risk appetite, should the application be approved, declined, or referred?
3. **Pricing** — If approved, what price appropriately reflects the applicant's risk?

This project focuses primarily on **risk estimation** and, as a separate downstream layer, **cost-sensitive decisioning**.

Risk-based pricing is explicitly outside the scope of this project.

---

## 2.2 Why Probability of Default Is the Central Quantity

Probability of Default is a fundamental quantity in credit risk because it can support multiple downstream activities, including:

| Downstream Application     | Relationship to PD                                   |
| -------------------------- | ---------------------------------------------------- |
| Credit approval            | PD can be evaluated against a defined risk threshold |
| Risk-based pricing         | PD is an important input to risk-adjusted pricing    |
| Expected Loss              | `EL = PD × LGD × EAD`                                |
| IFRS 9 ECL                 | PD contributes to expected-credit-loss estimation    |
| Capital modelling          | PD contributes to credit-risk capital calculations   |
| Portfolio analysis         | PD distributions support portfolio risk segmentation |
| Stress testing             | PD can be evaluated under adverse scenarios          |
| Collections prioritisation | PD can contribute to expected-loss prioritisation    |
| Risk reporting             | PD provides a common risk measure across portfolios  |

A well-designed PD model must therefore provide more than a high classification score. It must produce **reliable probabilities** that can be interpreted, validated, monitored, and used consistently downstream.

---

## 2.3 Stakeholders

| Stakeholder                      | Primary Interest                                       |
| -------------------------------- | ------------------------------------------------------ |
| Underwriting / Credit Operations | Fast and reliable credit-risk decisions                |
| Risk Management                  | Model validity, calibration, stability and backtesting |
| Finance / Treasury               | Reliable risk estimates for financial planning         |
| Compliance / Legal               | Explainability, governance and auditability            |
| Model Validation                 | Independent assessment of methodology and performance  |
| Product / Growth                 | Approval volume and customer experience                |
| Data Science                     | Statistical performance and modelling methodology      |
| ML Engineering                   | Reproducibility, deployment and reliability            |
| Senior Management                | Risk-adjusted business performance                     |

---

# 3. Problem Definition

## 3.1 Formal Definition

Given a set of applicant and credit-history features `X` that are legitimately available at or before the defined application decision point, estimate:

```text
PD(x) = P(TARGET = 1 | X = x)
```

where:

```text
TARGET = 1
```

represents the target payment-difficulty/default event supplied by the Home Credit dataset.

The system must not use information generated after the application decision point to predict the target.

The central modelling objective is therefore:

> **Estimate a calibrated conditional probability of the target credit-risk event using information available at the point of application.**

---

## 3.2 Model Requirements

The final PD system must:

1. Produce a probability in the interval `[0, 1]`.
2. Demonstrate meaningful discrimination between lower- and higher-risk applicants.
3. Demonstrate acceptable probability calibration.
4. Apply documented monotonicity constraints where the economic relationship between a feature and risk is known and appropriate.
5. Prevent target and temporal leakage.
6. Produce reproducible predictions across controlled model versions.
7. Support global and individual-level explanations.
8. Produce structured reason codes suitable for downstream adverse-action workflows.
9. Support low-latency inference.
10. Maintain a complete lineage from source data to model output.

---

# 4. Mathematical Formulation

## 4.1 Learning Objective

The model is trained to minimise a regularised, potentially weighted classification loss:

```text
L(θ) = Σᵢ wᵢ · ℓ(yᵢ, fθ(xᵢ)) + λΩ(θ)
```

where:

* `fθ` = model parameterised by `θ`
* `ℓ` = classification loss
* `wᵢ` = optional observation weight
* `Ω(θ)` = regularisation term
* `λ` = regularisation strength
* `yᵢ` = observed target

For probability estimation, **log-loss / binary cross-entropy** will be the primary training loss where appropriate.

Class imbalance will be addressed through documented modelling and evaluation methodology rather than indiscriminate oversampling.

---

## 4.2 Decision Objective

The PD model and the lending decision engine are separate components.

The PD model estimates:

```text
PD(x)
```

The decision layer then evaluates the PD against configurable business costs and risk appetite.

A simplified decision rule is:

```text
Approve loan  ⇔  PD(x) < τ*
```

where `τ*` is the selected decision threshold.

The optimal threshold depends on configurable assumptions regarding:

* cost of approving a defaulter
* cost of declining a creditworthy applicant
* expected margin
* risk appetite
* operational constraints

The threshold must therefore **not be hardcoded into the trained PD model**.

---

# 5. Problem Type and Evaluation Framework

| Dimension                        | Definition                                         |
| -------------------------------- | -------------------------------------------------- |
| Learning paradigm                | Supervised learning                                |
| Task                             | Binary classification / probability estimation     |
| Target                           | Home Credit `TARGET`                               |
| Output                           | Probability in `[0,1]`                             |
| Primary discrimination metric    | ROC-AUC                                            |
| Primary calibration metric       | Brier Score                                        |
| Secondary discrimination metrics | Gini, KS, PR-AUC                                   |
| Calibration diagnostics          | Calibration curve, calibration intercept and slope |
| Stability metrics                | PSI and distribution monitoring                    |
| Decision layer                   | Cost-sensitive threshold optimisation              |

No single metric will determine model acceptance.

A model with strong AUC but poor calibration will not automatically qualify as the preferred PD model.

---

# 6. Data Scope

## 6.1 Source Dataset

The project uses the public **Home Credit Default Risk** dataset.

The modelling data consists of:

### Application-level sources

* `application_train.csv`
* `application_test.csv`

### Relational historical sources

* `bureau.csv`
* `bureau_balance.csv`
* `previous_application.csv`
* `POS_CASH_balance.csv`
* `installments_payments.csv`
* `credit_card_balance.csv`

The project therefore treats the dataset as an **application-level modelling population supported by six major historical relational sources**.

`application_train.csv` provides the labelled development population because it contains `TARGET`.

`application_test.csv` is treated as an unlabeled scoring/reference population and must not be used to fit supervised transformations or model parameters.

Other downloaded files such as competition submission templates or documentation are not treated as modelling tables.

---

## 6.2 Relational Data Architecture

The expected logical structure is:

```text
                    APPLICATION
                         │
          ┌──────────────┼──────────────┐
          │              │              │
          ▼              ▼              ▼
       BUREAU       PREVIOUS_APP    other application
          │              │
          ▼              ▼
   BUREAU_BALANCE   POS_CASH_BALANCE
                         │
                 ┌───────┴────────┐
                 ▼                ▼
        INSTALLMENTS       CREDIT_CARD
        PAYMENTS           BALANCE
```

The exact join keys, cardinalities, duplicate behaviour, missingness and referential integrity must be validated during ingestion.

---

# 7. Data and Leakage Policy

## 7.1 Application-Time Availability

Every production feature must have a documented relationship to the application decision point.

Features will be classified as:

```text
AVAILABLE_AT_APPLICATION
DERIVED_FROM_PRE_APPLICATION_HISTORY
POST_APPLICATION
UNKNOWN
```

Only the first two categories may enter the production PD feature set.

Any feature whose availability cannot be established will be excluded or explicitly documented as a limitation.

---

## 7.2 Leakage Controls

The pipeline must explicitly test for:

* direct target leakage
* post-application variables
* future repayment information
* future credit-balance information
* accidental use of validation/test information during preprocessing
* aggregation leakage
* duplicate applicants across development and validation populations
* feature-generation leakage
* calibration leakage
* threshold-optimisation leakage

All transformations that learn parameters from data must be fitted exclusively on the relevant development dataset.

For example:

```text
TRAIN
  ↓
Fit imputation
Fit bins
Fit WoE
Fit scaling
Fit feature selection
Fit model
Fit calibration
  ↓
VALIDATION / TEST
  ↓
Transform using TRAIN parameters only
```

---

# 8. Data Validation Requirements

Before modelling, the ingestion pipeline must validate:

### Schema

* expected columns
* data types
* required identifiers
* target availability
* unexpected columns

### Quality

* missingness
* duplicate records
* invalid values
* impossible ranges
* cardinality
* categorical consistency

### Relational integrity

* join-key validity
* orphan records
* one-to-many relationships
* duplicate keys
* aggregation correctness

### Target integrity

* target distribution
* class imbalance
* missing target values
* inconsistent target labels

The output of validation must be a reproducible data-quality report.

---

# 9. Feature Engineering

## 9.1 Objective

Feature engineering will transform the relational Home Credit data into an application-level feature store.

The feature store should contain a substantial, documented feature set, with a target of:

> **At least 200 validated candidate features**, where justified by the underlying data.

The number of features is a target rather than a reason to generate meaningless variables.

---

## 9.2 Feature Categories

Candidate features may include:

### Application characteristics

* income
* credit amount
* annuity
* goods price
* employment characteristics
* household characteristics

### Credit bureau aggregates

* number of bureau accounts
* active accounts
* overdue accounts
* credit utilisation
* outstanding balances
* historical credit duration

### Previous application behaviour

* number of previous applications
* previous approval/rejection rates
* previous loan amounts
* previous application outcomes
* historical repayment-related indicators

### Installment behaviour

* payment delays
* missed payments
* payment ratios
* instalment burden
* historical payment regularity

### POS / cash-loan behaviour

* active contracts
* contract status
* delinquency patterns
* balance trends

### Credit-card behaviour

* balances
* utilisation
* payment behaviour
* delinquency indicators

Each feature must have documented provenance and business meaning.

---

# 10. Feature Metadata Contract

Each production candidate feature must have metadata including, where applicable:

```text
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
```

Example:

```text
Feature:
    bureau_active_credit_count

Source:
    bureau

Definition:
    Number of active bureau credit accounts
    associated with the applicant at the application point

Aggregation:
    COUNT

Availability:
    Pre-application historical information

Risk Direction:
    Empirically assessed

Transformation:
    Monotonic WoE bins where appropriate
```

---

# 11. WoE and IV Framework

Where appropriate for the scorecard champion, continuous and categorical variables will undergo **Weight of Evidence (WoE)** transformation and **Information Value (IV)** assessment.

The process will include:

```text
Raw Feature
    ↓
Missing-value treatment
    ↓
Initial binning
    ↓
Bin stability assessment
    ↓
WoE calculation
    ↓
Monotonicity assessment
    ↓
IV calculation
    ↓
Feature selection
```

Binning must be fitted on the development dataset only.

Monotonicity will be enforced only where an ordered relationship between the feature and risk is conceptually and empirically defensible.

Nominal categorical variables will not be given artificial ordering merely to satisfy a monotonicity requirement.

---

# 12. Champion Model

## 12.1 Elastic-Net Logistic Regression Scorecard

The primary model will be an **Elastic-Net Logistic Regression scorecard** using appropriately transformed features.

The champion model is preferred because it provides:

* strong interpretability
* transparent variable relationships
* stable probability estimation
* straightforward reason-code generation
* suitability for traditional credit-risk governance
* relatively low inference cost

The model will use:

* WoE-transformed variables where appropriate
* regularisation
* controlled feature selection
* documented coefficients
* reproducible training configuration

---

# 13. Challenger Model

A gradient-boosting model will be developed as a challenger.

The implementation may use:

* LightGBM
* XGBoost
* CatBoost

The challenger exists to determine whether nonlinear and interaction effects provide material improvements over the interpretable champion.

The challenger will be evaluated using the **same leakage controls and validation framework** as the champion.

A higher AUC alone will not automatically cause the challenger to replace the champion.

Model selection will consider:

```text
Discrimination
Calibration
Stability
Explainability
Fairness
Latency
Operational complexity
Governance requirements
```

---

# 14. Probability Calibration

Raw model scores will be evaluated for calibration.

Where required, calibration will be performed using an appropriate method such as:

* Platt / logistic calibration
* isotonic regression

Calibration must be fitted using data independent of the model-fitting process in order to avoid optimistic calibration estimates.

The final calibrated PD will be evaluated using:

* Brier Score
* calibration curve
* calibration intercept
* calibration slope
* observed vs predicted event rates across risk bands / deciles

The final production output is the **calibrated PD**, not merely the raw model score.

---

# 15. Validation Strategy

## 15.1 Development / Validation / Test Separation

The modelling framework will maintain strict separation between:

```text
Development Data
       ↓
Model fitting
       ↓
Validation Data
       ↓
Model / hyperparameter selection
       ↓
Final Test / Holdout
       ↓
Unbiased performance assessment
```

No test-set information may influence model development.

---

## 15.2 Out-of-Time Validation Limitation

A genuine out-of-time validation requires a reliable application-date or observation-time variable.

The public Home Credit dataset does not provide a straightforward absolute application calendar date suitable for a conventional chronological train/test split.

Therefore, the project will **not falsely label an arbitrary index-based split as true out-of-time validation**.

Where a defensible ordering proxy is available, an ordered holdout may be used and will explicitly be described as a **proxy temporal validation**.

If no defensible temporal ordering can be established, the validation report must state:

> A genuine calendar-time out-of-time validation could not be performed using the public dataset because an appropriate absolute application-date field is unavailable.

The project may subsequently demonstrate the architecture required for genuine OOT validation using an appropriately dated production-style dataset.

---

# 16. Performance Targets

The following are **acceptance targets**, not guaranteed outcomes.

| Metric            |    Target | Purpose                 |
| ----------------- | --------: | ----------------------- |
| ROC-AUC           |    ≥ 0.78 | Discrimination          |
| Gini              |    ≥ 0.56 | Rank-order separation   |
| KS statistic      |    ≥ 0.40 | Credit-risk separation  |
| Brier Score       |   ≤ 0.075 | Probability calibration |
| Calibration slope | 0.90–1.10 | Calibration quality     |
| PSI               |    ≤ 0.10 | Population stability    |

Performance will be reported honestly.

If a target is not achieved, the validation report will document:

* observed result
* confidence / uncertainty where appropriate
* likely causes
* model limitations
* mitigation options
* whether the model remains operationally acceptable

Targets must not be achieved through inappropriate leakage or test-set optimisation.

---

# 17. Cost-Sensitive Decisioning

The PD model produces a probability.

The decision engine converts that probability into a business action.

The engine will support configurable parameters including:

```text
C_FN
C_FP
risk appetite
minimum approval criteria
maximum acceptable PD
```

The decision layer may return:

```text
APPROVE
REFER
DECLINE
```

where the three-way outcome is used when operational policy requires manual review.

Cost parameters must be stored externally and must not be hardcoded into model parameters.

The decision threshold may therefore change without retraining the underlying PD model.

---

# 18. Explainability

## 18.1 Global Explainability

The system will provide global explanations including:

* feature importance
* coefficient interpretation for the champion
* WoE relationships
* model-level contribution analysis
* risk-direction analysis

---

## 18.2 Local Explainability

For individual applicants, the system should provide:

```text
Applicant
    ↓
PD
    ↓
Decision
    ↓
Top contributing factors
```

SHAP or another appropriate explanation framework may be used for the challenger and other models where technically appropriate.

---

# 19. Adverse-Action Reason Codes

Detailed model explanations and customer-facing adverse-action reasons are treated as separate concepts.

For example:

### Internal explanation

```text
Low external credit score
High existing credit exposure
Recent payment difficulties
```

### Structured reason code

```text
HIGH_EXISTING_CREDIT_OBLIGATIONS
```

The project will implement a deterministic reason-code mapping layer where appropriate.

Reason codes must:

* correspond to genuine model drivers
* be stable and reproducible
* avoid misleading statements
* be traceable to model features
* remain understandable to non-technical stakeholders

The project will demonstrate the technical capability required for adverse-action workflows without claiming that the public dataset alone constitutes a complete legal compliance implementation.

---

# 20. Fairness and Responsible Modelling

The system must include a fairness assessment where the available data supports one.

The assessment will consider:

* approval / decision rates
* predicted-risk distributions
* error rates where appropriate
* calibration across groups
* disparate-impact indicators
* proxy-variable risks

However, the public Home Credit dataset does not necessarily provide every protected characteristic required for a complete real-world fair-lending assessment.

Therefore:

> **The fairness report must clearly distinguish between fairness analyses supported by the available dataset and analyses that require protected-class information unavailable in the dataset.**

No unsupported claim of full regulatory fairness compliance will be made.

---

# 21. Monitoring

The deployed model must support monitoring of both **data stability** and **model performance**.

## 21.1 Data Monitoring

Monitor:

* missingness
* feature distributions
* category changes
* unexpected values
* schema changes
* feature availability

---

## 21.2 Drift Monitoring

The system will support:

* Population Stability Index (PSI)
* Kolmogorov-Smirnov (KS) distribution comparison
* feature-level drift
* prediction-distribution drift

Suggested PSI interpretation:

```text
PSI < 0.10
    Stable

0.10 ≤ PSI < 0.25
    Monitor / investigate

PSI ≥ 0.25
    Material instability requiring investigation
```

These thresholds are monitoring guidelines and must be configurable.

---

## 21.3 Performance Monitoring

When realised outcomes become available, monitor:

* ROC-AUC
* Gini
* KS
* Brier Score
* calibration slope
* calibration intercept
* observed vs expected event rates

The monitoring architecture must support future production outcome data even though the current project uses a static public dataset.

---

# 22. Production Architecture

The project will separate offline feature generation from online inference.

## Offline path

```text
Raw Relational Data
        ↓
Validation
        ↓
Feature Engineering
        ↓
Feature Store
        ↓
Model Training
        ↓
Model Registry / Artifacts
```

## Online path

```text
Applicant Request
        ↓
Input Validation
        ↓
Feature Transformation
        ↓
PD Model
        ↓
Calibration
        ↓
Decision Engine
        ↓
Reason Codes
        ↓
API Response
```

The online path must not require expensive full-table historical aggregation during each prediction request.

---

# 23. API Requirements

A FastAPI service will expose the trained model.

A conceptual prediction request will contain the applicant features required by the production feature contract.

The response should include, as appropriate:

```json
{
  "model_version": "1.0.0",
  "pd": 0.072,
  "decision": "APPROVE",
  "threshold": 0.10,
  "reason_codes": []
}
```

The exact API schema will be formally defined before implementation.

The API must validate:

* request schema
* data types
* required fields
* acceptable ranges
* model version
* feature availability

---

# 24. Engineering Requirements

| Requirement               |                                    Target |
| ------------------------- | ----------------------------------------: |
| Inference latency         |                               <100 ms p99 |
| Batch feature computation | <5 seconds per applicant where applicable |
| API availability target   |                                     99.9% |
| Reproducibility           |           Deterministic / seed-controlled |
| Test coverage             |                            ≥80% of `src/` |
| Model versioning          |                                  Required |
| Configuration versioning  |                                  Required |
| Data lineage              |                                  Required |

The <100ms target applies to the **online inference path**, not the complete offline relational feature-engineering pipeline.

---

# 25. Model Governance

The project will produce governance artifacts including:

* model card
* model development report
* validation report
* feature catalogue
* data-quality report
* leakage assessment
* calibration report
* fairness assessment
* explainability documentation
* monitoring specification
* model version history
* experiment metadata
* deployment documentation

The governance framework is designed to support principles associated with established model-risk and credit-risk practices, including documentation, validation, monitoring, explainability and controlled model change.

The project does not claim that a model trained on a public Kaggle dataset alone constitutes regulatory approval or production banking compliance.

---

# 26. Champion / Challenger Governance

The champion/challenger framework will operate as follows:

```text
                 ┌─────────────────┐
                 │ Champion        │
                 │ Elastic-Net LR  │
                 └────────┬────────┘
                          │
                          │
                     Evaluation
                          │
                          ▼
                 ┌─────────────────┐
                 │ Challenger      │
                 │ Gradient Boost  │
                 └────────┬────────┘
                          │
                          ▼
              ┌────────────────────────┐
              │ Independent comparison │
              │                        │
              │ AUC                    │
              │ Calibration            │
              │ KS                     │
              │ Stability              │
              │ Explainability         │
              │ Fairness               │
              │ Latency                │
              └───────────┬────────────┘
                          │
                          ▼
                  Model promotion
```

A challenger may only replace the champion following documented acceptance criteria and governance review.

---

# 27. Technical Constraints

The project must account for:

### Class imbalance

The target event represents a minority class.

Therefore, evaluation must use metrics appropriate for imbalanced classification, including PR-AUC alongside ROC-AUC.

### Missing data

Missingness may contain predictive information.

Missing values must therefore be investigated rather than automatically dropped.

### Relational scale

The historical tables contain millions of records and require efficient aggregation.

### Leakage

All features must respect the application-time information boundary.

### Monotonicity

Monotonicity must be applied selectively and according to documented economic/statistical relationships.

### Calibration

Calibration must be preserved and monitored after deployment.

### Reproducibility

All transformations, configurations, model parameters and artefacts must be version-controlled.

---

# 28. Business Constraints

The system must:

* support real-time decisioning
* maintain configurable decision thresholds
* support retraining
* support champion/challenger comparison
* preserve model lineage
* provide interpretable outputs
* support controlled model promotion
* avoid hardcoded business costs
* permit future replacement of the public dataset with production data

A recommended retraining cadence may be demonstrated, but the final production cadence must ultimately depend on observed model stability, portfolio change and governance requirements.

---

# 29. Out of Scope

The following are explicitly outside the scope of this project:

* Loss Given Default (LGD) modelling
* Exposure at Default (EAD) modelling
* risk-based pricing engine
* collections strategy
* fraud detection
* macroeconomic scenario generation
* full IFRS 9 lifetime ECL engine
* full Basel capital calculation engine
* real customer PII processing
* production banking infrastructure
* production customer data
* legal certification or regulatory approval
* comprehensive protected-class fairness analysis where required attributes are unavailable

The architecture may provide interfaces for these future capabilities without implementing them.

---

# 30. Key Assumptions

The project assumes that:

1. The Home Credit target is an appropriate proxy for the credit-risk event being modelled.
2. Historical patterns contain useful predictive information.
3. Training data is sufficiently representative for the modelling demonstration.
4. Features selected for production are available at the application decision point.
5. Target labels are sufficiently reliable for supervised learning.
6. Business costs can be provided as configurable inputs.
7. The public dataset is suitable for demonstrating the modelling architecture but is not equivalent to a production banking data environment.
8. Genuine calendar-time OOT validation is limited by the absence of a suitable absolute application-date field.

---

# 31. Success Criteria

## 31.1 Modelling

The project is successful when:

* the target is correctly defined and validated
* leakage controls are implemented
* relational data is successfully aggregated
* a documented feature store is created
* ≥200 meaningful candidate features are targeted
* WoE/IV analysis is implemented where appropriate
* champion and challenger models are trained
* probabilities are calibrated
* model performance is independently evaluated

---

## 31.2 Performance

The model should target:

```text
ROC-AUC       ≥ 0.78
Gini          ≥ 0.56
KS            ≥ 0.40
Brier         ≤ 0.075
Calibration   0.90–1.10 slope
```

These remain targets rather than guaranteed outcomes.

---

## 31.3 Engineering

The project is successful when:

* API inference meets the <100ms p99 target under the defined benchmark
* tests achieve ≥80% coverage of `src/`
* models are reproducible
* configurations are externalised
* artefacts are versioned
* Docker deployment works
* CI/CD quality gates pass
* monitoring can detect data and prediction drift

---

## 31.4 Governance

The project is successful when:

* a model card exists
* a validation report exists
* feature lineage is documented
* leakage assessment is documented
* calibration is documented
* explainability is documented
* fairness limitations are documented
* model versions are traceable
* deployment decisions are auditable

---

# 32. Definition of Done

The project will be considered complete when all of the following are satisfied:

* [ ] All application and relational source files are ingested and validated.
* [ ] Data schemas and relational integrity are tested.
* [ ] Target definition is documented.
* [ ] Application-time feature availability rules are documented.
* [ ] Leakage controls are implemented and tested.
* [ ] Relational data is aggregated to the application level.
* [ ] A documented feature store containing ≥200 meaningful candidate features is targeted.
* [ ] Feature metadata and lineage are maintained.
* [ ] WoE/IV transformations are implemented where appropriate.
* [ ] Monotonicity constraints are documented and validated where applicable.
* [ ] Champion Elastic-Net Logistic Regression scorecard is trained.
* [ ] Challenger gradient-boosting model is trained.
* [ ] Champion and challenger models are compared using a common validation framework.
* [ ] Probability calibration is implemented where required.
* [ ] Calibration performance is documented.
* [ ] Model performance is evaluated on an untouched holdout.
* [ ] Limitations of genuine out-of-time validation are explicitly documented.
* [ ] Cost-sensitive decision threshold optimisation is implemented.
* [ ] Decision thresholds are externally configurable.
* [ ] Global model explanations are generated.
* [ ] Local explanations are supported.
* [ ] Structured adverse-action reason codes are implemented where appropriate.
* [ ] Fairness analysis is performed to the extent supported by the available data.
* [ ] PSI/KS monitoring is implemented.
* [ ] FastAPI service is implemented.
* [ ] API latency is benchmarked against the <100ms p99 target.
* [ ] Docker deployment is functional.
* [ ] Automated tests achieve ≥80% coverage of `src/`.
* [ ] CI/CD quality gates pass.
* [ ] Model card is published.
* [ ] Validation report is published.
* [ ] Complete model/data lineage is documented.

---

# 33. Project Deliverables

The final project will contain:

```text
1. Data ingestion pipeline
2. Data validation framework
3. Leakage detection framework
4. Relational feature engineering pipeline
5. Feature store / feature catalogue
6. WoE / IV transformation pipeline
7. Champion scorecard
8. Challenger gradient-boosting model
9. Calibration pipeline
10. Model evaluation framework
11. Cost-sensitive decision engine
12. Explainability framework
13. Adverse-action reason-code framework
14. Fairness assessment
15. Drift monitoring
16. FastAPI inference service
17. Docker deployment
18. Automated test suite
19. CI/CD pipeline
20. Model card
21. Model validation report
22. Technical documentation
```

---

# 34. Why This Problem

This project is intentionally designed to demonstrate the difference between a conventional machine-learning experiment and a **production-oriented credit-risk modelling system**.

The project develops capability across:

* credit-risk modelling
* statistical learning
* probability calibration
* interpretable scorecards
* gradient boosting
* relational feature engineering
* model governance
* explainable AI
* responsible AI
* model monitoring
* API engineering
* containerisation
* automated testing
* CI/CD

The central objective is therefore not simply to maximise a leaderboard metric.

It is to demonstrate the ability to design, validate, explain, deploy and govern a **complete PD modelling system**.

---

# 35. Reference Framework

The project will use the following sources as conceptual and methodological references:

* Basel Committee on Banking Supervision — IRB / credit-risk modelling guidance
* IFRS 9 — Financial Instruments
* Federal Reserve — SR 11-7: Guidance on Model Risk Management
* Consumer credit / adverse-action requirements applicable to the intended jurisdiction
* GDPR requirements relevant to automated decision-making and data protection
* Home Credit Default Risk dataset documentation
* Siddiqi, N. — *Credit Risk Scorecards*
* Thomas, L., Edelman, D., Crook, J. — *Credit Scoring and Its Applications*

Regulatory references will be treated as **governance and design considerations**, not as a claim that this public-data project independently establishes regulatory compliance.

---

# 36. Final System Objective

The final system must answer the following question reliably:

> **Given only information legitimately available at the time of application, how likely is this applicant to experience the target credit-risk event, and how should that calibrated risk estimate be communicated to a downstream decisioning system?**

The final architecture is therefore:

```text
                 HOME CREDIT DATA
                         │
                         ▼
                DATA VALIDATION
                         │
                         ▼
              LEAKAGE / TIME CHECKS
                         │
                         ▼
              RELATIONAL AGGREGATION
                         │
                         ▼
                  FEATURE STORE
                         │
             ┌───────────┴───────────┐
             ▼                       ▼
        WoE / IV              Raw / engineered
             │                       │
             ▼                       ▼
       CHAMPION                  CHALLENGER
   Elastic-Net LR             Gradient Boosting
             │                       │
             └───────────┬───────────┘
                         ▼
                   CALIBRATION
                         │
                         ▼
                 MODEL VALIDATION
                         │
                         ▼
                 CALIBRATED PD
                         │
                         ▼
             COST-SENSITIVE DECISION
                         │
             ┌───────────┼───────────┐
             ▼           ▼           ▼
          APPROVE      REFER       DECLINE
                         │
                         ▼
              REASON CODES / SHAP
                         │
                         ▼
                    FASTAPI
                         │
                         ▼
                     DOCKER
                         │
                         ▼
              MONITORING + GOVERNANCE
                         │
                         ▼
                     CI / CD
```

**End of Problem Statement — Version 2.0**

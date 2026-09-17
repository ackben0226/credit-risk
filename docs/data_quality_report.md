markdown
# Data Quality Report — Home Credit Default Risk

**Document Version:** 1.0
**Status:** Approved
**Owner:** ML Engineering / Data Science
**Last Updated:** 2026-09-13
**Classification:** Internal — Data Documentation

**Source inspection:** `artifacts/reports/inspection_20260913T170549Z.json`
**Inspection script:** `scripts/inspect_data.py`
**Inspection run:** 2026-09-13T17:05:49Z

---

## 1. Purpose

This report documents the empirical state of the Home Credit Default Risk
raw data as it was received and inspected. It serves as the reference for
every downstream decision in the pipeline:

- Schema contracts in `src/credit_risk/data/validate.py`
- Feature availability rules in `src/credit_risk/data/leakage.py`
- Aggregation design in `src/credit_risk/features/aggregations/`
- Null-handling policy in the feature store
- Data-quality notes in the Model Card
- Limitations declared in the validation report

Every figure in this document is copied from the machine-readable
inspection report. If the raw data changes, re-run `scripts/inspect_data.py`
and regenerate this document.

---

## 2. File Inventory

| Table | File | Rows | Columns | Size Role |
|---|---|---:|---:|---|
| `application_train` | `application_train.csv` | 307,511 | 122 | Development population with `TARGET` |
| `application_test` | `application_test.csv` | 48,744 | 121 | Scoring / reference population (no `TARGET`) |
| `bureau` | `bureau.csv` | 1,716,428 | 17 | Bureau records |
| `bureau_balance` | `bureau_balance.csv` | 27,299,925 | 3 | Monthly bureau balances |
| `previous_application` | `previous_application.csv` | 1,670,214 | 37 | Prior Home Credit applications |
| `pos_cash_balance` | `POS_CASH_balance.csv` | 10,001,358 | 8 | POS / cash-loan monthly snapshots |
| `installments_payments` | `installments_payments.csv` | 13,605,401 | 8 | Repayment history |
| `credit_card_balance` | `credit_card_balance.csv` | 3,840,312 | 23 | Credit-card monthly snapshots |
| `columns_description` | `HomeCredit_columns_description.csv` | 219 | 5 | Data dictionary (reference, not modelling) |
| `sample_submission` | `sample_submission.csv` | 48,744 | 2 | Submission template (not used) |

**Modelling tables:** `application_train`, `bureau`, `bureau_balance`,
`previous_application`, `pos_cash_balance`, `installments_payments`,
`credit_card_balance`.

**Reference tables (loaded for metadata, not features):**
`columns_description`.

**Excluded:** `sample_submission` (Kaggle competition artifact).

---

## 3. Target

| Property | Value |
|---|---|
| Target column | `TARGET` |
| Positive class | Payment difficulty / default |
| Positive count | 24,825 |
| Negative count | 282,686 |
| **Positive rate** | **0.0807** |
| Null count | 0 |

### Implications

- **Class imbalance is severe** (~1:11.4). Accuracy is a disqualified metric.
- Primary discrimination metric: **ROC-AUC**.
- Co-primary for imbalanced selection: **PR-AUC**.
- Primary calibration metric: **Brier Score**.
- Decision threshold must be cost-sensitive, not 0.5 (see Model Card §12).

---

## 4. Encoding

The dataset ships with **mixed encodings**:

| File | Encoding | Notes |
|---|---|---|
| All 8 modelling CSVs | UTF-8 | Clean, standard |
| `HomeCredit_columns_description.csv` | **Windows-1252 (cp1252)** | Contains byte `0x85` at position 1283 (the ellipsis `…` in a description field) |

### Handling

The pipeline reads every CSV through a fallback chain:

```text
utf-8  →  cp1252  →  latin-1
latin-1 maps every byte to a codepoint and never raises, so the chain
is guaranteed to terminate successfully on any byte sequence.

This behavior is implemented in scripts/inspect_data.py
(read_csv_with_fallback, iter_csv_with_fallback) and will be
reused in src/credit_risk/data/ingest.py.

Documentation Note
This is a real production-relevant characteristic of the source, not
a bug. Metadata files exported from Windows systems routinely carry
cp1252. It must appear in any audit-ready documentation of the pipeline.

5. Relational Integrity
All join-key statistics were computed on the full table (not a sample).

5.1 Summary Table
| Child Table | Parent Table | Child Key | Parent Key | Child Rows | Unique Child Keys | Orphans | Orphan Rate | Coverage |
|---|---|---|---:|---:|---:|---:|---:|
| bureau | application_train | SK_ID_CURR | SK_ID_CURR | 1,716,428 | 305,811 | 42,320 | 0.138 | 0.857 |
| previous_application | application_train | SK_ID_CURR | SK_ID_CURR | 1,670,214 | 338,857 | 47,800 | 0.141 | 0.946 |
| bureau_balance | bureau | SK_ID_BUREAU | SK_ID_BUREAU | 27,299,925 | 817,395 | 43,041 | 0.053 | 0.451 |
| pos_cash_balance | previous_application | SK_ID_PREV | SK_ID_PREV | 10,001,358 | 936,325 | 37,422 | 0.040 | 0.538 |
| installments_payments | previous_application | SK_ID_PREV | SK_ID_PREV | 13,605,401 | 997,865 | 38,847 | 0.039 | 0.574 |
| credit_card_balance | previous_application | SK_ID_PREV | SK_ID_PREV | 3,840,312 | 104,307 | 11,372 | 0.109 | 0.056 |

Definitions:

Orphans — unique child keys not present in the parent key set.

Orphan rate — orphans ÷ unique child keys.

Coverage — parent keys that appear in the child ÷ parent unique keys.

5.2 application_train Integrity
Property	Value
Unique SK_ID_CURR	307,511
Duplicate SK_ID_CURR	0
Conclusion: application_train has a clean primary key. One row per
applicant. No deduplication needed.

6. Interpretation of Relational Findings
6.1 Orphans in bureau and previous_application
bureau has 42,320 orphan SK_ID_CURR values.

previous_application has 47,800 orphan SK_ID_CURR values.

These orphans do not appear in application_train. They most likely
belong to:

Applicants in application_test (the unlabeled scoring population), or

Historical clients not represented in either split.

Pipeline implication: When joining child tables to application_train,
use an inner join on SK_ID_CURR. Orphans drop silently, and the
drop rate is logged.

Follow-up check (not yet performed): quantify how many orphans belong
to application_test versus neither. If the majority align with
application_test, the interpretation is clean. If not, this becomes an
open data-provenance question.

6.2 bureau_balance coverage of bureau is 0.451
Only 45.1% of bureau records have any monthly balance history.

Implication: ~55% of bureau records have no bureau_balance rows.
Any aggregate feature built from bureau_balance will be null for the
corresponding bureau records, and consequently null for a substantial
fraction of applicants after roll-up.

Pipeline implication:

Preserve nulls. Do not impute 0 — the absence of monthly history is
itself informative.

Add a binary indicator has_bureau_balance_history at the applicant level.

Document this in the feature metadata contract.

Orphan interpretation: 43,041 unique SK_ID_BUREAU in bureau_balance
do not exist in bureau. Those rows cannot be joined and are dropped.

6.3 credit_card_balance coverage of previous_application is 0.056
Only 5.6% of prior applications resulted in credit-card products.

Implication: Credit-card aggregate features will be null for ~94% of
applicants.

Pipeline implication:

Treat credit-card features as a specialist family, not a bulk source.

Add has_credit_card_history at the applicant level.

Evaluate credit-card IV on the subset of applicants who have it, then
decide whether the family earns a place in the production feature set.

Do not impute credit-card features for the 94% without history.

6.4 POS / installments coverage is ~0.54–0.57
Roughly 55% of prior applications have POS / cash or installment records.

Implication: The corresponding aggregates will be null for ~45% of
prior-application rows.

Pipeline implication: Same treatment as bureau_balance — preserve nulls,
add indicator flags.

6.5 Orphan counts cluster in the 37K–48K range across child tables
text
bureau_balance → bureau              43,041
pos_cash_balance → previous_application   37,422
installments_payments → previous_application  38,847
bureau → application_train           42,320
previous_application → application_train    47,800
These similar magnitudes suggest a shared provenance — likely a
historical client cohort included in the child tables but not present in
the parent tables. It is not a modelling blocker.

Action: Document as an open data-provenance question. Include in the
validation report as a declared limitation.

7. Known Data Characteristics (Summary)
ID	Characteristic	Impact	Handling
DQ-01	TARGET positive rate 8.07%	Severe class imbalance	PR-AUC co-primary, cost-based threshold
DQ-02	columns_description is cp1252, not UTF-8	Metadata file fails on strict UTF-8 read	Encoding fallback chain
DQ-03	application_train.SK_ID_CURR is unique	No dedup required	—
DQ-04	bureau and previous_application have SK_ID_CURR orphans	Inner join drops ~14% of child unique keys	Log drop rate
DQ-05	bureau_balance coverage of bureau = 45.1%	Null-heavy aggregates after roll-up	Preserve nulls + indicator flag
DQ-06	credit_card_balance coverage = 5.6%	Specialist feature family	Indicator flag; separate IV evaluation
DQ-07	POS / installments coverage ~55%	Null-heavy aggregates	Preserve nulls + indicator flag
DQ-08	Similar orphan magnitudes across child tables	Provenance question	Document as open question
DQ-09	No calendar-date column in application_train	No genuine out-of-time validation possible	Documented in Model Card §5.3
8. Open Questions
Orphan provenance — Do the orphan SK_ID_CURR values in bureau
and previous_application align with application_test, or with
neither split?

Orphan magnitude cluster — Why do orphan counts across child
tables cluster in the 37K–48K range?

bureau_balance coverage — Is the 45% coverage an artifact of the
Home Credit data collection window, or a systematic issue?

Missing data informativeness — Do nulls in feature families
(credit-card, bureau_balance) correlate with TARGET? If yes, missing
indicators carry signal.

These questions do not block the project. Each will be resolved during
feature engineering (Phase 2) or explicitly carried as a documented
limitation.

9. Reproducibility
To regenerate this report from scratch:

bash
# From project root
python scripts/inspect_data.py
Outputs:

Console log (structured, timestamped)

artifacts/reports/inspect.log — persistent log file

artifacts/reports/inspection_<timestamp>.json — machine-readable

configs/data_config.yaml — updated with the latest report pointer

Every table, column, and statistic in this document is derived from the
JSON. Nothing is hand-typed except interpretation and narrative.

10. References
artifacts/reports/inspection_20260913T170549Z.json — source inspection

scripts/inspect_data.py — inspection implementation

configs/data_config.yaml — pipeline configuration and report pointer

docs/problem_statement.md — v2.0

docs/model_card.md — v2.0


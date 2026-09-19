### Observation: XNA as a structural sentinel in Home Credit

**Date:** 2026-09-15

The `"XNA"` value in Home Credit categorical columns is not a
uniform "missing" sentinel. Its prevalence varies widely by column:

| Column | XNA rows | XNA rate | Interpretation |
|---|---:|---:|---|
| NAME_CONTRACT_TYPE | 346 | 0.02% | Rare unknown |
| NAME_CLIENT_TYPE | 1,941 | 0.12% | Rare unknown |
| CODE_REJECT_REASON | 5,244 | 0.31% | Rare unknown |
| NAME_PORTFOLIO | 372,230 | 22.3% | Structural |
| NAME_YIELD_GROUP | 517,215 | 31.0% | Structural |
| NAME_PAYMENT_TYPE | 627,384 | 37.6% | Structural |
| NAME_SELLER_INDUSTRY | 855,720 | 51.2% | Structural |
| NAME_GOODS_CATEGORY | 950,809 | 56.9% | Structural |
| NAME_PRODUCT_TYPE | 1,063,666 | 63.7% | Structural |

Low-prevalence XNA (< 1%) is treated as an "unclassified" residual.
High-prevalence XNA (> 20%) is structural — for cash/revolving loans,
these fields do not apply, and Home Credit fills them with "XNA".
In this case, "XNA" should be treated as a first-class category
because it encodes product structure.

**Design rule:**
- XNA rate < 1% → fold into residual count
- XNA rate > 5% → emit as explicit category count (`*_xna`)
- XNA rate 1–5% → case-by-case

**Also observed:** `PRODUCT_COMBINATION` has 346 nulls and 0 XNA.
Likely the same 346 rows that have `NAME_CONTRACT_TYPE = "XNA"`. Worth
verifying if the patterns correlate — could indicate a small number
of anomalous application records.

## previous_application features

### Observation: `XNA` contract type and null `PRODUCT_COMBINATION` are perfectly correlated

**Date:** 2026-09-15

Diagnostic on the raw previous_application table:

| Condition | Rows |
|---|---|
| `NAME_CONTRACT_TYPE == "XNA"` | 346 |
| `PRODUCT_COMBINATION is null` | 346 |
| Both | 346 |
| Only XNA | 0 |
| Only null PRODUCT_COMBINATION | 0 |

Every row with `NAME_CONTRACT_TYPE == "XNA"` has `PRODUCT_COMBINATION` null,
and vice versa. All 346 rows have `NAME_CONTRACT_STATUS == "Canceled"` and
`AMT_CREDIT == 0.0`.

**Interpretation:** these are applications that were canceled before
product assignment. The nulls in `PRODUCT_COMBINATION` are genuinely
inapplicable, not missing. `XNA` in `NAME_CONTRACT_TYPE` is Home Credit's
"not applicable" code.

**Handling:**
- Do not impute `NAME_CONTRACT_TYPE` or `PRODUCT_COMBINATION` for these rows
- Do not drop these rows — they represent real application attempts
- Emit `prev_count_contract_type_xna` as an explicit feature: the count of
  prior applications canceled before product assignment. This is a signal
  in its own right.
- `Canceled` in `NAME_CONTRACT_STATUS` is already counted via the generic
  status loop, so it is preserved.

**No changes required to `bureau.py`** — this note informs the design of
`previous_application.py`.

Notes
-----
- POS_CASH_balance contains both SK_ID_PREV and SK_ID_CURR. This module
  deliberately joins through SK_ID_PREV → previous_application to enforce
  parent-child consistency. Relying on the child's own SK_ID_CURR is
  riskier because orphan records exist.
- Orphan rows (340,561, from 37,422 orphan SK_ID_PREV values) are dropped
  by the inner join. This is deliberate: without a valid parent, we cannot
  attribute the record to an applicant reliably. Revisit if a baseline
  model shows lost signal.
- NAME_CONTRACT_STATUS in POS_CASH_balance has 7 known values plus a small
  tail of unexpected values (observed: Canceled 15, XNA 2). Unknown
  statuses are counted in pos_count_status_other rather than dropped.

## POS/CASH features

### Observation: orphan rows dropped by inner join

**Date:** 2026-09-16

Confirmed via diagnostic: 340,561 rows in POS_CASH_balance have SK_ID_PREV
values that don't appear in previous_application. These orphans are dropped
by the inner join in `_aggregate_to_applicant`. The drop is deterministic:

    total pos_cash rows:          10,001,358
    orphan rows:                     340,561
    non-orphan rows:               9,660,797
    aggregated status counts sum:  9,660,797  

**Decision:** keep the drop. The child table's own SK_ID_CURR column could
be used as a fallback, but mixing two attribution paths introduces subtle
inconsistencies. Revisit if a baseline model shows that orphan signal
matters.

### Observation: unexpected status values

**Date:** 2026-09-16

Non-orphan POS_CASH_balance rows contain statuses `Canceled` (15 rows) and
`XNA` (2 rows) — not in the known status list. These are captured by
`pos_count_status_other` (total 17 rows). Design is working as intended:
named features for expected values, residual bucket for the rest.

## credit_card features

### Observation: orphan rows are full-history credit cards

**Date:** 2026-09-16

Diagnostic confirmed a striking pattern:

| Group | Count | Mean rows per key | Median | Max |
|---|---:|---:|---:|---:|
| Orphan SK_ID_PREV | 11,372 | 95.22 | 96 | 96 |
| Non-orphan SK_ID_PREV | 92,935 | 29.67 | 18 | 96 |

Orphan SK_ID_PREV values have nearly full 96-month histories (96 is the
maximum possible in credit_card_balance, corresponding to 8 years of
monthly snapshots). Non-orphan keys are much shorter — half have fewer
than 18 months of history.

**Interpretation:** orphans appear to be older credit cards whose
parent records were removed from previous_application. The full 96-month
histories suggest they're among the oldest accounts in the dataset.

**Consequence:** the inner join drops 1,082,816 rows (28% of raw
credit_card_balance) despite only 11% of SK_ID_PREV keys being orphans.
The row-drop rate is high because orphan keys have 3x more rows per key
on average.

**Decision:** keep the drop. If we ever add orphan rows via their own
SK_ID_CURR, we'd inject a long-history cohort that would skew aggregate
features. Documented for the record.

### Observation: utilization is bimodal

**Date:** 2026-09-16

`cc_utilization_max` has a bimodal distribution:

- 25th percentile: 0.00 (dormant card, never used)
- Median:          0.90 (heavy user, close to limit)
- 75th percentile: 1.04 (over-limit at some point)
- Max:             11.78 (extreme over-limit)

This bimodality will produce multiple WoE bins in Stage C (likely:
null / zero / low / medium / high / over-limit).

### Observation: ~33% of credit-card months have no activity

**Date:** 2026-09-16

Approximately 33% of applicants have null for the payment and drawing
features (`cc_payment_to_min_ratio_*`, `cc_net_drawing_*`,
`cc_drawings_*`). These correspond to dormant credit cards — accounts
opened but never used.

This is a behavioral pattern, not a data quality problem. Stage C should
emit an explicit indicator (`cc_is_dormant` or similar) to capture the
distinction.

### Observation: status counts are internally consistent

**Date:** 2026-09-16

Verified: sum of `cc_count_status_*` across all applicants = sum of
`cc_count_total_months` = 2,757,496. Zero difference. Every retained
row is accounted for by a status bucket.

## Stage C — Feature transformation (complete)

### Observation: two parallel feature representations

**Date:** 2026-09-17

The pipeline produces two distinct feature matrices from the same
assembled data:

- **Champion** (WoE-encoded, 180 features)
    - Nulls become their own WoE bin
    - Categoricals become WoE-binned
    - Consumed by Elastic-Net logistic regression
    - Suitable for a regulator-interpretable scorecard

- **Challenger** (raw + encoded, 362 features)
    - Nulls preserved (LightGBM handles them natively)
    - Categoricals factorized to integer codes
    - 5 explicit missing indicators for the sparsest families
    - Consumed by LightGBM

Both representations use the same DEV/VAL/HOLDOUT splits. All fitted
transformations (WoE bins, categorical factorizations) were fit on
DEV only.

### Observation: no imputation in either representation

The original plan called for an `imputation.py` module. It was
superseded:

- WoE encoding resolves nulls structurally for the champion — nulls
  get their own bin with their own numeric value.
- LightGBM handles nulls natively for the challenger — imputation
  would destroy the missingness signal.

Only 5 explicit missing indicators were added, for the sparsest
features where missingness is likely informative:
- `cc_utilization_last_is_null`   (74.94% null)
- `cc_balance_last_is_null`       (74.66% null)
- `bb_max_status_max_is_null`     (70.86% null)
- `bb_months_at_3_plus_total_is_null` (70.01% null)
- `bureau_credit_sum_overdue_max_is_null` (14.36% null)

### Observation: no unseen categorical levels in VAL or HOLDOUT

Verified via `.min()` = 0 for every categorical column across all
splits. Because the split is random (not temporal), the population is
homogeneous — rare categories appear in all splits proportionally.
This is expected and simplifies downstream categorical handling.

## Monitoring design

### Decision: reference split is VAL, not TRAIN

**Date:** 2026-09-18

The monitoring module compares a reference population against a
monitoring population. The initial configuration used the train split
as reference, which produced a spurious "AUC dropped by 7.5 pp" warning.

**Root cause:** train AUC (0.8597) is inflated by overfitting. Comparing
it against holdout AUC (0.7846) measures overfitting, not drift. AUC
drift is only meaningful when the reference is a held-out set that the
model was not trained on.

**Fix:** reference_split is now "val" (0.7881), which is a genuine
held-out population from the same time period as holdout. Any future
AUC drop on this comparison reflects real distribution or concept
drift, not memorization.

**General principle:** for performance-based drift comparisons, the
reference must be a population the model has not been trained on.
For feature-distribution-based drift (PSI), training data is
acceptable as reference because distributions are not affected by
model overfitting.
# Champion / Challenger Comparison

*Generated: 20260918T142727Z*

---

## Holdout Metrics

| Metric | Champion | Challenger | Δ (Ch - Cg) |
|---|---:|---:|---:|
| AUC | 0.772163 | 0.784622 | +0.012459 |
| Gini | 0.544326 | 0.569245 | +0.024919 |
| KS | 0.413586 | 0.431313 | +0.017728 |
| Brier (lower better) | 0.067079 | 0.065771 | -0.001307 |
| Calibration slope (target 1.0) | 1.004861 | 0.978661 | -0.026200 |
| Calibration intercept (target 0.0) | 0.012501 | -0.039801 | -0.052301 |

## Cross-checks

Recomputed metrics vs stored metrics. Tolerance 0.1%.

| Check | Stored | Recomputed | Delta | Status |
|---|---:|---:|---:|:---:|
| champion_auc | 0.772163 | 0.772163 | 0.000000 | ✅ |
| champion_brier | 0.067079 | 0.067079 | 0.000000 | ✅ |
| challenger_auc | 0.785178 | 0.784622 | 0.000556 | ✅ |

## Cost-Sensitive Comparison

Expected cost per applicant on holdout, at the optimal threshold for each model, across a range of C_FN/C_FP ratios.

| C_FN/C_FP | Champion thresh | Challenger thresh | Champion cost | Challenger cost | Savings |
|---:|---:|---:|---:|---:|---:|
| 5 | 0.1700 | 0.1900 | 0.327140 | 0.317081 | 3.075% |
| 10 | 0.0900 | 0.1100 | 0.504889 | 0.493941 | 2.168% |
| 20 | 0.0500 | 0.0400 | 0.687472 | 0.674204 | 1.930% |
| 30 | 0.0400 | 0.0400 | 0.783359 | 0.763739 | 2.505% |
| 50 | 0.0200 | 0.0300 | 0.882910 | 0.863767 | 2.168% |

## Top 10 Features

### Champion (WoE coefficient magnitude)

| Rank | Feature | Coefficient |
|---:|---|---:|
| 1 | `pos_count_total_months` | -0.6958 |
| 2 | `EXT_SOURCE_2` | -0.6942 |
| 3 | `CODE_GENDER` | -0.6698 |
| 4 | `pos_cnt_instalment_future_mean` | -0.6597 |
| 5 | `FLAG_DOCUMENT_3` | -0.6276 |
| 6 | `bureau_credit_sum_debt_mean` | -0.5864 |
| 7 | `pos_cnt_instalment_future_current` | -0.5737 |
| 8 | `EXT_SOURCE_3` | -0.5644 |
| 9 | `ORGANIZATION_TYPE` | -0.5394 |
| 10 | `OWN_CAR_AGE` | -0.5344 |

### Challenger (LightGBM gain)

| Rank | Feature | Gain |
|---:|---|---:|
| 1 | `EXT_SOURCE_3` | 15771.0 |
| 2 | `EXT_SOURCE_2` | 14967.8 |
| 3 | `EXT_SOURCE_1` | 5285.0 |
| 4 | `cc_utilization_last` | 3442.6 |
| 5 | `DAYS_BIRTH` | 2543.1 |
| 6 | `prev_refusal_rate` | 2283.0 |
| 7 | `bureau_days_credit_mean` | 1735.4 |
| 8 | `inst_delay_positive_rate` | 1701.4 |
| 9 | `pos_cnt_instalment_future_mean` | 1694.0 |
| 10 | `bureau_days_credit_max` | 1659.8 |

## Recommendation

### Statistical

Challenger is statistically superior: +1.25 pp AUC, +2.49 pp Gini, -0.0013 Brier (lower is better). Both models are within calibration tolerance.

### Operational

Challenger is the primary production model. At C_FN/C_FP=20, challenger expected cost is 1.93% lower. The champion remains available for regulatory demonstration and linear-scorecard-required contexts.

### Override Conditions

The operational recommendation above assumes no overriding constraint. The following conditions supersede it:

- Regulatory context requires a linear, points-based scorecard: use the champion.
- Explainability to a non-technical audience is the primary requirement: use the champion.
- Inference latency must be under 5 ms p99: use the champion (linear scoring is faster than tree traversal).
- Maximum discrimination is the priority and cost tolerance permits: use the challenger with calibrated PD.
- Subgroup fairness analysis reveals disparate impact in the challenger but not the champion: prefer the champion until the fairness issue is resolved.

## Notes

- Computation time: 0.7s

---

*End of comparison. This document is a companion to the Model Card and the Problem Statement.*

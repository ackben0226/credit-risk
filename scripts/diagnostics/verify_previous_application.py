"""
Verification for the previous_application aggregation output.

Run after scripts/run_previous_application_aggregation.py to sanity-check
the emitted features.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    path = project_root / "data" / "interim" / "previous_application_aggregated.parquet"

    df = pd.read_parquet(path)
    print(f"shape: {df.shape}")
    print()

    # ---- Counts --------------------------------------------------------
    print("Counts:")
    for c in [
        "prev_count_total",
        "prev_count_approved",
        "prev_count_refused",
        "prev_count_canceled",
        "prev_count_unused_offer",
    ]:
        s = df[c]
        print(f"  {c}: nulls={s.isna().sum()} min={s.min()} max={s.max()}")
    print()

    # ---- Status sum check ---------------------------------------------
    status_sum = (
        df["prev_count_approved"]
        + df["prev_count_refused"]
        + df["prev_count_canceled"]
        + df["prev_count_unused_offer"]
    )
    check = status_sum == df["prev_count_total"]
    print("Status sum check (approved+refused+canceled+unused_offer == total):")
    print(f"  matches: {check.sum()} of {len(df)}, mismatches: {(~check).sum()}")
    print()

    # ---- Contract type sum check --------------------------------------
    contract_sum = (
        df["prev_count_cash_loans"]
        + df["prev_count_consumer_loans"]
        + df["prev_count_revolving_loans"]
    )
    check2 = contract_sum == df["prev_count_total"]
    print("Contract type sum check (cash+consumer+revolving == total):")
    print(f"  matches: {check2.sum()} of {len(df)}, mismatches: {(~check2).sum()}")
    print()

    # ---- Approval rate -------------------------------------------------
    rate_min = df["prev_approval_rate"].min()
    rate_max = df["prev_approval_rate"].max()
    print("Approval rate sanity (should be in [0,1]):")
    print(f"  min={rate_min:.4f}, max={rate_max:.4f}")
    print()

    # ---- Recency distribution -----------------------------------------
    print("prev_last_name_contract_status distribution:")
    counts = (
        df["prev_last_name_contract_status"]
        .value_counts(dropna=False)
        .sort_index()
        .to_dict()
    )
    print(f"  {counts}")
    print("  (0=Approved, 1=Refused, 2=Canceled, 3=Unused offer)")


if __name__ == "__main__":
    main()
"""Verify challenger feature store outputs."""
import pandas as pd

train = pd.read_parquet("data/processed/challenger_train.parquet")
val = pd.read_parquet("data/processed/challenger_val.parquet")
holdout = pd.read_parquet("data/processed/challenger_holdout.parquet")

print("=" * 70)
print("Shapes")
print("=" * 70)
print(f"  train:   {train.shape}")
print(f"  val:     {val.shape}")
print(f"  holdout: {holdout.shape}")
print()

print("=" * 70)
print("Target distribution")
print("=" * 70)
print(f"  train:   {train['TARGET'].mean():.4f}  (n_positive={int(train['TARGET'].sum())})")
print(f"  val:     {val['TARGET'].mean():.4f}  (n_positive={int(val['TARGET'].sum())})")
print(f"  holdout: {holdout['TARGET'].mean():.4f}  (n_positive={int(holdout['TARGET'].sum())})")
print()

print("=" * 70)
print("Column layout")
print("=" * 70)
print(f"  first 5: {list(train.columns[:5])}")
print(f"  last 5:  {list(train.columns[-5:])}")
print()

print("=" * 70)
print("Categorical columns are now integer codes")
print("=" * 70)
for c in ["CODE_GENDER", "NAME_EDUCATION_TYPE", "ORGANIZATION_TYPE"]:
    print(f"  {c}:")
    print(f"    dtype: {train[c].dtype}")
    print(f"    values: {sorted(train[c].unique())[:10]}")
    print(f"    val unique: {sorted(val[c].unique())[:10]}")
print()

print("=" * 70)
print("Null preservation in sparse features")
print("=" * 70)
for c in ["cc_utilization_last", "bb_max_status_max", "bureau_credit_sum_overdue_max"]:
    print(f"  {c}: train_null={train[c].isna().mean():.4f}")
print()

print("=" * 70)
print("Missing indicators are binary")
print("=" * 70)
for c in ["cc_utilization_last_is_null", "bb_max_status_max_is_null",
          "bureau_credit_sum_overdue_max_is_null"]:
    vc = train[c].value_counts().to_dict()
    print(f"  {c}: {vc}")
print()

print("=" * 70)
print("Missing indicator consistency")
print("=" * 70)
print("  cc_utilization_last_is_null should equal cc_utilization_last.isna()")
match = (train["cc_utilization_last_is_null"] == train["cc_utilization_last"].isna().astype("int8")).all()
print(f"  consistency: {match}")
print()

print("=" * 70)
print("No unseen categories in val/holdout (all -1 counts)")
print("=" * 70)
for c in ["CODE_GENDER", "NAME_EDUCATION_TYPE", "ORGANIZATION_TYPE"]:
    train_min = train[c].min()
    val_min = val[c].min()
    holdout_min = holdout[c].min()
    val_neg1 = (val[c] == -1).sum()
    holdout_neg1 = (holdout[c] == -1).sum()
    print(f"  {c}:")
    print(f"    train_min={train_min}  val_min={val_min}  holdout_min={holdout_min}")
    print(f"    val unseen (-1): {val_neg1}  holdout unseen (-1): {holdout_neg1}")
print()

print("=" * 70)
print("Column consistency across splits")
print("=" * 70)
print(f"  train == val columns:     {list(train.columns) == list(val.columns)}")
print(f"  train == holdout columns: {list(train.columns) == list(holdout.columns)}")
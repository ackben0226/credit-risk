"""Quick verification of champion matrices."""
import pandas as pd

train = pd.read_parquet("data/processed/champion_train.parquet")
val = pd.read_parquet("data/processed/champion_val.parquet")
holdout = pd.read_parquet("data/processed/champion_holdout.parquet")

print("Shapes:", train.shape, val.shape, holdout.shape)
print()
print("Train columns (first 10):", list(train.columns[:10]))
print()

print("Target distribution per split:")
print(f"  train:   {train['TARGET'].mean():.4f}")
print(f"  val:     {val['TARGET'].mean():.4f}")
print(f"  holdout: {holdout['TARGET'].mean():.4f}")
print()

print("Sample feature value ranges:")
for c in ["EXT_SOURCE_3", "bureau_days_credit_mean", "cc_utilization_last"]:
    print(f"  {c}:")
    print(f"    nulls: {train[c].isna().sum()}")
    print(f"    min:   {train[c].min():.4f}")
    print(f"    max:   {train[c].max():.4f}")
    print(f"    mean:  {train[c].mean():.4f}")
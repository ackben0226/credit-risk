"""Check for XNA and null values in previous_application categoricals."""
from pathlib import Path
import pandas as pd

project_root = Path(__file__).resolve().parents[1]
df = pd.read_parquet(project_root / "data" / "interim" / "previous_application.parquet")

cols = [
    "NAME_CONTRACT_TYPE",
    "NAME_CONTRACT_STATUS",
    "NAME_CLIENT_TYPE",
    "NAME_GOODS_CATEGORY",
    "NAME_PORTFOLIO",
    "NAME_PRODUCT_TYPE",
    "NAME_PAYMENT_TYPE",
    "CODE_REJECT_REASON",
    "NAME_SELLER_INDUSTRY",
    "NAME_YIELD_GROUP",
    "CHANNEL_TYPE",
    "PRODUCT_COMBINATION",
]

print(f"{'column':<28} {'XNA':>10} {'null':>10} {'unique':>8}")
print("-" * 60)
for c in cols:
    if c not in df.columns:
        print(f"{c:<28} (not found)")
        continue
    xna = (df[c] == "XNA").sum()
    null = df[c].isna().sum()
    unique = df[c].nunique(dropna=False)
    print(f"{c:<28} {xna:>10,} {null:>10,} {unique:>8}")
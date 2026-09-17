"""Check whether the 346 XNA / null rows in previous_application correlate."""
from pathlib import Path
import pandas as pd

project_root = Path(__file__).resolve().parents[1]
df = pd.read_parquet(project_root / "data" / "interim" / "previous_application.parquet")

xna_ct = df["NAME_CONTRACT_TYPE"] == "XNA"
null_pc = df["PRODUCT_COMBINATION"].isna()

print(f"NAME_CONTRACT_TYPE == XNA:      {xna_ct.sum():,}")
print(f"PRODUCT_COMBINATION is null:     {null_pc.sum():,}")
print(f"Both conditions:                 {(xna_ct & null_pc).sum():,}")
print(f"Only XNA:                        {(xna_ct & ~null_pc).sum():,}")
print(f"Only null PRODUCT_COMBINATION:   {~xna_ct & null_pc.sum() if False else (~xna_ct & null_pc).sum():,}")
print()
print("Sample of rows where either condition is true:")
print(df.loc[xna_ct | null_pc, ["SK_ID_PREV", "SK_ID_CURR", "NAME_CONTRACT_TYPE", "PRODUCT_COMBINATION", "NAME_CONTRACT_STATUS", "AMT_CREDIT"]].head(10).to_string())
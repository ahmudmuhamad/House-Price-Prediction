"""
Clean the dataset.

- Reads from: data/interim/
- Logic:
    1. Drops bad columns.
    2. Calculates Medians on TRAIN.
    3. ***SAVES Medians to models/artifacts/imputer.json*** (The Artifact)
    4. Imputes 0s in all datasets using that saved artifact.
- Writes to:  data/interim/*_cleaned.csv
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

INTERIM_DIR = Path("data/processed")
ARTIFACTS_DIR = Path("models/artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Config
COLS_TO_DROP = [
    "median_sale_price", 
    "Total Population", 
    "Total School Enrollment",
    "city",
    "city_full"
]
ZERO_FIX_COLS = ["median_list_price", "median_ppsf"]

def _apply_generic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Stateless cleaning (Drops columns)."""
    df = df.copy()
    cols_present = [c for c in COLS_TO_DROP if c in df.columns]
    if cols_present:
        df = df.drop(columns=cols_present)
    return df

def _fix_zeros_with_medians(df: pd.DataFrame, medians: dict) -> pd.DataFrame:
    """Replace 0s with provided medians."""
    df = df.copy()
    for col, value in medians.items():
        if col in df.columns:
            # Replace 0 with NaN first, then fill
            df[col] = df[col].replace(0, np.nan)
            df[col] = df[col].fillna(value)
    return df

def run_cleaning(
    input_dir: Path | str = INTERIM_DIR,
    output_dir: Path | str = INTERIM_DIR
):
    print("🧹 Starting Data Cleaning...")
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)

    # 1. Load Data
    train = pd.read_csv(in_dir / "train.csv")
    eval_df = pd.read_csv(in_dir / "eval.csv")
    holdout = pd.read_csv(in_dir / "holdout.csv")

    # 2. Apply Generic Cleaning
    train = _apply_generic_cleaning(train)
    eval_df = _apply_generic_cleaning(eval_df)
    holdout = _apply_generic_cleaning(holdout)

    # 3. Calculate Medians (THE ARTIFACT)
    fill_values = {}
    for col in ZERO_FIX_COLS:
        if col in train.columns:
            # Median of rows where value is NOT 0
            median_val = float(train.loc[train[col] > 0, col].median())
            fill_values[col] = median_val
            print(f"   ℹ️ Calculated median for {col}: {median_val:,.2f}")

    # 4. Save Artifact
    artifact_path = ARTIFACTS_DIR / "imputer.json"
    with open(artifact_path, "w") as f:
        json.dump(fill_values, f)
    print(f"   💾 Saved Imputer Artifact to {artifact_path}")

    # 5. Apply Fixes (Using the artifact values)
    train = _fix_zeros_with_medians(train, fill_values)
    eval_df = _fix_zeros_with_medians(eval_df, fill_values)
    holdout = _fix_zeros_with_medians(holdout, fill_values)

    # 6. Save Data
    train.to_csv(out_dir / "train_cleaned.csv", index=False)
    eval_df.to_csv(out_dir / "eval_cleaned.csv", index=False)
    holdout.to_csv(out_dir / "holdout_cleaned.csv", index=False)

    print("✅ Cleaning complete.")

if __name__ == "__main__":
    run_cleaning()
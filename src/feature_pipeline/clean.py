"""
Clean the dataset.

- Reads from: data/interim/ (Train/Eval/Holdout)
- Performs:
    1. Drop high-VIF/redundant columns.
    2. Drop leakage columns.
    3. Drop unused text columns (City, etc.).
    4. Impute 'Zero' values in price columns using Training Medians.
- Writes to:  data/interim/ (Overwrites or saves as *_cleaned.csv - we will use suffix)
"""

import pandas as pd
import numpy as np
from pathlib import Path

INTERIM_DIR = Path("data/interim")

# 1. Columns to Drop
# 'median_sale_price': LEAKAGE. It's the answer key (almost identical to 'price').
# 'Total Population': High VIF (Sum of other columns).
# 'Total School Enrollment': High VIF (Duplicate of School Age Pop).
# 'city', 'city_full': Text columns not used in this numerical model.
COLS_TO_DROP = [
    "median_sale_price", 
    "Total Population", 
    "Total School Enrollment",
    "city",
    "city_full"
]

# 2. Columns where '0.0' is a bug (Money/Size cannot be zero)
ZERO_FIX_COLS = ["median_list_price", "median_ppsf"]

def _apply_generic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Stateless cleaning applied to any dataset (Train/Eval/Holdout)."""
    df = df.copy()

    # Drop Redundant/Leakage/Text Columns
    cols_present = [c for c in COLS_TO_DROP if c in df.columns]
    if cols_present:
        print(f"   Note: Dropping columns {cols_present}")
        df = df.drop(columns=cols_present)
    
    return df

def _fix_zeros_with_train_medians(df: pd.DataFrame, medians: dict) -> pd.DataFrame:
    """Replace 0s with provided medians (Stateful cleaning)."""
    df = df.copy()
    for col in ZERO_FIX_COLS:
        if col in df.columns:
            # Replace 0 with NaN first, then fill
            # This handles both explicit 0 and existing NaNs
            df[col] = df[col].replace(0, np.nan)
            df[col] = df[col].fillna(medians[col])
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

    # 2. Apply Generic Cleaning (Drops)
    train = _apply_generic_cleaning(train)
    eval_df = _apply_generic_cleaning(eval_df)
    holdout = _apply_generic_cleaning(holdout)

    # 3. Calculate Medians on TRAIN ONLY (Prevent Leakage)
    # We calculate the median of non-zero values
    fill_values = {}
    for col in ZERO_FIX_COLS:
        if col in train.columns:
            # Median of rows where value is NOT 0
            median_val = train.loc[train[col] > 0, col].median()
            fill_values[col] = median_val
            print(f"   ℹ️ Calculated median for {col} (Train): {median_val:,.2f}")

    # 4. Apply Fixes (Stateful)
    train = _fix_zeros_with_train_medians(train, fill_values)
    eval_df = _fix_zeros_with_train_medians(eval_df, fill_values)
    holdout = _fix_zeros_with_train_medians(holdout, fill_values)

    # 5. Save
    train.to_csv(out_dir / "train_cleaned.csv", index=False)
    eval_df.to_csv(out_dir / "eval_cleaned.csv", index=False)
    holdout.to_csv(out_dir / "holdout_cleaned.csv", index=False)

    print(f"✅ Cleaning complete. Files saved to {out_dir} with suffix '_cleaned.csv'")
    print(f"   Train Shape: {train.shape}")

if __name__ == "__main__":
    run_cleaning()
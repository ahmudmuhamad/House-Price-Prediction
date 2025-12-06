"""
Load & time-split the raw dataset.

- Reads from: data/raw/
- Writes to:  data/interim/ (Splitted but still 'raw' content)
"""

import pandas as pd
from pathlib import Path

# Change default to INTERIM to protect RAW
DEFAULT_RAW_PATH = Path("data/raw/ouseTS.csv") # Update if your file name differs
DEFAULT_OUTPUT_DIR = Path("data/interim")

def load_and_split_data(
    raw_path: Path | str = DEFAULT_RAW_PATH,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
):
    """Load raw dataset, split into train/eval/holdout by date, and save."""
    
    print(f"🔄 Loading data from {raw_path}...")
    df = pd.read_csv(raw_path)

    # Ensure datetime + sort (Critical for chronological split)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    # Define Cutoffs (Based on our earlier EDA findings)
    # Train: 2012 -> 2019 (8 years)
    # Eval:  2020 -> 2021 (2 years)
    # Holdout: 2022+ (The "Future")
    cutoff_date_eval = pd.Timestamp("2020-01-01")     
    cutoff_date_holdout = pd.Timestamp("2022-01-01")  

    # Splits
    train_df = df[df["date"] < cutoff_date_eval]
    eval_df = df[(df["date"] >= cutoff_date_eval) & (df["date"] < cutoff_date_holdout)]
    holdout_df = df[df["date"] >= cutoff_date_holdout]

    # Save to 'interim'
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    train_df.to_csv(outdir / "train.csv", index=False)
    eval_df.to_csv(outdir / "eval.csv", index=False)
    holdout_df.to_csv(outdir / "holdout.csv", index=False)

    print(f"✅ Data split completed (saved to {outdir}).")
    print(f"   Train:   {train_df.shape}")
    print(f"   Eval:    {eval_df.shape}")
    print(f"   Holdout: {holdout_df.shape}")

    return train_df, eval_df, holdout_df

if __name__ == "__main__":
    load_and_split_data()
"""
Apply Feature Engineering.

- Reads from: data/interim/*_cleaned.csv
- Performs:
    1. Date Feature Extraction (Month, ensure Year).
    2. Frequency Encoding on 'zipcode' (Learn from Train, Apply to All).
    3. Drops original 'date' column.
- Writes to:  data/processed/ (Ready for ML)
"""

import pandas as pd
from pathlib import Path

INTERIM_DIR = Path("data/interim")
PROCESSED_DIR = Path("data/processed")

def _extract_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts month/year and drops the raw date object."""
    df = df.copy()
    
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        
        # 'year' usually exists in the raw data, but we ensure it here
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        
        # Drop the raw date (models can't read '2012-04-01')
        df = df.drop(columns=["date"])
        
    return df

def _apply_frequency_encoding(train: pd.DataFrame, eval_df: pd.DataFrame, holdout: pd.DataFrame) -> tuple:
    """
    Calculates zipcode counts on TRAIN and maps them to all sets.
    Unseen zipcodes in Eval/Holdout get 0.
    """
    # 1. Learn (Fit) on Train
    zip_counts = train["zipcode"].value_counts()
    
    # 2. Apply (Transform) on Train
    train["zipcode_freq"] = train["zipcode"].map(zip_counts)
    
    # 3. Apply (Transform) on Eval & Holdout
    # fillna(0) handles zipcodes that exist in test but were never seen in training
    eval_df["zipcode_freq"] = eval_df["zipcode"].map(zip_counts).fillna(0)
    holdout["zipcode_freq"] = holdout["zipcode"].map(zip_counts).fillna(0)
    
    return train, eval_df, holdout

def run_transformation(
    input_dir: Path | str = INTERIM_DIR,
    output_dir: Path | str = PROCESSED_DIR
):
    print("⚙️ Starting Feature Transformation...")
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Cleaned Data
    print("   Loading cleaned data...")
    train = pd.read_csv(in_dir / "train_cleaned.csv")
    eval_df = pd.read_csv(in_dir / "eval_cleaned.csv")
    holdout = pd.read_csv(in_dir / "holdout_cleaned.csv")

    # 2. Stateless Transformations (Date extraction)
    train = _extract_date_features(train)
    eval_df = _extract_date_features(eval_df)
    holdout = _extract_date_features(holdout)

    # 3. Stateful Transformations (Frequency Encoding)
    # We must pass all three to keep them synced
    print("   Applying Frequency Encoding to zipcodes...")
    train, eval_df, holdout = _apply_frequency_encoding(train, eval_df, holdout)

    # 4. Save Final Processed Data
    train.to_csv(out_dir / "train_processed.csv", index=False)
    eval_df.to_csv(out_dir / "eval_processed.csv", index=False)
    holdout.to_csv(out_dir / "holdout_processed.csv", index=False)

    print(f"✅ Transformation complete. Files saved to {out_dir}")
    print(f"   Final Feature Count: {train.shape[1]}")
    print(f"   Columns: {list(train.columns[:5])} ...")

if __name__ == "__main__":
    run_transformation()
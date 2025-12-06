import pandas as pd
import numpy as np
import pytest
from pathlib import Path
import shutil
import sys

# --- FIX: Add project root to sys.path to find 'src' ---
# This points to the parent of the 'tests' folder (i.e., the project root)
sys.path.append(str(Path(__file__).parent.parent))

# Import the ACTUAL functions we wrote
from src.feature_pipeline.load import load_and_split_data
from src.feature_pipeline.clean import (
    _apply_generic_cleaning, 
    _fix_zeros_with_train_medians, 
    run_cleaning,
    COLS_TO_DROP
)
from src.feature_pipeline.transform import (
    _extract_date_features, 
    _apply_frequency_encoding, 
    run_transformation
)

# =========================
# 1. TEST: load.py
# =========================
def test_load_and_split_data_logic(tmp_path):
    # Setup dummy raw data
    dummy_csv = tmp_path / "raw_test.csv"
    df = pd.DataFrame({
        "date": ["2018-01-01", "2019-01-01", "2020-05-01", "2021-01-01", "2022-06-01"],
        "price": [100, 200, 300, 400, 500]
    })
    df.to_csv(dummy_csv, index=False)

    # Run function
    train, eval_df, holdout = load_and_split_data(raw_path=dummy_csv, output_dir=tmp_path)

    # Assertions
    # Train: < 2020
    assert len(train) == 2 
    assert train["date"].dt.year.max() < 2020
    
    # Eval: 2020 <= date < 2022
    assert len(eval_df) == 2
    assert eval_df["date"].dt.year.min() >= 2020
    
    # Holdout: >= 2022
    assert len(holdout) == 1
    assert holdout["date"].dt.year.min() >= 2022
    
    # Check files exist
    assert (tmp_path / "train.csv").exists()
    print("✅ Load and Split logic passed")

# =========================
# 2. TEST: clean.py
# =========================
def test_generic_cleaning_drops_columns():
    # Setup dataframe with bad columns
    df = pd.DataFrame({
        "price": [100],
        "median_sale_price": [99], # Should be dropped (Leakage)
        "Total Population": [500], # Should be dropped (High VIF)
        "city": ["New York"]       # Should be dropped (Text)
    })
    
    cleaned = _apply_generic_cleaning(df)
    
    for col in COLS_TO_DROP:
        assert col not in cleaned.columns
    assert "price" in cleaned.columns
    print("✅ Generic cleaning (dropping columns) passed")

def test_fix_zeros_imputation():
    # Setup dataframe with dangerous zeros
    df = pd.DataFrame({
        "median_list_price": [0, 500, 1000, 0], # Zeros should be fixed
        "other_col": [1, 2, 3, 4]
    })
    
    # Logic: We calculate median from Non-Zeros -> Median of [500, 1000] is 750
    medians = {"median_list_price": 750}
    
    fixed_df = _fix_zeros_with_train_medians(df, medians)
    
    # Zeros should be 750 now
    assert fixed_df["median_list_price"].iloc[0] == 750
    assert fixed_df["median_list_price"].iloc[3] == 750
    assert fixed_df["median_list_price"].iloc[1] == 500 # Original untouched
    print("✅ Zero imputation passed")

# =========================
# 3. TEST: transform.py
# =========================
def test_date_extraction():
    df = pd.DataFrame({"date": ["2020-05-15"]})
    transformed = _extract_date_features(df)
    
    assert "year" in transformed.columns
    assert "month" in transformed.columns
    assert "date" not in transformed.columns # Should be dropped
    assert transformed.iloc[0]["month"] == 5
    print("✅ Date extraction passed")

def test_frequency_encoding_logic():
    # Train: 'A' appears 2 times, 'B' appears 1 time
    train = pd.DataFrame({"zipcode": ["A", "A", "B"]})
    
    # Eval: 'A' exists, 'C' is new/unseen
    eval_df = pd.DataFrame({"zipcode": ["A", "C"]})
    holdout = pd.DataFrame({"zipcode": ["C"]})

    # Run Encoding
    t_out, e_out, h_out = _apply_frequency_encoding(train, eval_df, holdout)
    
    # Train checks
    # A -> 2, B -> 1
    assert t_out["zipcode_freq"].tolist() == [2, 2, 1]
    
    # Eval checks
    # A -> 2 (Known), C -> 0 (Unknown/New)
    assert e_out["zipcode_freq"].tolist() == [2, 0]
    
    print("✅ Frequency encoding (with unseen handling) passed")

# =========================
# 4. INTEGRATION TEST
# =========================
def test_full_pipeline_integration(tmp_path):
    # 1. Setup Directories
    raw_dir = tmp_path / "raw"
    interim_dir = tmp_path / "interim"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir()
    
    # 2. Create Dummy Raw Data
    df = pd.DataFrame({
        "date": pd.date_range("2018-01-01", periods=10, freq="180D"), # Spans 2018-2023
        "price": range(10),
        "zipcode": [1000, 1000, 2000, 1000, 2000, 3000, 1000, 2000, 3000, 4000],
        "median_list_price": [0, 100, 100, 100, 100, 100, 100, 100, 100, 100], # One zero to fix
        "median_sale_price": [99]*10 # Leakage to drop
    })
    raw_file = raw_dir / "data.csv"
    df.to_csv(raw_file, index=False)

    # 3. Run LOAD
    load_and_split_data(raw_path=raw_file, output_dir=interim_dir)
    assert (interim_dir / "train.csv").exists()

    # 4. Run CLEAN
    run_cleaning(input_dir=interim_dir, output_dir=interim_dir)
    assert (interim_dir / "train_cleaned.csv").exists()
    
    # Check cleaning happened
    train_clean = pd.read_csv(interim_dir / "train_cleaned.csv")
    assert "median_sale_price" not in train_clean.columns
    # The 0 in median_list_price should be filled (approx 100)
    assert train_clean["median_list_price"].min() > 0 

    # 5. Run TRANSFORM
    run_transformation(input_dir=interim_dir, output_dir=processed_dir)
    assert (processed_dir / "train_processed.csv").exists()
    
    # Check transformation
    train_proc = pd.read_csv(processed_dir / "train_processed.csv")
    assert "zipcode_freq" in train_proc.columns
    assert "month" in train_proc.columns
    assert "date" not in train_proc.columns

    print("✅ Full Pipeline Integration Passed")
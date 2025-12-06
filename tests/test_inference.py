import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

# Add project root to sys.path so we can import src
sys.path.append(str(Path(__file__).parent.parent))

from src.inference_pipeline.inference import preprocess_new_data, run_inference

# ==========================================
# 1. TEST PREPROCESSING LOGIC (Pure Function)
# ==========================================
def test_preprocess_new_data_logic():
    """
    Ensures that raw data is correctly transformed into model-ready numeric input
    using the provided artifacts (imputer, freq_map).
    """
    # 1. Setup Input Data (Raw, Dirty)
    raw_df = pd.DataFrame({
        "date": ["2023-01-01", "2023-06-01"],
        "zipcode": [90210, 10001],           # 90210 is known, 10001 is new
        "median_list_price": [0, 500_000],   # 0 is a bug, needs imputing
        "median_sale_price": [100, 100],     # Leakage, must be dropped
        "city": ["LA", "NY"],                # Text, must be dropped
        "extra_col": [1, 1]                  # Random col, should be kept or ignored? 
                                             # Our script drops 'state_id', keeps numeric.
    })

    # 2. Setup Dummy Artifacts
    # Imputer: If list_price is 0, fill with 999
    imputer_dict = {"median_list_price": 999}
    
    # Freq Map: 90210 appeared 50 times in training. 10001 is unknown.
    freq_map = {90210: 50}

    # 3. Run Logic
    processed = preprocess_new_data(raw_df, imputer_dict, freq_map)

    # 4. Assertions
    print(f"Processed Columns: {processed.columns.tolist()}")
    
    # Check Columns Dropped
    assert "median_sale_price" not in processed.columns
    assert "city" not in processed.columns
    assert "date" not in processed.columns
    assert "zipcode" not in processed.columns # We keep zipcode_freq, drop raw zip

    # Check Date Extraction
    assert "year" in processed.columns
    assert processed["month"].iloc[0] == 1

    # Check Imputation (The 0 should become 999)
    assert processed["median_list_price"].iloc[0] == 999
    assert processed["median_list_price"].iloc[1] == 500_000

    # Check Frequency Encoding
    # 90210 -> 50 (Known)
    # 10001 -> 0 (Unknown/Fillna)
    assert processed["zipcode_freq"].iloc[0] == 50
    assert processed["zipcode_freq"].iloc[1] == 0
    
    print("✅ Inference preprocessing logic passed")

# ==========================================
# 2. TEST FULL PIPELINE (Integration with Mocks)
# ==========================================
@patch("src.inference_pipeline.inference.load_artifacts") # Mock loading
@patch("src.inference_pipeline.inference.OUTPUT_DIR")     # Mock output path
def test_run_inference_e2e(mock_output_dir, mock_load_artifacts, tmp_path):
    """
    Tests the full run_inference flow.
    Mocks the model loading so we don't need real .pkl files.
    Redirects output to tmp_path.
    """
    # 1. Setup Mock Artifacts
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([100_000, 200_000]) # Fake predictions
    
    mock_imputer = {"median_list_price": 500}
    mock_freq_map = {90210: 10}
    
    # Configure the mock to return these when called
    mock_load_artifacts.return_value = (mock_model, mock_imputer, mock_freq_map)

    # 2. Setup Mock Output Path
    # We want the script to save predictions.csv to our temporary test folder
    mock_output_dir.__truediv__.return_value = tmp_path / "predictions.csv"

    # 3. Create Dummy Input CSV
    input_csv = tmp_path / "new_houses.csv"
    pd.DataFrame({
        "date": ["2023-01-01", "2023-01-02"],
        "zipcode": [90210, 90210],
        "median_list_price": [500_000, 600_000]
    }).to_csv(input_csv, index=False)

    # 4. Run Inference
    run_inference(input_path=input_csv)

    # 5. Assertions
    # Check if load_artifacts was called
    mock_load_artifacts.assert_called_once()
    
    # Check if model.predict was called
    mock_model.predict.assert_called_once()
    
    # Check if Output file was created
    assert (tmp_path / "predictions.csv").exists()
    
    # Verify contents
    results = pd.read_csv(tmp_path / "predictions.csv")
    assert "predicted_price" in results.columns
    assert results["predicted_price"].iloc[0] == 100_000
    
    print("✅ Full Inference Pipeline passed")
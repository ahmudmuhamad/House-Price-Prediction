import pytest
import pandas as pd
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# 1. Add Project Root to Path so imports work
sys.path.append(str(Path(__file__).parent.parent))

# 2. Import Pipeline Functions
from src.feature_pipeline.load import load_and_split_data
from src.feature_pipeline.clean import run_cleaning
from src.feature_pipeline.transform import run_transformation
from src.training_pipeline.train import train_model
from src.inference_pipeline.inference import run_inference

def test_smoke_end_to_end(tmp_path):
    """
    The Ultimate Sanity Check.
    Runs the full lifecycle on 10 rows of dummy data.
    Mocking MLflow prevents needing a server running.
    Redirects all file I/O to a temp directory.
    """
    print("\n💨 Starting Smoke Test...")

    # --- SETUP TEMP DIRECTORIES ---
    # We recreate your project structure inside a temp folder
    raw_dir = tmp_path / "data" / "raw"
    interim_dir = tmp_path / "data" / "interim"
    processed_dir = tmp_path / "data" / "processed"
    models_dir = tmp_path / "models"
    artifacts_dir = models_dir / "artifacts"
    predictions_dir = tmp_path / "data" / "predictions"
    
    for d in [raw_dir, interim_dir, processed_dir, models_dir, artifacts_dir, predictions_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # --- STEP 1: CREATE DUMMY RAW DATA ---
    df = pd.DataFrame({
        "date": pd.date_range("2018-01-01", periods=10, freq="180D"), # Spans 2018-2023
        "price": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
        "zipcode": [1000, 1000, 2000, 1000, 2000, 3000, 1000, 2000, 3000, 4000],
        "median_list_price": [100] * 10,
        "median_ppsf": [10] * 10,
        # Add columns that should be dropped (to test cleaning logic)
        "median_sale_price": [100] * 10, 
        "city": ["TestCity"] * 10,
        "Total Population": [5000] * 10
    })
    raw_path = raw_dir / "data.csv"
    df.to_csv(raw_path, index=False)

    # --- STEP 2: PATCH PATHS ---
    # We must trick the scripts into using our temp folders instead of the real ones
    with patch("src.feature_pipeline.clean.ARTIFACTS_DIR", artifacts_dir), \
         patch("src.feature_pipeline.transform.ARTIFACTS_DIR", artifacts_dir), \
         patch("src.training_pipeline.train.MODEL_DIR", models_dir), \
         patch("src.inference_pipeline.inference.MODEL_PATH", models_dir / "model.pkl"), \
         patch("src.inference_pipeline.inference.ARTIFACTS_DIR", artifacts_dir), \
         patch("src.inference_pipeline.inference.OUTPUT_DIR", predictions_dir), \
         patch("src.training_pipeline.train.mlflow") as mock_mlflow:

        # Mock MLflow to avoid connection errors
        mock_run = MagicMock()
        mock_run.info.run_id = "smoke_test_run"
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run
        mock_mlflow.active_run.return_value = mock_run

        # --- EXECUTE PIPELINE ---
        
        print("\n1️⃣ Running LOAD...")
        load_and_split_data(raw_path=raw_path, output_dir=interim_dir)
        assert (interim_dir / "train.csv").exists()

        print("2️⃣ Running CLEAN...")
        run_cleaning(input_dir=interim_dir, output_dir=interim_dir)
        assert (interim_dir / "train_cleaned.csv").exists()
        assert (artifacts_dir / "imputer.json").exists()

        print("3️⃣ Running TRANSFORM...")
        run_transformation(input_dir=interim_dir, output_dir=processed_dir)
        assert (processed_dir / "train_processed.csv").exists()
        assert (artifacts_dir / "freq_map.pkl").exists()

        print("4️⃣ Running TRAIN...")
        # Train uses the processed files we just made
        train_model(data_dir=processed_dir)
        assert (models_dir / "model.pkl").exists()

        print("5️⃣ Running INFERENCE...")
        # Run inference on the raw file we created at the start
        run_inference(input_path=raw_path)
        assert (predictions_dir / "predictions.csv").exists()

        # Check results
        results = pd.read_csv(predictions_dir / "predictions.csv")
        assert "predicted_price" in results.columns
        assert len(results) == 10
        
        print("\n✅ SMOKE TEST PASSED! The plumbing works.")
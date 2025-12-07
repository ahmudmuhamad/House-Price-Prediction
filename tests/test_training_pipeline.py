import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.training_pipeline.train import train_model
from src.training_pipeline.tune import run_tuning
from src.training_pipeline.eval import evaluate_model

@pytest.fixture
def dummy_data_dir(tmp_path):
    """Creates dummy processed data for testing."""
    # We match the numeric columns expected by the pipeline logic roughly
    df = pd.DataFrame({
        "price": [100, 200, 300, 400, 500],
        "zipcode_freq": [10, 20, 30, 40, 50],
        "median_list_price": [100, 100, 100, 100, 100],
        "year": [2020, 2020, 2020, 2020, 2020],
        "month": [1, 2, 3, 4, 5]
    })
    
    # Save as train, eval, and holdout
    df.to_csv(tmp_path / "train.csv", index=False)
    df.to_csv(tmp_path / "eval.csv", index=False)
    df.to_csv(tmp_path / "train_processed.csv", index=False)
    df.to_csv(tmp_path / "eval_processed.csv", index=False)
    df.to_csv(tmp_path / "holdout_processed.csv", index=False)
    
    return tmp_path

# ================================
# TEST: train.py (FIXED)
# ================================
@patch("src.training_pipeline.train.joblib.dump") # <--- STOP SAVE TO DISK
def test_train_model_runs(mock_dump, dummy_data_dir):
    """
    Test that train_model runs without error.
    Mocks joblib.dump so we DO NOT overwrite the real model.pkl
    """
    with patch("src.training_pipeline.train.mlflow") as mock_mlflow:
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id"
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run
        
        # Use small params for speed
        test_params = {"n_estimators": 2, "max_depth": 2}
        
        # Run function
        model = train_model(
            data_dir=dummy_data_dir,
            params=test_params,
            experiment_name="test_experiment"
        )
        
        # Assertions
        assert model is not None
        
        # Verify we TRIED to save (logic is correct), but nothing was written
        mock_dump.assert_called_once()
        print("✅ train_model test passed (Safe Mode)")

# ================================
# TEST: tune.py
# ================================
def test_tune_model_runs(dummy_data_dir):
    with patch("src.training_pipeline.tune.mlflow") as mock_mlflow:
        best_params = run_tuning(n_trials=1, data_dir=dummy_data_dir)
        assert isinstance(best_params, dict)
        print("✅ tune_model test passed")

# ================================
# TEST: eval.py
# ================================
def test_eval_model_logic(dummy_data_dir):
    with patch("src.training_pipeline.eval.mlflow") as mock_mlflow:
        mock_client = MagicMock()
        mock_version = MagicMock()
        mock_version.version = "1"
        mock_version.run_id = "test_run"
        mock_version.source = "mock_uri"
        
        with patch("src.training_pipeline.eval.MlflowClient", return_value=mock_client):
            mock_client.search_model_versions.return_value = [mock_version]
            
            fake_model = MagicMock()
            fake_model.predict.return_value = np.array([100, 200, 300, 400, 500])
            mock_mlflow.sklearn.load_model.return_value = fake_model
            
            metrics = evaluate_model(data_dir=dummy_data_dir)
            
            assert metrics["r2"] == 1.0
            print("✅ evaluate_model test passed")
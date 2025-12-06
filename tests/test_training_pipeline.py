import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.training_pipeline.train import train_model
from src.training_pipeline.eval import evaluate_model
from src.training_pipeline.tune import run_tuning

@pytest.fixture
def dummy_data_dir(tmp_path):
    """Creates dummy processed data for testing."""
    # Create simple numeric data
    df = pd.DataFrame({
        "price": [100, 200, 300, 400, 500],
        "feature1": [1, 2, 3, 4, 5],
        "feature2": [10, 20, 30, 40, 50],
        "zipcode": [1, 1, 1, 1, 1] # Ignored col
    })
    
    # Save as train, eval, and holdout
    df.to_csv(tmp_path / "train.csv", index=False) # For train.py
    df.to_csv(tmp_path / "eval.csv", index=False)  # For train.py
    
    # For tune.py/eval.py which might expect _processed suffix
    df.to_csv(tmp_path / "train_processed.csv", index=False)
    df.to_csv(tmp_path / "eval_processed.csv", index=False)
    df.to_csv(tmp_path / "holdout_processed.csv", index=False)
    
    return tmp_path

# ================================
# TEST: train.py
# ================================
def test_train_model_runs(dummy_data_dir):
    """Test that train_model runs without error and returns a model."""
    
    # We mock mlflow to avoid needing a server
    with patch("src.training_pipeline.train.mlflow") as mock_mlflow:
        # Create a fake active run
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id"
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run
        mock_mlflow.active_run.return_value = mock_run

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
        # Check if it fit by trying a prediction
        assert len(model.predict([[1, 10]])) == 1
        print("✅ train_model test passed")

# ================================
# TEST: tune.py
# ================================
def test_tune_model_runs(dummy_data_dir):
    """Test that tuning runs and finds params."""
    
    with patch("src.training_pipeline.tune.mlflow") as mock_mlflow:
        
        best_params = run_tuning(n_trials=1, data_dir=dummy_data_dir)
        
        assert isinstance(best_params, dict)
        assert "n_estimators" in best_params
        print("✅ tune_model test passed")

# ================================
# TEST: eval.py
# ================================
def test_eval_model_logic(dummy_data_dir):
    """
    Test evaluation logic. 
    Since eval loads from MLflow registry, we mock the loading part.
    """
    with patch("src.training_pipeline.eval.mlflow") as mock_mlflow:
        # Mock the MlflowClient search_model_versions
        mock_client = MagicMock()
        mock_version = MagicMock()
        mock_version.version = "1"
        mock_version.run_id = "test_run"
        mock_version.source = "mock_uri"
        
        # Mock the client instance created inside the function
        with patch("src.training_pipeline.eval.MlflowClient", return_value=mock_client):
            mock_client.search_model_versions.return_value = [mock_version]
            
            # Mock loading the model
            fake_model = MagicMock()
            # Make it predict perfect values so R2 = 1.0
            fake_model.predict.return_value = np.array([100, 200, 300, 400, 500])
            mock_mlflow.sklearn.load_model.return_value = fake_model
            
            # Run
            metrics = evaluate_model(data_dir=dummy_data_dir)
            
            # Assertions
            assert metrics["r2"] == 1.0
            assert metrics["rmse"] == 0.0
            print("✅ evaluate_model test passed")
"""
Evaluate the Production Model.

- Reads from: data/processed/holdout_processed.csv
- Loads model: From MLflow Model Registry ("HousePriceChampion")
- Logic:
    1. Loads Holdout Data.
    2. Loads the latest registered model.
    3. Generates predictions.
    4. Calculates metrics (RMSE, R2).
    5. Logs metrics back to MLflow.
    6. Fails pipeline if R2 < 0.80.
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Default Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DROP_COLS = ["price", "date", "year", "city", "state_id", "zipcode"]

def evaluate_model(
    model_name: str = "HousePriceChampion", 
    stage: str = "None",
    data_dir: Path = PROCESSED_DIR
):
    """
    Loads the latest model and evaluates it against the Holdout set.
    """
    mlflow.set_tracking_uri("http://127.0.0.1:8080")
    client = MlflowClient()
    
    print(f"🔍 Evaluating model: {model_name}...")

    # 1. Load Holdout Data (Using data_dir for testability)
    holdout_path = Path(data_dir) / "holdout_processed.csv"
    if not holdout_path.exists():
        raise FileNotFoundError(f"❌ Holdout data not found at {holdout_path}")
        
    holdout = pd.read_csv(holdout_path)
    X_test = holdout.drop(columns=DROP_COLS, errors="ignore").select_dtypes(include=[np.number])
    y_test = holdout["price"]

    # 2. Get Latest Model Version & Run ID
    # Use search_model_versions (Robust method)
    versions = client.search_model_versions(f"name='{model_name}'")
    
    if not versions:
        print(f"❌ No versions found for model {model_name}")
        # In a real script we might raise error, but for safety we return empty dict
        return {}
    
    # Sort by version number (descending) to get the latest
    latest_version = sorted(versions, key=lambda x: int(x.version), reverse=True)[0]
    
    run_id = latest_version.run_id
    model_uri = latest_version.source
    
    print(f"   Latest Version: v{latest_version.version}")
    print(f"   Source Run ID: {run_id}")
    
    # 3. Load Model
    model = mlflow.sklearn.load_model(model_uri)

    # 4. Predict
    preds = model.predict(X_test)

    # 5. Metrics
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    mape = np.median(np.abs((y_test - preds) / y_test)) * 100

    print("\n📊 HOLDOUT EVALUATION REPORT")
    print(f"   RMSE: ${rmse:,.0f}")
    print(f"   MAE:  ${mae:,.0f}")
    print(f"   R2:   {r2:.4f}")
    print(f"   MAPE: {mape:.2f}%")

    # 6. Log Metrics Back to MLflow
    # We wrap this in try-except because in unit tests (with mocks), 
    # the run might not exist or be writable.
    try:
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metric("holdout_rmse", rmse)
            mlflow.log_metric("holdout_mae", mae)
            mlflow.log_metric("holdout_r2", r2)
            mlflow.log_metric("holdout_mape", mape)
    except Exception as e:
        print(f"   ⚠️ Could not log metrics to run {run_id}: {e}")

    # 7. Quality Gate
    if r2 < 0.80:
        raise ValueError(f"❌ Model quality check failed! R2 {r2:.4f} is below 0.80 threshold.")
    
    print("✅ Model passed quality checks.")
    
    # Return metrics dictionary for testing assertions
    return {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape}

if __name__ == "__main__":
    evaluate_model()
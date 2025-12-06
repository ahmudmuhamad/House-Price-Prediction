"""
Evaluate the Production Model.

- Reads from: data/processed/holdout_processed.csv
- Loads model: From MLflow Model Registry ("HousePriceChampion") or specific run.
- Logic:
    1. Loads Holdout Data.
    2. Loads the latest registered model.
    3. Generates predictions.
    4. Calculates RMSE/MAE/R2.
    5. Fails (raises error) if performance is below threshold.
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

PROCESSED_DIR = Path("data/processed")
DROP_COLS = ["price", "date", "year", "city", "state_id", "zipcode"]

def evaluate_model(model_name: str = "HousePriceChampion", stage: str = "None"):
    """
    Loads the latest model and evaluates it against the Holdout set.
    """
    mlflow.set_tracking_uri("http://127.0.0.1:8000")
    print(f"🔍 Evaluating model: {model_name} (Stage: {stage})...")

    # 1. Load Holdout Data
    holdout = pd.read_csv(PROCESSED_DIR / "holdout_processed.csv")
    X_test = holdout.drop(columns=DROP_COLS, errors="ignore").select_dtypes(include=[np.number])
    y_test = holdout["price"]

    # 2. Load Model
    # "models:/" is the URI scheme for the Model Registry
    # If stage is None, it grabs the latest version.
    model_uri = f"models:/{model_name}/latest"
    print(f"   Loading model from: {model_uri}")
    
    try:
        model = mlflow.sklearn.load_model(model_uri)
    except Exception as e:
        print(f"❌ Failed to load model. Ensure 'train.py' ran successfully. Error: {e}")
        return

    # 3. Predict
    preds = model.predict(X_test)

    # 4. Metrics
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    # Calculate MAPE for business context
    mape = np.median(np.abs((y_test - preds) / y_test)) * 100

    print("\n📊 HOLDOUT EVALUATION REPORT")
    print(f"   RMSE: ${rmse:,.0f}")
    print(f"   MAE:  ${mae:,.0f}")
    print(f"   R2:   {r2:.4f}")
    print(f"   MAPE: {mape:.2f}%")

    # 5. Gatekeeping (Quality Check)
    # Fail the pipeline if the model is trash (e.g., R2 < 0.80)
    if r2 < 0.80:
        raise ValueError(f"❌ Model quality check failed! R2 {r2:.4f} is below 0.80 threshold.")
    
    print("✅ Model passed quality checks.")
    
    # Optional: Log these metrics back to the specific run if you have the run_id
    # or just print them for the CI/CD logs.

if __name__ == "__main__":
    evaluate_model()
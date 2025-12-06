"""
Train the Production Model.

- Reads from: data/processed/ (train_processed.csv + eval_processed.csv)
- Logic:
    1. Combines Train + Eval datasets (Maximize data for production).
    2. drops non-feature columns (date, year, raw zipcode).
    3. Trains the Champion XGBoost model (TransformedTargetRegressor).
    4. Logs to MLflow.
- Saves model to: models/ (and MLflow artifact store)
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from mlflow.models import infer_signature

# Setup Paths
PROCESSED_DIR = Path("data/processed")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# 🏆 Champion Parameters (Found via Optuna in Notebook 08)
CHAMPION_PARAMS = {
    'n_estimators': 766,
    'max_depth': 8,
    'learning_rate': 0.057,
    'subsample': 0.603,
    'colsample_bytree': 0.839,
    'min_child_weight': 1,
    'reg_alpha': 2.72e-05,
    'reg_lambda': 8.66,
    'random_state': 42,
    'n_jobs': -1,
    'tree_method': "hist"
}

# Features to DROP (Not used for training)
# Note: We keep 'month' for seasonality, but drop 'year' to avoid overfitting to specific years
# (The inflation signal is captured by 'median_list_price')
DROP_COLS = ["price", "date", "year", "city", "state_id", "zipcode"]

def load_and_combine_data(data_dir: Path):
    """Load train and eval, combine them for production training."""
    print("   Loading datasets...")
    train = pd.read_csv(data_dir / "train_processed.csv")
    eval_df = pd.read_csv(data_dir / "eval_processed.csv")
    
    # Concatenate for maximum data density
    full_df = pd.concat([train, eval_df], axis=0).reset_index(drop=True)
    
    # Separate Features (X) and Target (y)
    X = full_df.drop(columns=DROP_COLS, errors="ignore").select_dtypes(include=[np.number])
    y = full_df["price"]
    
    print(f"   Combined Data Shape: {X.shape}")
    return X, y

def train_model(
    data_dir: Path = PROCESSED_DIR,
    params: dict = CHAMPION_PARAMS,
    experiment_name: str = "House Price Prediction - Production"
):
    mlflow.set_tracking_uri("http://127.0.0.1:8000")
    mlflow.set_experiment(experiment_name)
    
    print("🚀 Starting Production Training...")
    
    X, y = load_and_combine_data(data_dir)
    
    with mlflow.start_run(run_name="Production_Train_Job"):
        # 1. Build Pipeline
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('regressor', XGBRegressor(**params))
        ])
        
        # 2. Wrap in Log-Transform (Critical for House Prices)
        final_model = TransformedTargetRegressor(
            regressor=pipeline,
            func=np.log1p,
            inverse_func=np.expm1
        )
        
        # 3. Train
        print(f"   Training XGBoost with params: {params}")
        final_model.fit(X, y)
        
        # 4. Log to MLflow
        mlflow.log_params(params)
        
        # Log Model Artifact
        # We infer signature using a sample of X
        signature = infer_signature(X.head(), final_model.predict(X.head()))
        
        model_info = mlflow.sklearn.log_model(
            sk_model=final_model,
            artifact_path="model",
            signature=signature,
            registered_model_name="HousePriceChampion" # Registers in Model Registry
        )
        
        print(f"✅ Model trained and logged to MLflow run: {mlflow.active_run().info.run_id}")
        print(f"   Model URI: {model_info.model_uri}")
        
    return final_model

if __name__ == "__main__":
    train_model()
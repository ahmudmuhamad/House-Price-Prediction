"""
Hyperparameter Tuning Script (Optuna).

- Reads from: data/processed/ (Train/Eval)
- Logic:
    1. Runs Optuna optimization on XGBoost.
    2. Logs results to MLflow nested runs.
    3. Prints the best parameters (to be copied into train.py or config).
"""

import optuna
import mlflow
import numpy as np
import pandas as pd
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error

PROCESSED_DIR = Path("data/processed")
DROP_COLS = ["price", "date", "year", "city", "state_id", "zipcode"]

def load_data():
    train = pd.read_csv(PROCESSED_DIR / "train_processed.csv")
    eval_df = pd.read_csv(PROCESSED_DIR / "eval_processed.csv")
    
    X_train = train.drop(columns=DROP_COLS, errors="ignore").select_dtypes(include=[np.number])
    y_train = train["price"]
    
    X_eval = eval_df.drop(columns=DROP_COLS, errors="ignore").select_dtypes(include=[np.number])
    y_eval = eval_df["price"]
    
    return X_train, y_train, X_eval, y_eval

def run_tuning(n_trials: int = 15):
    mlflow.set_tracking_uri("http://127.0.0.1:8000")
    mlflow.set_experiment("House Price Prediction - Tuning")
    
    X_train, y_train, X_eval, y_eval = load_data()
    
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "random_state": 42,
            "n_jobs": -1,
            "tree_method": "hist"
        }
        
        with mlflow.start_run(nested=True):
            pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('regressor', XGBRegressor(**params))
            ])
            
            model = TransformedTargetRegressor(
                regressor=pipeline, 
                func=np.log1p, 
                inverse_func=np.expm1
            )
            
            model.fit(X_train, y_train)
            preds = model.predict(X_eval)
            rmse = np.sqrt(mean_squared_error(y_eval, preds))
            
            mlflow.log_params(params)
            mlflow.log_metric("rmse", rmse)
            
        return rmse

    print("🚀 Starting Optuna Tuning...")
    with mlflow.start_run(run_name="Optuna_Study"):
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)
        
        print("\n🏆 Best Params Found:")
        print(study.best_params)
        mlflow.log_params(study.best_params)

if __name__ == "__main__":
    run_tuning(n_trials=10)
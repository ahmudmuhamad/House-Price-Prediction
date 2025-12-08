# Training Pipeline

The training pipeline is responsible for producing the Champion Model.

## Workflow
1.  **Load Data:** Combines `train` and `eval` sets for maximum data density.
2.  **Pipeline Construction:**
    * `SimpleImputer` (Median)
    * `XGBoostRegressor`
3.  **Target Transformation:** Uses `TransformedTargetRegressor` to log-transform the price, handling skewness.
4.  **Logging:** All metrics (RMSE, MAE, R2) and parameters are logged to **MLflow**.

## Hyperparameter Tuning
We use **Optuna** to optimize:
* `n_estimators`
* `learning_rate`
* `max_depth`

!!! info "Champion Parameters"
    The current champion model uses `n_estimators=766` and `max_depth=8`.

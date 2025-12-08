# Inference Pipeline

**File:** `src/inference_pipeline/inference.py`

The inference pipeline is designed for **Batch Prediction**. It processes a CSV of raw data and outputs predictions, ensuring consistency with the training environment.

## Workflow

```mermaid
graph TD
    Input[New Data CSV] --> Load[Load Artifacts]
    Load --> Clean[Clean <br> (Apply imputer.json)]
    Clean --> Transform[Transform <br> (Apply freq_map.pkl)]
    Transform --> Predict[Model.predict()]
    Predict --> Output[predictions.csv]
```

## Reproducibility Guarantee
To guarantee that inference matches training exactly, the pipeline **loads** the frozen artifacts instead of recalculating anything.

### 1. Artifact Loading
The script looks for these critical files:
*   `models/model.pkl`: The trained XGBoost model.
*   `models/artifacts/imputer.json`: The median values for filling 0s.
*   `models/artifacts/freq_map.pkl`: The zipcode frequency mapping.

### 2. Preprocessing
It applies the transformations:
*   **Imputation:** Fills 0s with the fixed values from `imputer.json`.
*   **Feature Engineering:** Extracts Month/Year and maps zipcodes using `freq_map.pkl`. Unseen zipcodes are handled gracefully (mapped to 0).

## Usage
The pipeline checks for input data. If none is provided, it defaults to using the `holdout.csv` to demonstrate performance on unseen data.

**Output Location:** `data/predictions/predictions.csv`

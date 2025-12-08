# Feature Pipeline

The feature pipeline ensures data consistency and prevents training-serving skew by rigorously separating "learning" (fitting) from "applying" (transforming).

## 1. Load & Split
**File:** `src/feature_pipeline/load.py`

We split the data strictly **chronologically** to simulate real-world forecasting. Random splitting is strictly avoided for time-series data.

*   **Train (2012-2019):** Used to learn parameters (medians, frequencies, model weights).
*   **Eval (2020-2021):** Used for hyperparameter tuning and model selection.
*   **Holdout (2022+):** Unseen "future" data for final validation.

## 2. Cleaning (Imputation)
**File:** `src/feature_pipeline/clean.py`

We handle missing values (often encoded as `0` in this dataset) by replacing them with the **Median**.

### Skew Prevention
Crucially, the median is calculated **only on the Training set**. This value is saved as an artifact (`imputer.json`) and applied blindly to Eval, Holdout, and Production data.

*   **Artifact:** `models/artifacts/imputer.json`
*   **Target Columns:** `median_list_price`, `median_ppsf`

## 3. Transformation (Feature Engineering)
**File:** `src/feature_pipeline/transform.py`

### Date Features
Raw dates are converted into:
*   `year`: Captures long-term inflation trends.
*   `month`: Captures seasonality (summer vs winter sales).

### Frequency Encoding
**High Cardinality Config:** `zipcode` has too many unique values for One-Hot Encoding.

We replace each zipcode with its **frequency (count)** in the training set.
*   **Train:** Mapped to count.
*   **Eval/Inference:** Mapped to the count observed in Train. If a zipcode wasn't seen in Train, it gets `0`.
*   **Artifact:** `models/artifacts/freq_map.pkl`

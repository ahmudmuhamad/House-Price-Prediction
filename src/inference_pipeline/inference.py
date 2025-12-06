"""
Batch Inference Script.

- Reads from: data/raw/inference_input.csv (or specified path)
- Logic:
    1. Loads Artifacts (Model, Imputer, Freq Map).
    2. Loads New Data.
    3. Cleans & Transforms (using saved rules).
    4. Generates Predictions.
- Writes to: data/predictions/predictions.csv
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

# ==========================================
# PATH SETUP
# ==========================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "model.pkl"
ARTIFACTS_DIR = PROJECT_ROOT / "models" / "artifacts"
DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "data" / "predictions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Config (Must match training!)
DROP_COLS = [
    "median_sale_price", 
    "Total Population", 
    "Total School Enrollment", 
    "city", 
    "city_full"
]

def load_artifacts():
    """Load model and preprocessing rules."""
    print("   Loading artifacts...")
    
    if not MODEL_PATH.exists():
        raise FileNotFoundError("❌ Model not found. Run 'src/training_pipeline/train.py' first.")
        
    model = joblib.load(MODEL_PATH)
    
    # Load Imputer (JSON)
    imputer_path = ARTIFACTS_DIR / "imputer.json"
    if not imputer_path.exists():
        raise FileNotFoundError("❌ Imputer artifact missing. Run 'src/feature_pipeline/clean.py' first.")
        
    with open(imputer_path, "r") as f:
        imputer_dict = json.load(f)
        
    # Load Freq Map (PKL)
    freq_map_path = ARTIFACTS_DIR / "freq_map.pkl"
    if not freq_map_path.exists():
        raise FileNotFoundError("❌ Frequency map missing. Run 'src/feature_pipeline/transform.py' first.")
        
    freq_map = joblib.load(freq_map_path)
    
    return model, imputer_dict, freq_map

def preprocess_new_data(df: pd.DataFrame, imputer_dict: dict, freq_map: dict) -> pd.DataFrame:
    """
    Apply the EXACT same transformations as training, but using saved artifacts.
    """
    df = df.copy()
    
    # 1. Drop unused columns (if they exist in input)
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    
    # 2. Fix Zeros (Using SAVED medians)
    for col, median_val in imputer_dict.items():
        if col in df.columns:
            # Handle explicit 0 and NaN
            df[col] = df[col].replace(0, np.nan)
            df[col] = df[col].fillna(median_val)
            
    # 3. Date Features
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        df = df.drop(columns=["date"])
        
    # 4. Frequency Encoding (Using SAVED map)
    # Map values; if zip not in map, fill with 0 (unseen location)
    if "zipcode" in df.columns:
        # Convert map keys to int if necessary (json saves keys as strings sometimes)
        # But here we load pickle so types are preserved
        df["zipcode_freq"] = df["zipcode"].map(freq_map).fillna(0)
        
    # 5. Filter for Numeric Columns Only (Safety)
    # Drop IDs that aren't features but were kept for tracking
    model_input = df.drop(columns=["zipcode", "state_id"], errors="ignore")
    model_input = model_input.select_dtypes(include=[np.number])
    
    return model_input

def run_inference(input_path: Path | str = None):
    print("🚀 Starting Inference Pipeline...")
    
    # 1. Setup Input Path (Default to holdout for testing if no new data provided)
    if input_path is None:
        # We use the raw holdout CSV as a "Test" input
        input_path = PROJECT_ROOT / "data" / "interim" / "holdout.csv"
        print(f"   No input provided. Using dummy data: {input_path}")
    
    if not Path(input_path).exists():
        raise FileNotFoundError(f"❌ Input file not found: {input_path}")

    # 2. Load Resources
    model, imputer, freq_map = load_artifacts()
    
    # 3. Load & Process Data
    raw_df = pd.read_csv(input_path)
    print(f"   Raw Data Shape: {raw_df.shape}")
    
    X_new = preprocess_new_data(raw_df, imputer, freq_map)
    print(f"   Processed Data Shape: {X_new.shape}")
    
    # 4. Predict
    preds = model.predict(X_new)
    
    # 5. Save Results
    # We attach predictions back to the raw data so you know which house is which
    results = raw_df.copy()
    results["predicted_price"] = preds
    
    # Optional: Calculate error if truth is available (for checking holdout)
    if "price" in results.columns:
        results["error"] = results["price"] - results["predicted_price"]
        results["abs_error_pct"] = (results["error"].abs() / results["price"]) * 100
        print(f"   Median Error on this batch: {results['abs_error_pct'].median():.2f}%")

    save_path = OUTPUT_DIR / "predictions.csv"
    results.to_csv(save_path, index=False)
    
    print(f"✅ Predictions saved to: {save_path}")
    print(f"   Sample Prediction: ${preds[0]:,.0f}")

if __name__ == "__main__":
    run_inference()
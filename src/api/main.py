from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import pandas as pd
import sys
import boto3
import joblib
import json
from pathlib import Path
from botocore.exceptions import NoCredentialsError, ClientError

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import existing pipeline logic
from src.inference_pipeline.inference import preprocess_new_data
from src.api.schemas import HousePredictionInput, PredictionOutput

# Global state
ml_resources = {}

# Config
BUCKET_NAME = "my-house-price-bucket-ml-project"
REGION = "eu-north-1"

def download_s3_artifacts():
    s3_client = boto3.client('s3', region_name=REGION)
    
    downloads = {
        "models/model.pkl": PROJECT_ROOT / "models" / "model.pkl",
        "models/artifacts/imputer.json": PROJECT_ROOT / "models" / "artifacts" / "imputer.json",
        "models/artifacts/freq_map.pkl": PROJECT_ROOT / "models" / "artifacts" / "freq_map.pkl",
        # We try to download whatever reference data exists
        "data/processed/train_cleaned.csv": PROJECT_ROOT / "data" / "processed" / "reference_train.csv" 
    }
    
    print(f"⬇️ Checking S3 Artifacts in bucket: {BUCKET_NAME}...")
    for s3_key, local_path in downloads.items():
        local_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            s3_client.head_object(Bucket=BUCKET_NAME, Key=s3_key)
            print(f"   Downloading {s3_key}...")
            s3_client.download_file(BUCKET_NAME, s3_key, str(local_path))
        except ClientError:
            print(f"   ⚠️ S3 Key {s3_key} not found. Continuing...")

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 App Startup...")
    
    # 1. Sync
    try:
        download_s3_artifacts()
    except Exception as e:
        print(f"⚠️ S3 Sync Failed: {e}")

    # 2. Load Resources
    print("📦 Loading ML Resources...")
    try:
        ml_resources["model"] = joblib.load(PROJECT_ROOT / "models" / "model.pkl")
        ml_resources["freq_map"] = joblib.load(PROJECT_ROOT / "models" / "artifacts" / "freq_map.pkl")
        with open(PROJECT_ROOT / "models" / "artifacts" / "imputer.json", "r") as f:
            ml_resources["imputer"] = json.load(f)
            
        # --- ROBUST REFERENCE DATA LOADING ---
        ref_path = PROJECT_ROOT / "data" / "processed" / "train_cleaned.csv"
        
        if ref_path.exists():
            print("   Loading Reference Data...")
            df_ref = pd.read_csv(ref_path)
            
            # A. Calculate Global Defaults (The Safety Net)
            # This ensures we ALWAYS have a row of valid numbers (Median Age, Income, etc.)
            # even if zipcode lookup fails.
            ml_resources["global_context"] = df_ref.median(numeric_only=True).to_frame().T
            
            # B. Check for Zipcode support
            if "zipcode" in df_ref.columns:
                ml_resources["reference_data"] = df_ref
                print("   ✅ Reference data loaded with Zipcode support.")
            else:
                print("   ⚠️ Reference data missing 'zipcode'. API will use Global Averages for context.")
                ml_resources["reference_data"] = None
        else:
            print("   ❌ Reference data missing! Creating empty fallback.")
            ml_resources["global_context"] = pd.DataFrame()
            ml_resources["reference_data"] = None
            
        print("✅ All resources loaded.")
    except Exception as e:
        print(f"❌ Failed to load resources: {e}")
        
    yield
    ml_resources.clear()

app = FastAPI(title="House Price Estimator API", lifespan=lifespan)

@app.get("/health")
def health_check():
    if "model" not in ml_resources:
        raise HTTPException(503, "Model not loaded")
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionOutput)
def predict_price(input_data: HousePredictionInput):
    if "model" not in ml_resources:
        raise HTTPException(503, "Model not initialized")

    try:
        input_dict = input_data.model_dump()
        target_zip = input_dict["zipcode"]
        
        # --- Context Enrichment Logic ---
        ref_df = ml_resources.get("reference_data")
        global_ctx = ml_resources.get("global_context")
        
        context_row = pd.DataFrame() # Start empty

        # Try Specific Lookup first
        if ref_df is not None:
            zip_context = ref_df[ref_df["zipcode"] == target_zip]
            if not zip_context.empty:
                context_row = zip_context.median(numeric_only=True).to_frame().T
        
        # Fallback to Global if specific lookup failed or wasn't possible
        if context_row.empty and global_ctx is not None:
            context_row = global_ctx.copy()
        
        # --- Overwrite with User Inputs ---
        # Crucial: We must overwrite the "Average" values with the User's specific values
        context_row["zipcode"] = target_zip
        context_row["median_list_price"] = input_dict["list_price"]
        context_row["date"] = input_dict["date"]
        
        # Drop target if it leaked in via the reference median
        if "price" in context_row.columns:
            context_row = context_row.drop(columns=["price"])

        # --- Pipeline ---
        processed_df = preprocess_new_data(
            context_row, 
            ml_resources["imputer"], 
            ml_resources["freq_map"]
        )

        prediction = ml_resources["model"].predict(processed_df)[0]

        return {"predicted_price": float(prediction)}

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction Error: {str(e)}")
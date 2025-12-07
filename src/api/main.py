from fastapi import FastAPI
from contextlib import asynccontextmanager
from pathlib import Path
import sys
import boto3
import joblib
import json
import pandas as pd
from botocore.exceptions import ClientError

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import the shared state and the v1 router
from src.api.state import ml_resources
from src.api.v1.endpoints import router as v1_router

# Config
BUCKET_NAME = "my-house-price-bucket-ml-project"
REGION = "eu-north-1"

def download_s3_artifacts():
    s3_client = boto3.client('s3', region_name=REGION)
    downloads = {
        "models/model.pkl": PROJECT_ROOT / "models" / "model.pkl",
        "models/artifacts/imputer.json": PROJECT_ROOT / "models" / "artifacts" / "imputer.json",
        "models/artifacts/freq_map.pkl": PROJECT_ROOT / "models" / "artifacts" / "freq_map.pkl",
        "data/processed/train_cleaned.csv": PROJECT_ROOT / "data" / "processed" / "train_cleaned.csv" 
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
    try:
        download_s3_artifacts()
    except Exception as e:
        print(f"⚠️ S3 Sync Warning: {e}")

    print("📦 Loading ML Resources...")
    try:
        # Load artifacts into the shared dictionary from state.py
        ml_resources["model"] = joblib.load(PROJECT_ROOT / "models" / "model.pkl")
        ml_resources["freq_map"] = joblib.load(PROJECT_ROOT / "models" / "artifacts" / "freq_map.pkl")
        with open(PROJECT_ROOT / "models" / "artifacts" / "imputer.json", "r") as f:
            ml_resources["imputer"] = json.load(f)
            
        ref_path = PROJECT_ROOT / "data" / "processed" / "reference_train.csv"
        if ref_path.exists():
            df_ref = pd.read_csv(ref_path)
            ml_resources["global_context"] = df_ref.median(numeric_only=True).to_frame().T
            if "zipcode" in df_ref.columns:
                ml_resources["reference_data"] = df_ref
            else:
                ml_resources["reference_data"] = None
        else:
            ml_resources["global_context"] = pd.DataFrame()
            
        print("✅ Resources loaded into shared state.")
    except Exception as e:
        print(f"❌ Failed to load resources: {e}")
        
    yield
    ml_resources.clear()

# Initialize App
app = FastAPI(title="House Price Estimator API", lifespan=lifespan)

# --- MOUNT VERSIONS ---
# This prefixes all routes in v1_router with /api/v1
app.include_router(v1_router, prefix="/api/v1", tags=["v1"])

@app.get("/")
def root():
    return {"message": "Welcome to House Price API. Use /api/v1/predict"}
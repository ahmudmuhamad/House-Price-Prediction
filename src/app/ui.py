import streamlit as st
import pandas as pd
import requests
import boto3
import os
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# ============================
# 1. CONFIG & SETUP
# ============================
load_dotenv()

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/api/v1/predict")
S3_BUCKET = "my-house-price-bucket-ml-project"
REGION = os.getenv("AWS_DEFAULT_REGION", "eu-north-1")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
LOCAL_DATA_DIR = BASE_DIR / "data" / "processed"
LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)

s3 = boto3.client("s3", region_name=REGION)

def load_from_s3(s3_key, local_filename):
    """Download file from S3 if it doesn't exist locally."""
    local_path = LOCAL_DATA_DIR / local_filename
    if not local_path.exists():
        st.info(f"📥 Downloading {s3_key} from S3...")
        try:
            s3.download_file(S3_BUCKET, s3_key, str(local_path))
            st.success("Download complete!")
        except Exception as e:
            st.error(f"Failed to download {s3_key}: {e}")
            st.stop()
    return local_path

# ============================
# 2. DATA LOADING (Reference Only)
# ============================
# We still load this to get the list of valid Zipcodes for the dropdown
HOLDOUT_PATH = load_from_s3(
    "data/processed/holdout_cleaned.csv", 
    "holdout_for_ui.csv"
)

@st.cache_data
def load_valid_zips():
    df = pd.read_csv(HOLDOUT_PATH)
    if "zipcode" in df.columns:
        # Clean and sort unique zipcodes
        zips = sorted(df["zipcode"].dropna().astype(int).unique())
        return zips
    return []

valid_zips = load_valid_zips()

# ============================
# 3. USER INPUTS
# ============================
st.title("🏡 House Price Predictor")
st.markdown("Enter property details below to get an AI-powered valuation.")

col1, col2 = st.columns(2)

with col1:
    # Zipcode Dropdown (Populated from S3 data)
    if valid_zips:
        selected_zip = st.selectbox("Zip Code", valid_zips)
    else:
        # Fallback if data is missing 'zipcode' column
        selected_zip = st.number_input("Zip Code", value=90210, step=1)

    # Date Input
    selected_date = st.date_input("Listing Date", datetime.today())

with col2:
    # Price Input
    list_price = st.number_input("Listing Price ($)", min_value=10000, value=500000, step=10000)

# ============================
# 4. SINGLE PREDICTION LOGIC
# ============================
if st.button("Predict Price 🚀", type="primary"):
    
    # 1. Prepare Payload (Single Item)
    payload = {
        "zipcode": int(selected_zip),
        "list_price": float(list_price),
        "date": selected_date.strftime("%Y-%m-%d")
    }
    
    st.write("Sending data to API...", payload)
    
    try:
        # 2. Call API
        resp = requests.post(API_URL, json=payload)
        
        if resp.status_code == 200:
            result = resp.json()
            predicted_price = result["predicted_price"]
            
            # 3. Display Result
            st.success("Prediction Successful!")
            
            st.metric(
                label="Estimated Sale Price", 
                value=f"${predicted_price:,.0f}",
                delta=f"${predicted_price - list_price:,.0f} vs List Price"
            )
            
        else:
            st.error(f"API Error ({resp.status_code}): {resp.text}")
            
    except Exception as e:
        st.error(f"Connection Error: {e}")
        st.warning("Ensure the FastAPI server is running: `uvicorn src.api.main:app --reload`")
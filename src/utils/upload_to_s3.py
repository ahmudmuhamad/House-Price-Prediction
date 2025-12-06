import boto3
import os
from pathlib import Path
from botocore.exceptions import NoCredentialsError

# ==========================================
# CONFIGURATION
# ==========================================
BUCKET_NAME = "my-house-price-bucket-ml-project"
REGION = "Europe (Stockholm) eu-north-1"                        

# Define what to upload and where it goes in S3
UPLOAD_RULES = {
    # Local Path : S3 Folder Key
    "data/raw/data.csv": "data/raw/data.csv",
    "data/processed/train_processed.csv": "data/processed/train_processed.csv",
    "data/processed/eval_processed.csv": "data/processed/eval_processed.csv",
    "data/processed/holdout_processed.csv": "data/processed/holdout_processed.csv",
    "models/artifacts/imputer.json": "artifacts/imputer.json",
    "models/artifacts/freq_map.pkl": "artifacts/freq_map.pkl"
}

def upload_files():
    # Setup Paths
    project_root = Path(__file__).resolve().parent.parent.parent
    s3_client = boto3.client('s3', region_name=REGION)

    print(f"🚀 Starting Upload to S3 Bucket: {BUCKET_NAME}...")

    for local_rel_path, s3_key in UPLOAD_RULES.items():
        local_path = project_root / local_rel_path
        
        if not local_path.exists():
            print(f"⚠️ Skipping missing file: {local_rel_path}")
            continue

        try:
            print(f"   Uploading {local_rel_path} -> s3://{BUCKET_NAME}/{s3_key}")
            s3_client.upload_file(str(local_path), BUCKET_NAME, s3_key)
        except NoCredentialsError:
            print("❌ Error: AWS Credentials not found.")
            return
        except Exception as e:
            print(f"❌ Error uploading {local_path}: {e}")

    print("\n✅ Upload Complete!")

if __name__ == "__main__":
    # Make sure you have set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY env vars
    # or configured them via 'aws configure' in terminal
    upload_files()
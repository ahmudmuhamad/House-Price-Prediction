from fastapi import APIRouter, HTTPException
import pandas as pd
from src.api.schemas import HousePredictionInput, PredictionOutput
from src.api.state import ml_resources # Import shared state
from src.inference_pipeline.inference import preprocess_new_data

# Create a router (like a mini-app)
router = APIRouter()

@router.get("/health")
def health_check():
    if "model" not in ml_resources:
        raise HTTPException(503, "Model not loaded")
    return {"status": "healthy", "version": "v1"}

@router.post("/predict", response_model=PredictionOutput)
def predict_price(input_data: HousePredictionInput):
    if "model" not in ml_resources:
        raise HTTPException(503, "Model not initialized")

    try:
        input_dict = input_data.model_dump()
        target_zip = input_dict["zipcode"]
        
        # --- Context Enrichment ---
        ref_df = ml_resources.get("reference_data")
        global_ctx = ml_resources.get("global_context")
        context_row = pd.DataFrame() 

        # Lookup Logic
        if ref_df is not None:
            zip_context = ref_df[ref_df["zipcode"] == target_zip]
            if not zip_context.empty:
                context_row = zip_context.median(numeric_only=True).to_frame().T
        
        if context_row.empty and global_ctx is not None:
            context_row = global_ctx.copy()
        
        # --- Overwrite User Inputs ---
        context_row["zipcode"] = target_zip
        context_row["median_list_price"] = input_dict["list_price"]
        context_row["date"] = input_dict["date"]
        
        if "price" in context_row.columns:
            context_row = context_row.drop(columns=["price"])

        # --- Preprocessing & Prediction ---
        processed_df = preprocess_new_data(
            context_row, 
            ml_resources["imputer"], 
            ml_resources["freq_map"]
        )

        prediction = ml_resources["model"].predict(processed_df)[0]

        return {"predicted_price": float(prediction)}

    except Exception as e:
        # In v1, we print errors. In v2, you might log them to CloudWatch.
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction Error: {str(e)}")
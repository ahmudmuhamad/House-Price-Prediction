from pydantic import BaseModel, Field
from datetime import datetime

class HousePredictionInput(BaseModel):
    zipcode: int = Field(..., example=90210, description="5-digit Zip Code")
    list_price: float = Field(..., gt=0, example=500000, description="Listing Price in USD")
    date: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"), description="Date of listing (YYYY-MM-DD)")
    
    
class PredictionOutput(BaseModel):
    predicted_price: float
    currency: str = "USD"

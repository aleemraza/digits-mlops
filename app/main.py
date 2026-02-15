from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from typing import List
import numpy as np
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Mock model for demonstration
class MockModel:
    def predict(self, X):
        # Simple mock: return first digit based on some feature
        return np.array([int(np.mean(x) % 10) for x in X])

model = MockModel()

# Pydantic v2 Input Model
class InputData(BaseModel):
    """Input data for digit prediction"""
    features: List[float] = Field(
        ..., 
        min_length=64, 
        max_length=64,
        description="64 features (8x8 pixels) representing a handwritten digit"
    )
    
    @field_validator('features')
    @classmethod
    def validate_features_range(cls, v: List[float]) -> List[float]:
        """Validate pixel values are in 0-16 range"""
        if any(x < 0 or x > 16 for x in v):
            raise ValueError('Pixel values must be between 0 and 16')
        return v
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "features": [0.0, 0.0, 5.0, 13.0, 9.0, 1.0, 0.0, 0.0, 0.0, 0.0, 13.0, 15.0, 10.0, 15.0, 5.0, 0.0, 0.0, 3.0, 15.0, 2.0, 0.0, 11.0, 8.0, 0.0, 0.0, 4.0, 12.0, 0.0, 0.0, 8.0, 8.0, 0.0, 0.0, 5.0, 8.0, 0.0, 0.0, 9.0, 8.0, 0.0, 0.0, 4.0, 11.0, 0.0, 1.0, 12.0, 7.0, 0.0, 0.0, 2.0, 14.0, 5.0, 10.0, 12.0, 0.0, 0.0, 0.0, 0.0, 6.0, 13.0, 10.0, 0.0, 0.0, 0.0]
                }
            ]
        }
    }

class BatchInputData(BaseModel):
    """Batch input data for prediction"""
    batch_features: List[List[float]] = Field(
        ...,
        description="List of feature vectors, each with 64 values"
    )
    
    @field_validator('batch_features')
    @classmethod
    def validate_batch_features(cls, v: List[List[float]]) -> List[List[float]]:
        """Validate each feature vector in the batch"""
        for i, features in enumerate(v):
            if len(features) != 64:
                raise ValueError(f'Features at index {i} must have exactly 64 elements, got {len(features)}')
            if any(x < 0 or x > 16 for x in features):
                raise ValueError(f'Pixel values at index {i} must be between 0 and 16')
        return v

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict")
def predict(data: InputData):
    start = time.time()
    
    try:
        # Convert to numpy array and reshape
        features = np.array(data.features).reshape(1, -1)
        prediction = model.predict(features)
        latency = round(time.time() - start, 4)
        
        logger.info(f"Prediction: {prediction[0]}, Latency: {latency}s")
        
        return {
            "prediction": int(prediction[0]),
            "latency": latency,
            "model_stage": "Production"
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
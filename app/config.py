from pydantic import BaseSettings

class Settings(BaseSettings):
    """Application settings"""
    app_name: str = "MLflow Model API"
    mlflow_tracking_uri: str = "http://localhost:5000"
    model_name: str = "DigitsClassifier"
    model_stage: str = "Production"
    
    class Config:
        env_file = ".env"

settings = Settings()
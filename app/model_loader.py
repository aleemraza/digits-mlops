import mlflow
import mlflow.sklearn
from mlflow.exceptions import MlflowException
import logging

logger = logging.getLogger(__name__)

def load_model():
    """Load MLflow model from registry"""
    try:
        # MLflow tracking URI
        mlflow.set_tracking_uri("http://localhost:5000")
        
        MODEL_NAME = "DigitsClassifier"
        
        # Try to load from production stage
        try:
            logger.info(f"Loading model '{MODEL_NAME}' from Production stage...")
            model_uri = f"models:/{MODEL_NAME}/Production"
            model = mlflow.sklearn.load_model(model_uri)
            logger.info("✅ Model loaded from Production stage")
            return model
        except MlflowException:
            # Fallback to latest version
            logger.warning("Production stage not found, loading latest version...")
            model_uri = f"models:/{MODEL_NAME}/latest"
            model = mlflow.sklearn.load_model(model_uri)
            logger.info("✅ Model loaded from latest version")
            return model
            
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        # For development, you can load from local path
        try:
            logger.info("Trying to load model from local artifacts...")
            # Adjust path based on your local structure
            model = mlflow.sklearn.load_model("mlruns/0/<run_id>/artifacts/model")
            logger.info("✅ Model loaded from local artifacts")
            return model
        except Exception as local_error:
            logger.error(f"❌ Failed to load local model: {local_error}")
            raise
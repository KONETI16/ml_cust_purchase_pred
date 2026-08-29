from fastapi import FastAPI, HTTPException
import pickle
import numpy as np
import logging
import mlflow
import os
from pydantic import BaseModel, Field

# Configure the standard production logging to console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ProductionLogger")

# Configure MLflow tracking dynamically from environment (works in Docker Compose)
# Dynamically fall back to local execution if the environment variable isn't injected
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
try:
    if TRACKING_URI:
        mlflow.set_tracking_uri(TRACKING_URI)
        # Use a stable experiment name for inference monitoring
        mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT", "Customer_Inference_Monitoring"))
        USE_MLFLOW = True
        logger.info("MLflow configured. Tracking URI: %s, Experiment: %s", TRACKING_URI, os.getenv("MLFLOW_EXPERIMENT", "Customer_Inference_Monitoring"))
    else:
        USE_MLFLOW = False
        logger.info("MLFLOW_TRACKING_URI explicitly empty; MLflow logging disabled.")
except Exception as e:
    USE_MLFLOW = False
    logger.warning("Failed to configure MLflow; MLflow logging disabled. Error: %s", e)


class CustomerData(BaseModel):
    age: int = Field(..., description="Age of the customer", example=30, ge=18, le=100)
    salary: int = Field(..., description="Salary of the customer", example=50000, ge=0)


app = FastAPI(name="Customer Purchase Prediction API", description="API for predicting customer purchase behavior", version="1.0.0")

PIPELINE_PATH = "pipeline.pkl"


@app.on_event("startup")
def load_pipeline():
    """Load the pipeline artifact once at startup and keep it in app.state.model."""
    try:
        with open(PIPELINE_PATH, 'rb') as f:
            model = pickle.load(f)
        app.state.model = model
        logger.info("Machine learning pipeline successfully loaded into memory at startup.")
    except FileNotFoundError:
        app.state.model = None
        logger.error(f"Pipeline file not found at {PIPELINE_PATH} during startup.")
    except Exception as e:
        app.state.model = None
        logger.critical(f"Failed to load pipeline artifact at startup: {str(e)}")


@app.get("/", tags=["Root"])
def root():
    return {"message": "Customer Purchase Prediction API is running.", "endpoints": ["/health", "/predict"]}


@app.get("/health", tags=["Health Check"])
def health_check():
    return {"status": "healthy"}


async def _predict_logic(age: int, salary: int):
    logger.info(f"Incoming prediction request received - Age: {age}, Salary: {salary}")

    model = getattr(app.state, "model", None)
    if model is None:
        logger.error("Model pipeline is not loaded.")
        raise HTTPException(status_code=500, detail="Model not available")

    raw_input = np.array([[age, salary]])

    try:
        prediction = model.predict(raw_input)
        pred_label = int(prediction[0])
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(e))

    # Attempt to get probabilities in a robust way
    purchase_prob = None
    probs = None
    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(raw_input)[0]
            buy_index = None
            if hasattr(model, "classes_"):
                try:
                    buy_index = int(np.where(model.classes_ == 1)[0][0])
                except Exception:
                    buy_index = None
            if buy_index is None:
                buy_index = int(np.argmax(probs))
            purchase_prob = float(probs[buy_index])
        except Exception:
            purchase_prob = None

    # Log to MLflow if configured
    if USE_MLFLOW:
        try:
            with mlflow.start_run(run_name="Customer_Purchase_Prediction_Inference"):
                active = mlflow.active_run()
                run_id = active.info.run_id if active is not None else "unknown"
                mlflow.log_param("customer_age", int(age))
                mlflow.log_param("customer_salary", int(salary))
                mlflow.log_metric("prediction_outcome", float(pred_label))
                if purchase_prob is not None:
                    mlflow.log_metric("purchase_probability", float(purchase_prob))
                logger.info("MLflow run recorded: %s", run_id)
        except Exception:
            logger.exception("MLflow logging failed; continuing without blocking response.")

    logger.info(f"Prediction successful. Output Class: {pred_label}")

    if purchase_prob is not None:
        confidence = f"{purchase_prob:.2%}"
    else:
        confidence = "N/A"

    if pred_label == 1:
        result = {"prediction": "Customer will buy the Product", "confidence_level": confidence}
    else:
        result = {"prediction": "Customer will not buy the Product", "confidence_level": confidence}

    return result


@app.get("/predict", tags=["Prediction"])
async def predict_get(age: int, salary: int):
    return await _predict_logic(age, salary)


@app.post("/predict", tags=["Prediction"])
async def predict_post(data: CustomerData):
    return await _predict_logic(data.age, data.salary)
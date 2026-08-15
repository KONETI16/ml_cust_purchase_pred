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

# Make MLflow configuration optional via env var to support containerized runs.
# If the variable is unset, default to the local tracking server used by the
# development workflow. An explicit empty string still disables MLflow.
_tracking_uri_env = os.environ.get("MLFLOW_TRACKING_URI")
if _tracking_uri_env is None:
    MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
    logger.info("MLFLOW_TRACKING_URI not set; defaulting to local MLflow server at %s", MLFLOW_TRACKING_URI)
elif _tracking_uri_env.strip():
    MLFLOW_TRACKING_URI = _tracking_uri_env.strip()
    logger.info("MLFLOW_TRACKING_URI set to %s", MLFLOW_TRACKING_URI)
else:
    MLFLOW_TRACKING_URI = ""
    logger.info("MLFLOW_TRACKING_URI explicitly empty; MLflow logging disabled.")

USE_MLFLOW = bool(MLFLOW_TRACKING_URI)

if USE_MLFLOW:
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT", "Customer_Purchase_Prediction_Experiment"))
        logger.info("MLflow configured. Tracking URI: %s, Experiment: %s", MLFLOW_TRACKING_URI, os.getenv("MLFLOW_EXPERIMENT", "Customer_Purchase_Prediction_Experiment"))
    except Exception as e:
        logger.warning("Failed to configure MLflow tracking server; MLflow logging disabled. Error: %s", e)
        USE_MLFLOW = False
else:
    logger.info("MLflow tracking URI not set; MLflow logging disabled.")


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
# ml_cust_purchase_pred

Customer purchase prediction API using FastAPI and a pickled machine learning pipeline.

## Features

- Health check endpoint: `GET /health`
- Prediction endpoint: `GET /predict?age=<age>&salary=<salary>`
- Prediction endpoint: `POST /predict` with JSON body
- Optional MLflow logging for inference tracking

## Local development

1. Activate your Python virtual environment

```bash
source .venv/Scripts/activate
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Start a local MLflow tracking server

```bash
python -m mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000
```

4. Run the app locally

```bash
uvicorn app:app --host 0.0.0.0 --port 5000
```

5. Call the API

```bash
curl "http://localhost:5000/predict?age=50&salary=90000"
```

Or with POST:

```bash
curl -X POST "http://localhost:5000/predict" -H "Content-Type: application/json" -d '{"age": 50, "salary": 90000}'
```

## Docker

Build the Docker image:

```bash
docker build -t customer-api .
```

Start the MLflow tracking server on the host first:

```bash
python -m mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000
```

Run the container with the host machine IP that your Docker container can reach:

```bash
docker run -d -p 8000:8000 -e MLFLOW_TRACKING_URI="http://192.168.29.192:5000" --name ml_service2 customer-api
```

Then access the API at:

```bash
http://localhost:8000/predict?age=50&salary=90000
```

## MLflow UI

Open the experiment runs page in your browser at:

```bash
http://localhost:5000/#/experiments/1/runs?startTime=ALL
```

Use `localhost:5000` for the browser UI and the host IP `192.168.29.192:5000` for the container's `MLFLOW_TRACKING_URI`.

## Notes

- `pipeline.pkl` must be present in the project root and copied into the container.
- The `Dockerfile` uses Python 3.12.
- The app loads the model at startup and supports both query-string and JSON-based prediction requests.

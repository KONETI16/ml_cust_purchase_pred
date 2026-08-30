import numpy as np
import time
import pytest
import sys
import types
from fastapi.testclient import TestClient

# Stub mlflow to avoid importing heavy external dependency during tests
mlflow_stub = types.ModuleType("mlflow")
mlflow_stub.set_tracking_uri = lambda uri: None
mlflow_stub.set_experiment = lambda name: None
def _start_run(*args, **kwargs):
    class DummyRun:
        info = types.SimpleNamespace(run_id="dummy")
    class Ctx:
        def __enter__(self):
            return DummyRun()
        def __exit__(self, exc_type, exc, tb):
            return False
    return Ctx()
mlflow_stub.start_run = _start_run
mlflow_stub.active_run = lambda: types.SimpleNamespace(info=types.SimpleNamespace(run_id="dummy"))
mlflow_stub.log_param = lambda *a, **k: None
mlflow_stub.log_metric = lambda *a, **k: None
sys.modules['mlflow'] = mlflow_stub

# Ensure project root is on sys.path so tests can import app
import os
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from app import app


class MockModel:
    def __init__(self):
        self.classes_ = np.array([0, 1])

    def predict(self, X):
        return np.array([1])

    def predict_proba(self, X):
        return np.array([[0.2, 0.8]])


@pytest.fixture(scope="session")
def client():
    # Override any heavy model loaded at startup with a lightweight mock
    app.state.model = MockModel()
    with TestClient(app) as c:
        yield c

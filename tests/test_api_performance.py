import time
import pytest


def assert_latency_within(ms_threshold, elapsed_seconds):
    assert elapsed_seconds * 1000 <= ms_threshold, f"Response time {elapsed_seconds*1000:.1f}ms exceeded {ms_threshold}ms"


@pytest.mark.performance
def test_health_check_latency(client):
    start = time.perf_counter()
    resp = client.get("/health")
    elapsed = time.perf_counter() - start
    assert resp.status_code == 200
    assert_latency_within(100, elapsed)


@pytest.mark.performance
def test_predict_get_latency_and_response(client):
    params = {"age": 30, "salary": 50000}
    start = time.perf_counter()
    resp = client.get("/predict", params=params)
    elapsed = time.perf_counter() - start
    assert resp.status_code == 200
    body = resp.json()
    assert "prediction" in body
    assert "confidence_level" in body
    assert_latency_within(200, elapsed)


@pytest.mark.performance
def test_predict_post_latency_and_response(client):
    payload = {"age": 40, "salary": 70000}
    start = time.perf_counter()
    resp = client.post("/predict", json=payload)
    elapsed = time.perf_counter() - start
    assert resp.status_code == 200
    body = resp.json()
    assert "prediction" in body
    assert "confidence_level" in body
    assert_latency_within(250, elapsed)

import time

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from coupon.serve import create_app, load_assignments


@pytest.fixture
def client(tmp_path):
    p = tmp_path / "a.parquet"
    pd.DataFrame({"user_id": [1, 2], "coupon_amount": [5.0, 0.0],
                  "expected_uplift": [1.2, 0.0], "expected_cost": [2.0, 0.0]}).to_parquet(p)
    return TestClient(create_app(p))


def test_health(client):
    assert client.get("/health").json() == {"ready": True, "n_users": 2}


def test_known_user(client):
    r = client.get("/coupon/1").json()
    assert r["coupon_amount"] == 5.0 and r["reason"] == "model"


def test_unknown_user_defaults_to_no_coupon(client):
    r = client.get("/coupon/999").json()
    assert r["coupon_amount"] == 0.0 and r["reason"] == "not_scored"


def test_invalid_user_id(client):
    assert client.get("/coupon/abc").status_code == 422


def test_missing_file_not_ready(tmp_path):
    assert load_assignments(tmp_path / "nope.parquet") == {}
    assert TestClient(create_app(tmp_path / "nope.parquet")).get("/health").json()["ready"] is False


def test_latency_under_200ms(client):
    client.get("/coupon/1")
    t0 = time.perf_counter()
    for _ in range(100):
        client.get("/coupon/1")
    assert (time.perf_counter() - t0) / 100 < 0.2

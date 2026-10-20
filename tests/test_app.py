import pytest
from app import app
from src.features import load_data, add_features
from src.predict import predict_sales


@pytest.fixture
def client():
    return app.test_client()


def test_data_clean():
    df = load_data()
    assert df.shape[1] == 4 and not df.isna().any().any()


def test_feature_engineering():
    assert "TV_Radio" in add_features(load_data()).columns


def test_prediction_reasonable():
    assert 15 < predict_sales(230.1, 37.8, 69.2) < 29  # actual: 22.1


def test_more_tv_more_sales():
    assert predict_sales(250, 20, 20) > predict_sales(50, 20, 20)


def test_negative_rejected():
    with pytest.raises(ValueError):
        predict_sales(-1, 0, 0)


def test_health(client):
    assert client.get("/health").json == {"status": "ok"}


def test_api_ok(client):
    r = client.post("/api/predict", json={"tv": 100, "radio": 20, "newspaper": 10})
    assert r.status_code == 200 and "predicted_sales" in r.json


def test_api_bad_input(client):
    assert client.post("/api/predict", json={"tv": "x"}).status_code == 400

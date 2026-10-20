"""Prediction helpers (CLI + importable)."""
import argparse
import json
import joblib
import pandas as pd
from src.features import METRICS_PATH, MODEL_PATH, add_features

_model = None


def get_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


def predict_sales(tv: float, radio: float, newspaper: float) -> float:
    """Predict sales (thousand units) for ad spend in thousand dollars."""
    for name, v in (("TV", tv), ("Radio", radio), ("Newspaper", newspaper)):
        if v < 0:
            raise ValueError(f"{name} spend cannot be negative")
    X = add_features(pd.DataFrame([{"TV": tv, "Radio": radio, "Newspaper": newspaper}]))
    return float(get_model().predict(X)[0])


def load_metrics() -> dict:
    return json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Predict sales from ad spend")
    p.add_argument("--tv", type=float, required=True)
    p.add_argument("--radio", type=float, required=True)
    p.add_argument("--newspaper", type=float, required=True)
    a = p.parse_args()
    print(f"Predicted sales: {predict_sales(a.tv, a.radio, a.newspaper):.2f}")

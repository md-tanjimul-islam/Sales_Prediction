"""Prediction helpers (CLI + importable)."""
import argparse
import json
from functools import lru_cache

import joblib
import pandas as pd
from src.features import METRICS_PATH, MODEL_PATH, add_features, load_data

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


CHANNELS = ["TV", "Radio", "Newspaper"]


@lru_cache(maxsize=1)
def dataset_stats() -> dict:
    """Ranges and averages from the training data (used for context and warnings)."""
    df = load_data()
    return {"n": int(len(df)), "mean_sales": float(df["Sales"].mean()),
            "ranges": {c: [float(df[c].min()), float(df[c].max())] for c in CHANNELS},
            "means": {c: float(df[c].mean()) for c in CHANNELS}}


def _pred(vals: dict) -> float:
    return predict_sales(vals["TV"], vals["Radio"], vals["Newspaper"])


def explain(tv: float, radio: float, newspaper: float) -> dict:
    """Prediction plus everything needed to understand it."""
    vals = {"TV": tv, "Radio": radio, "Newspaper": newspaper}
    st, m = dataset_stats(), load_metrics()
    base = _pred(vals)
    rmse = m.get("results", {}).get(m.get("best_model", ""), {}).get("test_rmse", 0.4)
    margin = 1.96 * rmse  # approximate 95% range from hold-out error

    removal, marginal = {}, {}
    for c in CHANNELS:
        off = dict(vals); off[c] = 0.0
        up = dict(vals); up[c] += 10.0
        removal[c] = round(base - _pred(off), 2)      # sales lost if this channel were switched off
        marginal[c] = round(_pred(up) - base, 3)       # sales gained from +10k on this channel
    best = max(marginal, key=lambda c: marginal[c])
    total = tv + radio + newspaper

    warnings = [f"{c} spend ({vals[c]:g}) is outside the range seen in training data "
                f"({st['ranges'][c][0]:g}-{st['ranges'][c][1]:g}); treat the result with caution."
                for c in CHANNELS if not st["ranges"][c][0] * 0.9 <= vals[c] <= st["ranges"][c][1] * 1.1]

    diff = (base / st["mean_sales"] - 1) * 100
    insights = [f"Predicted sales are {abs(diff):.0f}% {'above' if diff >= 0 else 'below'} "
                f"the dataset average of {st['mean_sales']:.1f}k units."]
    insights.append(f"The best place for an extra $10k right now is {best}: about "
                    f"+{marginal[best]:.2f}k units of sales.")
    if marginal["Newspaper"] < 0.05 and newspaper > 0:
        insights.append("Newspaper spend contributes almost nothing; moving it to Radio or TV "
                        "should raise sales at no extra cost.")
    if marginal["TV"] < marginal["Radio"] and tv > st["means"]["TV"]:
        insights.append("TV is past its average spend and returns are diminishing; Radio now "
                        "gives more sales per extra dollar.")
    if radio < st["means"]["Radio"] * 0.5 and tv > 0:
        insights.append("Radio spend is low. TV and Radio reinforce each other, so adding some "
                        "Radio lifts the effect of your TV budget.")

    curves = {}
    for c in ["TV", "Radio"]:
        lo, hi = st["ranges"][c]
        pts = []
        for i in range(31):
            x = lo + (hi - lo) * i / 30
            pts.append([round(x, 2), round(_pred({**vals, c: x}), 3)])
        curves[c] = pts

    return {"predicted_sales": round(base, 2), "unit": "thousand units",
            "range_low": round(max(base - margin, 0), 2), "range_high": round(base + margin, 2),
            "total_spend": round(total, 2),
            "sales_per_1k_spend": round(base / total, 3) if total > 0 else None,
            "vs_average_pct": round(diff, 1), "dataset_mean_sales": round(st["mean_sales"], 2),
            "channel_removal_effect": removal, "gain_per_10k": marginal, "best_channel": best,
            "insights": insights, "warnings": warnings, "curves": curves,
            "inputs": vals}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Predict sales from ad spend")
    p.add_argument("--tv", type=float, required=True)
    p.add_argument("--radio", type=float, required=True)
    p.add_argument("--newspaper", type=float, required=True)
    a = p.parse_args()
    print(f"Predicted sales: {predict_sales(a.tv, a.radio, a.newspaper):.2f}")
"""Flask web app + JSON API for sales prediction."""
import os
from flask import Flask, jsonify, render_template, request
from src.predict import dataset_stats, explain, load_metrics

app = Flask(__name__)


def _parse(payload):
    try:
        return [float(payload[k]) for k in ("tv", "radio", "newspaper")]
    except (KeyError, TypeError, ValueError):
        raise ValueError("Provide numeric 'tv', 'radio' and 'newspaper' values")


@app.get("/")
def index():
    return render_template("index.html", metrics=load_metrics(), stats=dataset_stats())


@app.get("/health")
def health():
    return jsonify(status="ok")


@app.get("/api/meta")
def meta():
    return jsonify(stats=dataset_stats(), metrics=load_metrics())


@app.post("/api/predict")
def api_predict():
    try:
        tv, radio, news = _parse(request.get_json(silent=True) or request.form)
        if min(tv, radio, news) < 0 or max(tv, radio, news) > 1e5:
            raise ValueError("Spend values must be between 0 and 100000")
        return jsonify(explain(tv, radio, news))
    except ValueError as e:
        return jsonify(error=str(e)), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
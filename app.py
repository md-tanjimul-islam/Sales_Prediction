"""Flask web app + JSON API for sales prediction."""
import os
from flask import Flask, jsonify, render_template, request
from src.predict import load_metrics, predict_sales

app = Flask(__name__)


def _parse(payload):
    try:
        return [float(payload[k]) for k in ("tv", "radio", "newspaper")]
    except (KeyError, TypeError, ValueError):
        raise ValueError("Provide numeric 'tv', 'radio' and 'newspaper' values")


@app.get("/")
def index():
    return render_template("index.html", metrics=load_metrics())


@app.get("/health")
def health():
    return jsonify(status="ok")


@app.post("/api/predict")
def api_predict():
    try:
        tv, radio, news = _parse(request.get_json(silent=True) or request.form)
        return jsonify(predicted_sales=round(predict_sales(tv, radio, news), 2),
                       unit="thousand units")
    except ValueError as e:
        return jsonify(error=str(e)), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

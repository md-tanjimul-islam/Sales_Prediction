# CodeAlpha Sales Prediction

Predict product sales from advertising spend (TV, Radio, Newspaper) with machine learning.
Built for the **CodeAlpha Data Science Internship — Task 4: Sales Prediction using Python**.

The project covers the full workflow: data cleaning, exploratory analysis, feature engineering,
model comparison with cross-validation, business insights, and deployment as a web app and JSON API.

## Live demo

**[https://codealpha-sales-prediction.onrender.com](https://sales-prediction-g2aj.onrender.com/)**

(Hosted on Render's free plan, so the first load after inactivity can take about 50 seconds. Health check: `/health`.)

## Features

- **Data cleaning** – drops the row-ID column, strips names, coerces to numeric, removes duplicates, missing and impossible (negative) values (`src/features.py`)
- **Data audit** – skewness, kurtosis, IQR outliers, normality tests, VIF; saved to `reports/data_audit.txt`
- **EDA plots** – distributions and boxplots, correlation heatmap, spend-vs-sales curves, TV×Radio interaction view
- **Feature engineering** – `TV×Radio` interaction, `TV²`, `Radio²` and `log(TV)` (diminishing returns on TV spend)
- **Seven models compared** – Linear (raw and engineered), Ridge, Lasso, ElasticNet, tuned Random Forest, tuned Gradient Boosting
- **Reliable evaluation** – repeated 5×5 cross-validation, a held-out 20% test set, nested CV for tuned models; the model is chosen by CV, never by test score
- **Diagnostics** – actual vs predicted, residual histogram and Q-Q plot, model comparison chart
- **Advertising impact analysis** – estimated sales change per extra unit of spend on each channel
- **Interactive web dashboard** – sliders and presets with live prediction, responsive layout, automatic dark mode
- **Explained predictions** – every forecast includes a 95% likely range, comparison with the dataset average, channel contribution, what-if for +$10k per channel, plain-language insights, an interactive response curve, and warnings when inputs fall outside the training range
- **REST API** (`POST /api/predict`, `GET /api/meta`, `GET /health`) and **CLI** (`python -m src.predict --tv 150 --radio 25 --newspaper 30`)
- **Tests** (pytest) and **Render blueprint** (`render.yaml`)

## Results

| Model | CV R² | Test R² | Test RMSE |
|---|---|---|---|
| Linear (raw features) | 0.882 | 0.899 | 1.782 |
| Linear (engineered) | 0.996 | 0.996 | 0.373 |
| Ridge (engineered) | 0.996 | 0.996 | 0.373 |
| Lasso (engineered) | 0.996 | 0.996 | 0.369 |
| **ElasticNet (engineered)** (selected) | **0.996** | **0.996** | **0.368** |
| Random Forest (tuned) | 0.984 | 0.991 | 0.523 |
| Gradient Boosting (tuned) | 0.986 | 0.991 | 0.520 |

Feature engineering, not a fancier algorithm, produced the gain: error fell from RMSE 1.78 to 0.37. Sales saturate as TV spend grows, which `log(TV)` captures. The four linear models are effectively tied, and the simple model beats the tree ensembles on this small dataset (200 rows). The final model is refit on all 200 rows, with repeated-CV R² of 0.996.

### Business insights

- **TV** correlates most with sales (0.78), then **Radio** (0.58); **Newspaper** is weak (0.23).
- Near average budgets, +1 unit (thousand $) of spend adds about **0.19** (Radio), **0.037** (TV) and **0.001** (Newspaper) thousand units of sales. Radio gives the most per dollar, and Newspaper adds almost nothing.
- TV and Radio reinforce each other, and TV returns diminish at high spend, so a balanced split beats heavy TV alone.
- Recommendation: move budget away from Newspaper toward Radio, and avoid pushing TV into the saturated range.

*Caveat: the data is observational (200 markets), so impact numbers are estimates, not proven causal effects. Predictions outside the training range (TV 0.7–296, Radio 0–50, Newspaper 0.3–114) are unreliable.*

## Project structure

```
CodeAlpha_Sales_Prediction/
├── app.py                 # Flask app + API
├── data/Advertising.csv   # dataset
├── src/
│   ├── features.py        # loading + feature engineering
│   ├── train.py           # EDA, training, evaluation, saving
│   └── predict.py         # prediction helper + CLI
├── templates/index.html   # interactive dashboard
├── tests/test_app.py      # pytest suite
├── models/                # saved model + metrics.json
├── reports/               # generated plots + data_audit.txt
├── render.yaml            # Render deployment blueprint
└── requirements.txt
```

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m src.train              # trains model, creates plots
pytest                           # run tests
python app.py                    # open http://localhost:5000
```

## API

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"tv": 150, "radio": 25, "newspaper": 30}'
```

Response (curves omitted):

```json
{
  "predicted_sales": 15.02, "unit": "thousand units",
  "range_low": 14.29, "range_high": 15.74,
  "total_spend": 205.0, "sales_per_1k_spend": 0.073,
  "vs_average_pct": 7.1, "dataset_mean_sales": 14.02,
  "channel_removal_effect": {"TV": 14.27, "Radio": 4.87, "Newspaper": 0.01},
  "gain_per_10k": {"TV": 0.385, "Radio": 2.004, "Newspaper": 0.004},
  "best_channel": "Radio",
  "insights": ["..."], "warnings": []
}
```

| Field | Meaning |
|---|---|
| `range_low` / `range_high` | Approximate 95% range, from hold-out error (±1.96 × RMSE) |
| `channel_removal_effect` | Sales lost if that channel were switched off |
| `gain_per_10k` | Extra sales from adding $10k to that channel |
| `warnings` | Inputs outside the training data range |

Invalid input returns HTTP 400 with an error message. `GET /api/meta` returns dataset statistics and model metrics.

## Dataset

200 markets; columns `TV`, `Radio`, `Newspaper` (ad spend, thousand $) and `Sales` (thousand units). No missing values.

## Tech stack

Python, pandas, NumPy, scikit-learn, Matplotlib, Seaborn, Flask, Gunicorn, pytest.

## 👤 Author

**Md. Tanjimul Islam** — Data Science Intern at CodeAlpha
GitHub: [md-tanjimul-islam](https://github.com/md-tanjimul-islam)

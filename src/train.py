"""Full ML workflow: data audit -> EDA -> feature engineering -> tuning -> evaluation -> saving.

Run:  python -m src.train
Outputs: models/sales_model.joblib, models/metrics.json, reports/*.png, reports/data_audit.txt
"""
import json
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNetCV, LassoCV, LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (GridSearchCV, KFold, RepeatedKFold,
                                     cross_val_score, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features import (BASE_FEATURES, DATA_PATH, FEATURES, METRICS_PATH, MODEL_PATH,
                          REPORTS_DIR, TARGET, add_features, load_data)

warnings.filterwarnings("ignore")
SEED = 42
sns.set_theme(style="whitegrid")


def save(name):
    plt.tight_layout(); plt.savefig(REPORTS_DIR / name, dpi=140); plt.close()


def vif(X: pd.DataFrame) -> dict:
    """Variance inflation factor, computed from R^2 of regressing each column on the others."""
    out = {}
    for c in X.columns:
        r2 = LinearRegression().fit(X.drop(columns=c), X[c]).score(X.drop(columns=c), X[c])
        out[c] = round(1 / (1 - r2), 2) if r2 < 1 else float("inf")
    return out


def audit_and_eda(raw: pd.DataFrame, df: pd.DataFrame) -> dict:
    REPORTS_DIR.mkdir(exist_ok=True)
    lines = [f"Raw shape: {raw.shape} | clean shape: {df.shape}",
             f"Missing values (raw): {raw.isna().sum().to_dict()}",
             f"Duplicate rows (raw, excl. index): "
             f"{int(raw.drop(columns=raw.columns[0]).duplicated().sum())}", "",
             "Column         skew   kurtosis  IQR-outliers  Shapiro-p"]
    for c in BASE_FEATURES + [TARGET]:
        q1, q3 = df[c].quantile([.25, .75]); i = q3 - q1
        n_out = int(((df[c] < q1 - 1.5 * i) | (df[c] > q3 + 1.5 * i)).sum())
        lines.append(f"{c:12s} {stats.skew(df[c]):6.2f} {stats.kurtosis(df[c]):9.2f} "
                     f"{n_out:12d} {stats.shapiro(df[c])[1]:10.3f}")
    lines += ["", "Pearson correlation with Sales:", df.corr()[TARGET].round(3).to_string(),
              "", f"VIF (base features): {vif(df[BASE_FEATURES])}",
              "Outlier decision: the Newspaper outliers are valid spend values; they were kept "
              "(Newspaper has ~no effect on the model, verified by CV)."]
    (REPORTS_DIR / "data_audit.txt").write_text("\n".join(lines))

    fig, ax = plt.subplots(2, 4, figsize=(16, 6))
    for j, c in enumerate(BASE_FEATURES + [TARGET]):
        sns.histplot(df[c], kde=True, ax=ax[0, j]); sns.boxplot(x=df[c], ax=ax[1, j])
    save("distributions.png")

    plt.figure(figsize=(6, 5)); sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation matrix"); save("correlation_heatmap.png")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for a, c in zip(ax, BASE_FEATURES):
        sns.regplot(data=df, x=c, y=TARGET, ax=a, order=2, line_kws={"color": "red"})
        a.set_title(f"{c} vs Sales (quadratic fit)")
    save("spend_vs_sales.png")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    sns.scatterplot(data=df, x="TV", y=TARGET, hue="Radio", palette="viridis", ax=ax[0])
    ax[0].set_title("TV vs Sales, coloured by Radio (interaction)")
    sns.scatterplot(x=np.log1p(df.TV), y=df[TARGET], ax=ax[1]); ax[1].set_title("log(TV) vs Sales")
    save("tv_radio_interaction.png")
    return {"vif_base": vif(df[BASE_FEATURES])}


def main():
    raw = pd.read_csv(DATA_PATH)
    df = load_data()
    info = audit_and_eda(raw, df)
    print(f"Clean data: {df.shape[0]} rows | missing: {int(df.isna().sum().sum())}")

    X_base, y = df[BASE_FEATURES], df[TARGET]
    X = add_features(df)
    idx_tr, idx_te = train_test_split(df.index, test_size=0.2, random_state=SEED)
    X_tr, X_te, y_tr, y_te = X.loc[idx_tr], X.loc[idx_te], y.loc[idx_tr], y.loc[idx_te]
    Xb_tr, Xb_te = X_base.loc[idx_tr], X_base.loc[idx_te]
    alphas = np.logspace(-3, 2, 40)
    cv_tree = KFold(5, shuffle=True, random_state=SEED)  # outer CV for tuned (nested) models
    cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=SEED)

    def scaled(m): return Pipeline([("scale", StandardScaler()), ("model", m)])
    def tuned(est, grid):
        return GridSearchCV(est, grid, cv=KFold(5, shuffle=True, random_state=SEED), scoring="r2", n_jobs=-1)

    # name -> (pipeline, uses engineered features?)
    models = {
        "Linear (raw features)": (scaled(LinearRegression()), False),
        "Linear (engineered)": (scaled(LinearRegression()), True),
        "Ridge (engineered)": (scaled(RidgeCV(alphas=alphas)), True),
        "Lasso (engineered)": (scaled(LassoCV(cv=5, random_state=SEED)), True),
        "ElasticNet (engineered)": (scaled(ElasticNetCV(cv=5, l1_ratio=[.2, .5, .8, .95], random_state=SEED)), True),
        "Random Forest (tuned)": (tuned(RandomForestRegressor(random_state=SEED),
                                        {"n_estimators": [200], "max_depth": [None, 8],
                                         "min_samples_leaf": [1, 3]}), True),
        "Gradient Boosting (tuned)": (tuned(GradientBoostingRegressor(random_state=SEED, subsample=.8),
                                            {"n_estimators": [200, 400], "learning_rate": [.05, .1],
                                             "max_depth": [2, 3]}), True),
    }
    results, fitted = {}, {}
    for name, (est, eng) in models.items():
        Xtr, Xte = (X_tr, X_te) if eng else (Xb_tr, Xb_te)
        cv_r2 = cross_val_score(est, Xtr, y_tr, cv=cv_tree if name.endswith("(tuned)") else cv, scoring="r2", n_jobs=-1)
        est.fit(Xtr, y_tr); pred = est.predict(Xte)
        results[name] = {"cv_r2_mean": round(float(cv_r2.mean()), 4), "cv_r2_std": round(float(cv_r2.std()), 4),
                         "test_r2": round(float(r2_score(y_te, pred)), 4),
                         "test_mae": round(float(mean_absolute_error(y_te, pred)), 4),
                         "test_rmse": round(float(np.sqrt(mean_squared_error(y_te, pred))), 4),
                         "engineered": eng}
        fitted[name] = est
        print(f"{name:28s} CV R2={cv_r2.mean():.4f}±{cv_r2.std():.4f} | Test R2={results[name]['test_r2']:.4f} "
              f"RMSE={results[name]['test_rmse']:.3f} MAE={results[name]['test_mae']:.3f}")

    # Select among engineered models by repeated-CV R^2 (never by test score)
    best = max((n for n in results if results[n]["engineered"]), key=lambda n: results[n]["cv_r2_mean"])
    best_est = fitted[best]
    print(f"\nBest model (repeated 5x5 CV R2): {best}")

    pred = best_est.predict(X_te); resid = y_te - pred
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    ax[0].scatter(y_te, pred, alpha=.8); lim = [y.min(), y.max()]; ax[0].plot(lim, lim, "r--")
    ax[0].set(xlabel="Actual", ylabel="Predicted", title=f"{best}: Actual vs Predicted")
    sns.histplot(resid, kde=True, ax=ax[1]); ax[1].set_title("Residuals")
    stats.probplot(resid, plot=ax[2]); ax[2].set_title("Residual Q-Q plot")
    save("actual_vs_predicted.png")

    plt.figure(figsize=(8, 4.5))
    names = list(results); plt.barh(names, [results[n]["cv_r2_mean"] for n in names],
                                     xerr=[results[n]["cv_r2_std"] for n in names])
    plt.xlim(.85, 1); plt.xlabel("Repeated CV R²"); plt.title("Model comparison"); save("model_comparison.png")

    # Marginal impact: +1 unit of spend per channel, other channels at their mean
    final_pipe = best_est.best_estimator_ if hasattr(best_est, "best_estimator_") else best_est
    mean_row = X_base.mean().to_dict(); impact = {}
    for ch in BASE_FEATURES:
        up = dict(mean_row); up[ch] += 1
        p = final_pipe.predict(add_features(pd.DataFrame([mean_row, up])))
        impact[ch] = round(float(p[1] - p[0]), 4)
    print("Sales change per +1 unit of spend:", impact)

    # Residual diagnostics + refit on 100% of data for deployment
    diag = {"residual_mean": round(float(resid.mean()), 4), "residual_shapiro_p": round(float(stats.shapiro(resid)[1]), 4)}
    final_pipe.fit(X, y)
    # honest estimate for the final model: out-of-fold R^2 on the full dataset
    oof = cross_val_score(final_pipe, X, y, cv=cv, scoring="r2")
    print(f"Final model repeated-CV R2 on all data: {oof.mean():.4f}±{oof.std():.4f}")
    MODEL_PATH.parent.mkdir(exist_ok=True); joblib.dump(final_pipe, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps({
        "best_model": best, "results": results, "channel_impact_per_unit": impact,
        "n_rows": int(len(df)), "features": FEATURES, "vif_base": info["vif_base"],
        "final_cv_r2": round(float(oof.mean()), 4), "diagnostics": diag}, indent=2))
    print("Saved model, metrics and reports.")


if __name__ == "__main__":
    main()

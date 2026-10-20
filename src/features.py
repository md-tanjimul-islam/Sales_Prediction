"""Shared data loading and feature engineering (used by training and the web app)."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "Advertising.csv"
MODEL_PATH = ROOT / "models" / "sales_model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"
REPORTS_DIR = ROOT / "reports"

BASE_FEATURES = ["TV", "Radio", "Newspaper"]
FEATURES = BASE_FEATURES + ["TV_Radio", "TV_sq", "Radio_sq", "TV_log"]
TARGET = "Sales"


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load and clean the dataset.

    - drops the row-index column (an ID with no predictive meaning)
    - strips column names, coerces to numeric, removes duplicates and rows with missing values
    - removes impossible values (negative spend or sales)
    """
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed") or c.strip() == ""])
    df.columns = [c.strip() for c in df.columns]
    df = df.apply(pd.to_numeric, errors="coerce").dropna().drop_duplicates()
    df = df[(df[BASE_FEATURES + [TARGET]] >= 0).all(axis=1)]
    return df.reset_index(drop=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering.

    TV_Radio  : interaction - TV and Radio reinforce each other
    TV_sq     : curvature of the TV response
    Radio_sq  : curvature of the Radio response
    TV_log    : diminishing returns on TV spend (the single biggest accuracy gain)
    """
    out = df[BASE_FEATURES].astype(float).copy()
    out["TV_Radio"] = out["TV"] * out["Radio"]
    out["TV_sq"] = out["TV"] ** 2
    out["Radio_sq"] = out["Radio"] ** 2
    out["TV_log"] = np.log1p(out["TV"])
    return out[FEATURES]

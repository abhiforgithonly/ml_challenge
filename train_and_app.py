import os
import io

import joblib
import numpy as np
import pandas as pd

from django.http import HttpResponse
from django.urls import path
from django.shortcuts import render

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------
# 1. PREPROCESSING
# ----------------------------

NUMERIC = [
    "Clicks",
    "Impressions",
    "Cost",
    "Leads",
    "Conversions",
    "ConvRate_calc",
    "Ad_year",
    "Ad_month",
    "Ad_dayofweek",
]

CATEGORICAL = [
    "Location_clean",
    "Device_clean",
    "Campaign_clean",
    "Keyword_clean",
]

# NOTE: no "Conversion Rate" here – we recompute ConvRate_calc ourselves
REQUIRED_COLS = [
    "Ad_ID",
    "Campaign_Name",
    "Clicks",
    "Impressions",
    "Cost",
    "Leads",
    "Conversions",
    "Ad_Date",
    "Location",
    "Device",
    "Keyword",
]


def clean_money(s: pd.Series) -> pd.Series:
    """Remove currency symbols/commas and cast to float."""
    return (
        s.astype(str)
        .str.replace(r"[\$,]", "", regex=True)
        .replace("", np.nan)
        .astype(float)
    )


def fe(df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
    """Basic feature engineering/cleaning."""
    df = df.copy()

    # money
    df["Cost"] = clean_money(df["Cost"])
    if is_train and "Sale_Amount" in df.columns:
        df["Sale_Amount"] = clean_money(df["Sale_Amount"])

    # categorical clean-up
    df["Location_clean"] = df["Location"].astype(str).str.lower().str.strip()
    df["Device_clean"] = df["Device"].astype(str).str.lower().str.strip()
    df["Campaign_clean"] = df["Campaign_Name"].astype(str).str.lower().str.strip()
    df["Keyword_clean"] = df["Keyword"].astype(str).str.lower().str.strip()

    # dates (let pandas infer formats)
    df["Ad_Date_parsed"] = pd.to_datetime(
        df["Ad_Date"],
        errors="coerce",
    )
    df["Ad_year"] = df["Ad_Date_parsed"].dt.year
    df["Ad_month"] = df["Ad_Date_parsed"].dt.month
    df["Ad_dayofweek"] = df["Ad_Date_parsed"].dt.dayofweek

    # recompute conversion rate (ignore original "Conversion Rate" column)
    clicks = df["Clicks"].replace({0: np.nan})
    df["ConvRate_calc"] = df["Conversions"] / clicks

    return df


def X_y(df: pd.DataFrame):
    """Training features + target."""
    df_fe = fe(df, is_train=True)
    df_fe = df_fe.dropna(subset=["Sale_Amount"])
    X = df_fe[NUMERIC + CATEGORICAL]
    y = df_fe["Sale_Amount"]
    return X, y


def X_inf(df: pd.DataFrame):
    """Inference features (no target)."""
    df_fe = fe(df, is_train=False)
    X = df_fe[NUMERIC + CATEGORICAL]
    return X


# ----------------------------
# 2. ANALYSIS PLOTS
# ----------------------------

def save_analysis_plots(X_val: pd.DataFrame, y_val: pd.Series, model):
    """
    Save basic visualizations into static/:
      - target_distribution.png
      - feature_importance.png
    """
    os.makedirs("static", exist_ok=True)

    # Target distribution
    plt.figure()
    y_val.plot.hist(bins=20)
    plt.title("Sale_Amount distribution (validation)")
    plt.xlabel("Sale_Amount")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join("static", "target_distribution.png"))
    plt.close()

    # Permutation importance (top 10 features)
    result = permutation_importance(
        model,
        X_val,
        y_val,
        n_repeats=10,
        random_state=42,
        n_jobs=-1,
    )
    importances_mean = result.importances_mean
    indices = np.argsort(importances_mean)[::-1][:10]
    names = X_val.columns[indices]
    values = importances_mean[indices]

    plt.figure()
    plt.barh(range(len(indices)), values)
    plt.yticks(range(len(indices)), names)
    plt.title("Permutation importance (top 10 features)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join("static", "feature_importance.png"))
    plt.close()


# ----------------------------
# 3. TRAINING + EVALUATION
# ----------------------------

def train_model():
    """
    Train model, evaluate, save to model.joblib, and generate analysis plots.

    - Uses an 80/20 train/validation split to report MAE (model vs baseline).
    - Generates basic visualizations (target & feature importance) from validation data.
    - Then refits on the full dataset and reports training R².
    """
    if not os.path.exists("train.csv"):
        raise FileNotFoundError("train.csv not found in project root.")

    df = pd.read_csv("train.csv")
    X, y = X_y(df)

    # 80/20 split for validation MAE
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    numeric_pipe = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC),
            ("cat", categorical_pipe, CATEGORICAL),
        ]
    )

    base_model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )

    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", base_model),
        ]
    )

    # --- fit on train and evaluate on validation (MAE only) ---
    pipe.fit(X_train, y_train)
    y_pred_val = pipe.predict(X_val)

    mae_val = mean_absolute_error(y_val, y_pred_val)

    # naive baseline: always predict mean of training target
    baseline_val = np.full_like(y_val, y_train.mean(), dtype=float)
    mae_baseline = mean_absolute_error(y_val, baseline_val)

    print(f"Validation MAE (model):    {mae_val:.2f}")
    print(f"Validation MAE (baseline): {mae_baseline:.2f}")

    # --- generate analysis plots from validation data ---
    save_analysis_plots(X_val, y_val, pipe)

    # --- refit on full data and report training R² ---
    pipe.fit(X, y)
    y_pred_train = pipe.predict(X)
    r2_train = r2_score(y, y_pred_train)
    print(f"Training R² (model):       {r2_train:.3f}")

    joblib.dump(pipe, "model.joblib")
    print("Model saved -> model.joblib")


# ----------------------------
# 4. DJANGO VIEW + URLS
# ----------------------------

model = None
if __name__ != "__main__":
    if os.path.exists("model.joblib"):
        model = joblib.load("model.joblib")
    else:
        print("model.joblib not found, training model now...")
        train_model()
        model = joblib.load("model.joblib")


def predict_view(request):
    context = {}
    if request.method == "GET":
        return render(request, "predict.html", context)

    f = request.FILES.get("file")
    if not f:
        context["error"] = "Please upload a CSV file."
        return render(request, "predict.html", context)

    try:
        df = pd.read_csv(f)
    except Exception as e:
        context["error"] = f"Could not read CSV: {e}"
        return render(request, "predict.html", context)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        context["error"] = "Missing required columns: " + ", ".join(missing)
        return render(request, "predict.html", context)

    # Make predictions
    X = X_inf(df)
    preds = model.predict(X)

    df_out = df.copy()
    df_out["Predicted_Sale_Amount"] = preds.round(2)

    # Ensure static/ exists
    os.makedirs("static", exist_ok=True)

    # 1) Save predictions CSV so the UI can link to it
    predictions_csv_path = os.path.join("static", "predictions.csv")
    df_out.to_csv(predictions_csv_path, index=False)

    # 2) Save predictions distribution plot
    plt.figure()
    pd.Series(preds).plot.hist(bins=20)
    plt.title("Predicted Sale_Amount distribution")
    plt.xlabel("Predicted_Sale_Amount")
    plt.ylabel("Count")
    plt.tight_layout()
    predictions_plot_path = os.path.join("static", "predictions_distribution.png")
    plt.savefig(predictions_plot_path)
    plt.close()

    # Indicate to the template that results (and download) are ready
    context["results_ready"] = True

    return render(request, "predict.html", context)


urlpatterns = [
    path("", predict_view),
]



if __name__ == "__main__":
    train_model()

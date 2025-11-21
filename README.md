# AdAstraa AI – 24h ML + Django Challenge – Abhijeet Suryavanshi

This project is a minimal end-to-end solution for the **Machine Learning Engineer (Full Stack)** challenge at AdAstraa AI:

- Clean messy eCommerce ad campaign data  
- Train a regression model to predict `Sale_Amount`  
- Host the trained model inside a Django backend  
- Accept a CSV upload (without `Sale_Amount`)  
- Return a `predictions.csv` with `Predicted_Sale_Amount` appended  
- Show basic visualizations (target distribution, feature importance, prediction distribution) in the UI

The focus is on **data preprocessing quality**, a **robust ML pipeline**, and a **small but polished Django app**.

---

## 1. Project Structure

Intentionally kept very small and focused:

```text
.
├─ manage.py              # Django entrypoint
├─ settings.py            # Minimal Django settings (no DB)
├─ train_and_app.py       # Data cleaning, model training, Django view & URLs
├─ train.csv              # Provided training data (input)
├─ model.joblib           # Trained model pipeline (generated)
├─ templates/
│  └─ predict.html        # UI for upload, download & visualizations
└─ static/
   ├─ target_distribution.png          # Sale_Amount distribution (validation)
   ├─ feature_importance.png           # Permutation-based feature importance
   ├─ predictions_distribution.png     # Predicted_Sale_Amount distribution (latest run)
   └─ predictions.csv                  # Latest predictions (downloaded via UI)
```

**Key idea:** one main Python file (`train_and_app.py`) contains:
- Preprocessing / feature engineering
- Model training + evaluation + analysis plots
- The prediction view and URL configuration for Django

---

## 2. Setup & How to Run

### 2.1 Install Dependencies

From the project root:

```bash
pip install -r requirements.txt
```

If you don't use `requirements.txt`, the core dependencies are:

```bash
pip install django pandas numpy scikit-learn joblib matplotlib
```

### 2.2 Train the Model (and generate analysis plots)

Make sure `train.csv` is in the project root, then:

```bash
python train_and_app.py
```

This will:
- Clean the data
- Train the model
- Print validation + training metrics
- Save the trained pipeline to `model.joblib`
- Generate two analysis plots in `static/`:
  - `target_distribution.png` – histogram of `Sale_Amount` (validation set)
  - `feature_importance.png` – permutation importance (top 10 features)

### 2.3 Run the Django App

```bash
python manage.py runserver
```

Open in your browser:

```
http://127.0.0.1:8000/
```

Upload a CSV that:
- Has the same columns as `train.csv` except `Sale_Amount`
- Example expected columns:
  ```
  Ad_ID, Campaign_Name, Clicks, Impressions, Cost, Leads,
  Conversions, Ad_Date, Location, Device, Keyword
  ```

After upload:
- The app generates `static/predictions.csv` with a new `Predicted_Sale_Amount` column
- It also creates `static/predictions_distribution.png` (prediction distribution plot)
- The UI shows:
  - A "Download latest predictions.csv" button
  - All three visualizations (target, feature importance, prediction distribution)

---

## 3. Data Cleaning & Feature Engineering

The dataset simulates messy, real-world ad data: mixed date formats, inconsistent money formats, typos, casing issues, and incorrect/missing Conversion Rate.

All preprocessing is implemented in `train_and_app.py`:
- `fe(df, is_train)`
- `X_y(df)`
- `X_inf(df)`

### 3.1 Handled Data Issues

**Money fields** (`Cost`, `Sale_Amount`):
- Remove `$` and commas with a regex
- Cast to float

```python
df["Cost"] = clean_money(df["Cost"])
if is_train and "Sale_Amount" in df.columns:
    df["Sale_Amount"] = clean_money(df["Sale_Amount"])
```

**Categorical text noise:**
- Normalized to lowercase and stripped whitespace to handle casing/spacing differences:

```python
df["Location_clean"] = df["Location"].astype(str).str.lower().str.strip()
df["Device_clean"] = df["Device"].astype(str).str.lower().str.strip()
df["Campaign_clean"] = df["Campaign_Name"].astype(str).str.lower().str.strip()
df["Keyword_clean"] = df["Keyword"].astype(str).str.lower().str.strip()
```

**Dates** (`Ad_Date`):
- Mixed formats parsed with `pandas.to_datetime` and invalid dates coerced to `NaT`:

```python
df["Ad_Date_parsed"] = pd.to_datetime(
    df["Ad_Date"],
    errors="coerce",
)
```

- Derived calendar features:

```python
df["Ad_year"] = df["Ad_Date_parsed"].dt.year
df["Ad_month"] = df["Ad_Date_parsed"].dt.month
df["Ad_dayofweek"] = df["Ad_Date_parsed"].dt.dayofweek
```

These capture temporal patterns like seasonality and weekday effects.

**Conversion rate:**
- The original `Conversion Rate` column may be wrong or missing.
- I do not rely on it and instead recompute a clean version:

```python
clicks = df["Clicks"].replace({0: np.nan})
df["ConvRate_calc"] = df["Conversions"] / clicks
```

This ensures consistency between training and inference and avoids depending on noisy input.

### 3.2 Final Feature Set

The model uses the following features:

**Numeric:**
- `Clicks`
- `Impressions`
- `Cost`
- `Leads`
- `Conversions`
- `ConvRate_calc` (recomputed)
- `Ad_year`
- `Ad_month`
- `Ad_dayofweek`

**Categorical:**
- `Location_clean`
- `Device_clean`
- `Campaign_clean`
- `Keyword_clean`

`X_y(df)` is used when training; `X_inf(df)` is used in the Django view.  
Both call `fe(df, ...)`, so training and prediction stay in sync.

---

## 4. Modeling Approach & Evaluation

### 4.1 Model & Pipeline

**Model:** `RandomForestRegressor` inside a single `sklearn.Pipeline`.

**Preprocessing with `ColumnTransformer`:**
- **Numeric:**
  - `SimpleImputer(strategy="median")`
- **Categorical:**
  - `SimpleImputer(strategy="most_frequent")`
  - `OneHotEncoder(handle_unknown="ignore")`

**Estimator:**

```python
RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
)
```

This design is:
- Robust to mixed feature types (numeric + categorical)
- Handles non-linear relationships and interactions
- Safe with unseen categories in production (via `handle_unknown="ignore"`)

### 4.2 Evaluation Strategy

Implemented in `train_model()`:
- Load `train.csv`
- Build `X`, `y` via `X_y(df)`
- Split into 80% train / 20% validation
- Train the pipeline on the training split
- Compare against a baseline that always predicts the mean `Sale_Amount`
- Generate:
  - `target_distribution.png` from validation targets
  - `feature_importance.png` (permutation importance on validation)
- Refit the pipeline on the full dataset and report training R²
- Save model as `model.joblib`

**Actual metrics on the provided dataset:**
- Validation MAE (model): **250.31**
- Validation MAE (baseline, mean predictor): **248.29**
- Training R² (model): **0.856**

### 4.3 Interpretation

The target `Sale_Amount` lies in a relatively narrow band (~1000–2000), so:
- A naive model that always predicts the mean already performs decently.
- The Random Forest model fits the training data well (R² ≈ 0.86),
- But on a held-out validation set, it performs roughly on par with the baseline MAE.

This suggests the current feature set has limited predictive signal for `Sale_Amount`.  
In a real production scenario, the next step would be to improve:
- Feature richness (e.g., aggregations over time, product-level features)
- Data quality and label design

The code explicitly prints both model vs baseline MAE, and the README calls this out to show awareness of model performance vs a simple reference.

---

## 5. Django App Flow (Upload → Predict → Download + Visualize)

All Django logic lives in `train_and_app.py` and `templates/predict.html`.

### 5.1 Backend: `predict_view`

**GET /:**
- Renders `predict.html` with:
  - Upload form
  - Training-time visualizations:
    - `target_distribution.png`
    - `feature_importance.png`

**POST /:**
- Reads uploaded CSV into a DataFrame
- Validates presence of required columns:
  ```
  Ad_ID, Campaign_Name, Clicks, Impressions, Cost,
  Leads, Conversions, Ad_Date, Location, Device, Keyword
  ```
- Applies the same preprocessing as training (`X_inf(df)`)
- Uses `model.joblib` to predict `Sale_Amount`
- Creates a new DataFrame with:
  - All original columns
  - A new `Predicted_Sale_Amount` column (rounded to 2 decimals)
- Writes this to `static/predictions.csv`
- Generates `static/predictions_distribution.png` (histogram of predicted values)
- Renders `predict.html` with `results_ready=True`

### 5.2 Frontend: `predict.html`

The UI includes:
- A card-style layout with:
  - Title + subtitle
  - File upload field
  - Button: **Upload & Predict**
  - Clear hint about expected columns
- When errors occur (e.g., missing columns, bad CSV), a red error box appears.

After a successful prediction:
- A **"⬇ Download latest predictions.csv"** button appears
- A third plot appears in the "Model analysis" section:
  - `predictions_distribution.png`

The **Model analysis** section shows:
- `target_distribution.png` – `Sale_Amount` distribution on validation data
- `feature_importance.png` – permutation-based feature importance (top 10 features)
- `predictions_distribution.png` – distribution of `Predicted_Sale_Amount` for the most recent upload

This gives stakeholders a quick, visual sense of:
- How the target behaves
- What features matter most (for this model)
- How the latest predictions are distributed

---

## 6. Assumptions & Limitations

**Schema:**
- Upload files must contain the required columns listed above.
- `Sale_Amount` is not required in uploads (and usually absent).

**Conversion rate:**
- The original `Conversion Rate` column is ignored.
- A clean `ConvRate_calc` is recomputed from `Clicks` and `Conversions`.

**No database:**
- Django is used purely as a thin HTTP layer.
- No models, migrations, or authentication.

**Model performance:**
- The model is competitive with a mean baseline but doesn't outperform it by a large margin, due to limited signal in the provided features.
- The focus is on a clean pipeline + deployment rather than chasing tiny metric gains.

---

## 7. What I'd Improve with More Time

### 7.1 Deeper Modeling & Features

**Feature Engineering:**
- Aggregate campaign performance over multiple days (time windows).
- Engineer CTR, CPC, CPM, etc. with robust handling for low-volume campaigns.
- Normalize and cluster `Campaign_Name` / `Keyword` (e.g., text embeddings, fuzzy matching for typos).

**Modeling:**
- Try gradient boosting models (XGBoost, LightGBM, CatBoost).
- Use cross-validation and hyperparameter tuning (Random Search, Optuna).
- Calibrate predictions if needed.

**Explainability & Monitoring:**
- Add SHAP-based explanations per prediction.
- Track input distributions and prediction drift over time.
- Log model metrics and request stats for monitoring.

**Productionization:**
- Package as a Docker image.
- Add validation using a schema layer (e.g., Pydantic or DRF serializers).
- Introduce CI/CD for running tests and deploying updates safely.

### 7.2 Known Problems Encountered & How I Addressed Them

**Model not clearly better than baseline:**
- **Problem:** Validation MAE for the Random Forest (≈250.31) is very close to a naive mean predictor (≈248.29), and an earlier version showed a slightly negative validation R².
- **What I did now:**
  - Explicitly compare model MAE vs baseline MAE in `train_model()`.
  - Report positive training R² separately to show that the model can fit the training signal while being honest about generalization.
- **Future improvement:**
  - Focus on richer features (temporal aggregates, product/ad-level signals) and better label design rather than just swapping algorithms.

**Date parsing warnings and mixed formats:**
- **Problem:** Initial implementation using `dayfirst=True` / `infer_datetime_format` raised warnings due to mixed formats (e.g., `%Y-%m-%d` vs `DD-MM-YY`).
- **What I did now:**
  - Simplified to `pd.to_datetime(..., errors="coerce")` and derived `Ad_year`, `Ad_month`, `Ad_dayofweek` from that, which is robust and warning-free.
- **Future improvement:**
  - Add explicit format hints or separate parsing for known formats, plus validation reporting how many dates were invalid.

**Column name mismatch in uploads (Conversion Rate vs Conversion_Rate):**
- **Problem:** Some files used `Conversion_Rate` instead of `Conversion Rate`, causing "Missing required columns" errors even though that column isn't actually needed.
- **What I did now:**
  - Removed `Conversion Rate` from `REQUIRED_COLS` and fully rely on recomputed `ConvRate_calc = Conversions / Clicks`.
  - This makes the app tolerant to missing or misnamed conversion rate columns.
- **Future improvement:**
  - Make the schema validator more flexible (e.g., allow multiple aliases for certain columns) but still log what was inferred.

**Training running twice and cluttered logs:**
- **Problem:** In an earlier version, importing the module and running it directly both triggered `train_model()`, causing duplicate training runs and duplicate logs.
- **What I did now:**
  - Wrapped auto-training inside `if __name__ != "__main__":` and kept explicit training behind `if __name__ == "__main__": train_model()`.
- **Future improvement:**
  - Move training to a separate management command or script and configure logging properly instead of using `print` statements.

**Static prediction file being overwritten:**
- **Problem:** `static/predictions.csv` and `predictions_distribution.png` are overwritten on each new upload. This is fine for a single-user demo, but not ideal for multi-user / multi-session usage.
- **What I did now:**
  - Kept the behavior simple: the UI always reflects the latest run, which is enough for this challenge.
- **Future improvement:**
  - Use unique filenames per session/request (e.g., UUID or timestamp), or store outputs in object storage (e.g., S3) with links tied to user sessions.

---

## 8. Notes for Reviewers

The project demonstrates:
- Careful handling of messy, real-world-like data
- A single, shared preprocessing pipeline used for both training and inference
- A minimal Django app that:
  - Accepts CSV uploads
  - Produces predictions in CSV
  - Surfaces helpful visualizations (target distribution, feature importance, prediction distribution)

If helpful, I'd be happy to walk through the pipeline and design choices live.

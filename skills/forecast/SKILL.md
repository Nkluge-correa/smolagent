---
name: forecast
description: "End-to-end time-series sales forecasting pipeline: download, preprocess, train XGBoost, forecast, plot, and generate a report."
version: 0.0.0
author: Nkluge-correa
license: MIT
---

# Time-Series Sales Forecasting

Download a time-series dataset from the Hugging Face Hub, preprocess it with feature engineering, train an XGBoost regressor using time-series cross-validation, generate a 7-day sales forecast with blended seasonal adjustment, create visualisation plots, and produce a structured report with all artifacts.

## Quick Reference

| Step          | Tool                          | Default Input                              | Default Output                |
|---------------|-------------------------------|--------------------------------------------|-------------------------------|
| 1. Download   | `download_dataset_from_hub`   | Dataset path (HF Hub)                      | `data/raw_sales.csv`          |
| 2. Preprocess | `preprocess_time_series_data` | `data/raw_sales.csv`                       | `data/preprocessed.csv`       |
| 3. Train      | `train_xgboost_forecaster`    | `data/preprocessed.csv`                    | `model/forecaster.pkl`        |
| 4. Forecast   | `forecast_next_7_days`        | `model/forecaster.pkl`                     | `data/forecast.csv`           |
| 5. Plot       | `create_forecast_plot`        | `data/raw_sales.csv` + `data/forecast.csv` | `plots/forecast.png`          |
| 6. Report     | `generate_final_report`       | All artifacts                              | `report-YYYY-MM-DD/report.md` |

## Pipeline Overview

The forecasting pipeline follows six sequential stages. Each stage involves a tool that the agent can call independently. Intermediate artifacts (CSVs, model files, plots) are saved to disk, so you can inspect or reuse them between runs or restart from any stage.

```
Dataset (HF Hub)
     │
     ▼
┌──────────────────┐
│ 1. Download      │  download_dataset_from_hub()
│    raw_sales.csv │
└────────┬─────────┘
         ▼
┌──────────────────┐
│ 2. Preprocess    │  preprocess_time_series_data()
│    features,     │
│    scaler, meta  │
└────────┬─────────┘
         ▼
┌──────────────────┐
│ 3. Train XGBoost │  train_xgboost_forecaster()
│    forecaster.pkl│
└────────┬─────────┘
         ▼
┌──────────────────┐
│ 4. Forecast      │  forecast_next_7_days()
│    7 days ahead  │
└────────┬─────────┘
         ▼
┌──────────────────┐
│ 5. Create Plot   │  create_forecast_plot()
│    forecast.png  │
└────────┬─────────┘
         ▼
┌──────────────────┐
│ 6. Final Report  │  generate_final_report()
│    report.md     │
└──────────────────┘
```

## Dependencies

| Package           | Purpose                                                   |
|-------------------|-----------------------------------------------------------|
| `datasets`        | Download data from Hugging Face Hub                       |
| `pandas`, `numpy` | Data manipulation                                         |
| `scikit-learn`    | `StandardScaler`, `TimeSeriesSplit`, `mean_squared_error` |
| `xgboost`         | Gradient-boosted tree regressor                           |
| `matplotlib`      | Plotting                                                  |

**Notes:**

- You will probably be working in a virtual environment (e.g., `venv` or `conda`) that already has these packages installed. If not, you should create one and install the dependencies before running the pipeline.

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install datasets pandas numpy scikit-learn xgboost matplotlib
```

## 1. Download Dataset — `download_dataset_from_hub`

Fetches a dataset from the Hugging Face Hub using the `datasets` library and saves it as a local CSV. The output directory is created automatically.

**Parameters:**

| Parameter      | Type  | Default                | Description                                                     |
|----------------|-------|------------------------|-----------------------------------------------------------------|
| `dataset_path` | `str` | —                      | Hugging Face dataset path, e.g. `'AiresPucrs/time-series-data'` |
| `output_csv`   | `str` | `'data/raw_sales.csv'` | Where to save the CSV                                           |

**Code:**

```python
def download_dataset_from_hub(
    dataset_path: str,
    split: str = "train",
    output_csv: str = "data/raw_sales.csv",
) -> str:
    from datasets import load_dataset
    import os

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)

    try:
        dataset = load_dataset(dataset_path, split=split)
        df = dataset.to_pandas()
        df.to_csv(output_csv, index=False)
        return (
            f"Successfully downloaded '{dataset_path}' "
            f"({len(df)} rows) -> saved to '{output_csv}'."
        )
    except Exception as e:
        return f"Error downloading '{dataset_path}': {e}"
```

**Notes:**

- Uses the `datasets` library (install with `pip install datasets` if not already available).
- The `split="train"` argument loads the training split — adjust if your dataset uses a different split name.
- Supports any dataset on the Hugging Face Hub that can be loaded with `load_dataset`.

## 2. Preprocess Time-Series — `preprocess_time_series_data`

Transforms raw sales data into a feature-rich matrix ready for XGBoost. The pipeline performs the following steps in order:

1. **Outlier capping**: values above `mean + 3σ` are clamped to the cap.
2. **Lag features**: `difference_1` through `difference_7` (consecutive sales differences), `difference_month` (28-day difference), `difference_year` (366-day difference).
3. **Raw sales lags**: `sales_lag_1` through `sales_lag_7` (shifted raw sales).
4. **Rolling averages**: 7-day, 14-day, and 30-day moving means.
5. **Calendar features**: day-of-week, day-of-year, quarter, month, year (extracted from the `dates` column).
6. **Cyclical encoding**: sine/cosine transforms for day-of-week and month to preserve circular proximity (e.g., Sunday is close to Monday).
7. **One-hot encoding**: categorical calendar columns are expanded.
8. **Standard scaling**: all numeric features (differences, rolling means, lags, cyclical encodings) are scaled to zero mean and unit variance.

Rows with `NaN` (from lag shifts at the start of the series) are dropped.

**Persistent artifacts:**

| File                           | Contents                                                 |
|--------------------------------|----------------------------------------------------------|
| `data/preprocessed.csv`        | The full preprocessed DataFrame                          |
| `data/preprocessed_scaler.pkl` | Fitted `StandardScaler` (used by `forecast_next_7_days`) |
| `data/preprocessed_meta.json`  | Feature column lists (used by `forecast_next_7_days`)    |

**Parameters:**

| Parameter    | Type  | Default                   | Description                        |
|--------------|-------|---------------------------|------------------------------------|
| `csv_path`   | `str` | `'data/raw_sales.csv'`    | Path to the raw CSV                |
| `output_csv` | `str` | `'data/preprocessed.csv'` | Where to save the preprocessed CSV |

**Code:**

```python
def preprocess_time_series_data(
    csv_path: str = "data/raw_sales.csv",
    output_csv: str = "data/preprocessed.csv",
) -> str:
    from sklearn.preprocessing import StandardScaler

    df = pd.read_csv(csv_path)
    n_rows = len(df)

    # Outlier capping
    mean_s, std_s = df["sales"].mean(), df["sales"].std()
    cap = mean_s + 3 * std_s
    n_outliers = (df["sales"] > cap).sum()
    df.loc[df["sales"] > cap, "sales"] = cap

    # Sales-difference & rolling-mean features
    prev = df.sales.shift(1)
    df["difference_1"] = df.sales - prev
    for i in range(1, 7):
        df[f"difference_{i+1}"] = df["difference_1"].shift(i)
    # Raw sales lags (model can learn non-linear lag relationships)
    for i in range(1, 8):
        df[f"sales_lag_{i}"] = df.sales.shift(i)
    df["moving_average_week"] = df.sales.rolling(7).mean()
    df["moving_average_two_weeks"] = df.sales.rolling(14).mean()
    df["moving_average_month"] = df.sales.rolling(30).mean()
    df["difference_month"] = df.sales - df.sales.shift(28)
    df["difference_year"] = df.sales - df.sales.shift(366)

    # Calendar features
    df["dates"] = pd.to_datetime(df["dates"])
    df["day_of_week"] = df.dates.dt.day_of_week
    df["day_of_year"] = df.dates.dt.day_of_year
    df["quarter"] = df.dates.dt.quarter
    df["month"] = df.dates.dt.month
    df["year"] = df.dates.dt.year

    # Cyclical sine/cosine encoding for day-of-week (captures Mon≈Sun proximity)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    df = df.dropna()

    # One-hot encode categoricals
    cat_cols = ["day_of_week", "day_of_year", "quarter", "month", "year"]
    df = pd.get_dummies(df, columns=cat_cols)

    # Scale numerical features (all diff, rolling, lag, and cyclical columns)
    num_cols = [
        c for c in df.columns
        if c.startswith("difference_") or c.startswith("moving_average_")
        or c.startswith("sales_lag_") or c.endswith("_sin") or c.endswith("_cos")
    ]
    scaler = StandardScaler()
    scaler.fit(df[num_cols])
    df[num_cols] = scaler.transform(df[num_cols])

    # Save scaler and feature metadata for later use
    scaler_path = output_csv.replace(".csv", "_scaler.pkl")
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    meta = {
        "feature_cols": [c for c in df.columns if c not in {"product_id", "sales", "dates"}],
        "num_cols": num_cols,
        "cat_cols": cat_cols,
    }
    import json

    meta_path = output_csv.replace(".csv", "_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f)

    df.to_csv(output_csv, index=False)

    feature_count = len(df.columns) - 2  # exclude product_id & sales
    return (
        f"Preprocessing complete.\n"
        f"  Rows: {n_rows} -> {len(df)} (after NaN drop)\n"
        f"  Outliers capped: {n_outliers}\n"
        f"  Features created: {feature_count}\n"
        f"  Saved to: {output_csv}\n"
        f"  Scaler saved to: {scaler_path}"
    )
```

**Notes:**

- The scaler and metadata files are **required** by `forecast_next_7_days` — keep them alongside the preprocessed CSV.
- The `dates` column is parsed automatically. Ensure your raw CSV has a `dates` column with parseable date strings.
- A `product_id` column is ignored if present (the pipeline treats the series as univariate).

## 3. Train XGBoost — `train_xgboost_forecaster`

Trains an `XGBRegressor` on the preprocessed features using **time-series cross-validation** (`TimeSeriesSplit` with 5 splits, 60-sample test windows, and a gap of 1). After cross-validation, a final model is fit on all available data and saved to disk.

**Parameters:**

| Parameter               | Type    | Default                   | Description                     |
|-------------------------|---------|---------------------------|---------------------------------|
| `preprocessed_csv`      | `str`   | `'data/preprocessed.csv'` | Path to preprocessed CSV        |
| `model_output_path`     | `str`   | `'model/forecaster.pkl'`  | Where to save the trained model |
| `n_estimators`          | `int`   | `2000`                    | Number of boosting rounds       |
| `max_depth`             | `int`   | `6`                       | Maximum tree depth              |
| `learning_rate`         | `float` | `0.03`                    | Learning rate                   |
| `early_stopping_rounds` | `int`   | `80`                      | Early stopping patience         |
| `subsample`             | `float` | `0.85`                    | Row subsample ratio per tree    |
| `colsample_bytree`      | `float` | `0.85`                    | Column subsample ratio per tree |
| `reg_alpha`             | `float` | `0.1`                     | L1 regularisation               |

**Code:**

```python
def train_xgboost_forecaster(
    preprocessed_csv: str = "data/preprocessed.csv",
    model_output_path: str = "model/forecaster.pkl",
    n_estimators: int = 2000,
    max_depth: int = 6,
    learning_rate: float = 0.03,
    early_stopping_rounds: int = 80,
    subsample: float = 0.85,
    colsample_bytree: float = 0.85,
    reg_alpha: float = 0.1,
) -> str:
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import TimeSeriesSplit
    import xgboost as xgb

    df = pd.read_csv(preprocessed_csv)

    exclude = {"product_id", "sales", "dates"}
    feature_cols = [c for c in df.columns if c not in exclude]

    X = df[feature_cols].values
    y = df["sales"].values

    tss = TimeSeriesSplit(n_splits=5, test_size=60, gap=1)
    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        booster="gbtree",
        early_stopping_rounds=early_stopping_rounds,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        objective="reg:squarederror",
        random_state=42,
    )

    scores = []
    for fold, (train_idx, val_idx) in enumerate(tss.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False,
        )
        preds = model.predict(X_val)
        rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
        scores.append(rmse)

    # Final fit on all data
    model.fit(X, y, eval_set=[(X, y)], verbose=False)

    Path(model_output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(model_output_path, "wb") as f:
        pickle.dump(model, f)

    avg_rmse = np.mean(scores)
    return (
        f"XGBRegressor trained successfully.\n"
        f"  Features: {len(feature_cols)}\n"
        f"  Samples: {len(df)}\n"
        f"  CV RMSE: {[f'{s:.2f}' for s in scores]}\n"
        f"  Average RMSE: {avg_rmse:.2f}\n"
        f"  Model saved to: {model_output_path}"
    )
```

**Notes:**

- Cross-validation uses **4 actual folds** (5 splits with 60 test samples each). The `gap=1` parameter ensures no data leakage between train and test sets.
- The model is saved via `pickle`. Load it later with `pickle.load()`.
- Feature columns are inferred automatically — all columns except `product_id`, `sales`, and `dates` are treated as predictors.

## 4. Forecast Next 7 Days — `forecast_next_7_days`

Loads the trained model, scaler, and feature metadata, then iteratively predicts sales for the next 7 days. Each prediction is appended to the sales history so subsequent lag features are computed correctly (autoregressive roll-forward).

A **seasonal blending** step mixes the model's prediction with the historical monthly average to keep forecasts grounded:

```
prediction = 0.75 × model_output + 0.25 × historical_month_mean
```

**Parameters:**

| Parameter             | Type  | Default                   | Description                  |
|-----------------------|-------|---------------------------|------------------------------|
| `model_path`          | `str` | `'model/forecaster.pkl'`  | Path to the trained model    |
| `original_csv`        | `str` | `'data/raw_sales.csv'`    | Path to the raw dataset CSV  |
| `preprocessed_csv`    | `str` | `'data/preprocessed.csv'` | Path to the preprocessed CSV |
| `forecast_output_csv` | `str` | `'data/forecast.csv'`     | Where to save the forecast   |

**Code:**

```python
def forecast_next_7_days(
    model_path: str = "model/forecaster.pkl",
    original_csv: str = "data/raw_sales.csv",
    preprocessed_csv: str = "data/preprocessed.csv",
    forecast_output_csv: str = "data/forecast.csv",
) -> str:
    import json

    scaler_path = preprocessed_csv.replace(".csv", "_scaler.pkl")
    meta_path = preprocessed_csv.replace(".csv", "_meta.json")

    if not os.path.exists(scaler_path):
        return f"Error: scaler file not found at '{scaler_path}'."
    if not os.path.exists(meta_path):
        return f"Error: metadata file not found at '{meta_path}'. Re-run preprocessing."

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    with open(meta_path) as f:
        meta = json.load(f)

    feature_cols = meta["feature_cols"]
    num_cols = meta["num_cols"]

    # Load raw data and last preprocessed row
    raw = pd.read_csv(original_csv)
    raw["dates"] = pd.to_datetime(raw["dates"])

    preproc = pd.read_csv(preprocessed_csv)
    last_features = preproc.iloc[-1][feature_cols].to_dict()

    # Build a sales history series (with outlier cap) for lag computation
    mean_s, std_s = raw["sales"].mean(), raw["sales"].std()
    cap = mean_s + 3 * std_s
    sales_series = raw["sales"].clip(upper=cap).tolist()
    last_date = raw["dates"].max()

    forecasts = []
    for day_offset in range(1, 8):
        future_date = last_date + timedelta(days=day_offset)

        # Build feature dict from last known features, updating calendar + lags
        row = {}
        for col in feature_cols:
            if col.startswith("difference_month"):
                row[col] = sales_series[-1] - sales_series[-29] if len(sales_series) >= 29 else last_features.get(col, 0.0)
            elif col.startswith("difference_year"):
                row[col] = sales_series[-1] - sales_series[-367] if len(sales_series) >= 367 else last_features.get(col, 0.0)
            elif col.startswith("difference_"):
                # Recompute lag features from the sales series
                idx = int(col.split("_")[1])  # e.g. difference_1 -> 1
                if len(sales_series) >= idx + 1:
                    row[col] = sales_series[-1] - sales_series[-(idx + 1)]
                else:
                    row[col] = last_features.get(col, 0.0)
            elif col.startswith("sales_lag_"):
                idx = int(col.split("_")[-1])
                if len(sales_series) >= idx:
                    row[col] = sales_series[-(idx)]
                else:
                    row[col] = last_features.get(col, 0.0)
            elif col.startswith("moving_average_"):
                if "month" in col:
                    window = 30
                elif "two_weeks" in col:
                    window = 14
                else:
                    window = 7
                if len(sales_series) >= window:
                    row[col] = np.mean(sales_series[-window:])
                else:
                    row[col] = last_features.get(col, 0.0)
            elif col == "dow_sin":
                row[col] = float(np.sin(2 * np.pi * future_date.day_of_week / 7))
            elif col == "dow_cos":
                row[col] = float(np.cos(2 * np.pi * future_date.day_of_week / 7))
            elif col == "month_sin":
                row[col] = float(np.sin(2 * np.pi * future_date.month / 12))
            elif col == "month_cos":
                row[col] = float(np.cos(2 * np.pi * future_date.month / 12))
            elif col.startswith("day_of_week_"):
                row[col] = 1.0 if int(col.split("_")[-1]) == future_date.day_of_week else 0.0
            elif col.startswith("day_of_year_"):
                row[col] = 1.0 if int(col.split("_")[-1]) == future_date.day_of_year else 0.0
            elif col.startswith("quarter_"):
                row[col] = 1.0 if int(col.split("_")[-1]) == future_date.quarter else 0.0
            elif col.startswith("month_"):
                row[col] = 1.0 if int(col.split("_")[-1]) == future_date.month else 0.0
            elif col.startswith("year_"):
                row[col] = 1.0 if int(col.split("_")[-1]) == future_date.year else 0.0
            else:
                row[col] = last_features.get(col, 0.0)

        # Build DataFrame row and scale
        row_df = pd.DataFrame([row])[feature_cols]
        row_df[num_cols] = scaler.transform(row_df[num_cols])

        pred = float(model.predict(row_df.values)[0])
        pred = abs(pred)
        # Blend with seasonal baseline to keep predictions anchored to reality
        month_avg = raw[raw["dates"].dt.month == future_date.month]["sales"].mean()
        pred = 0.75 * pred + 0.25 * month_avg  # 75% model, 25% historical monthly mean
        forecasts.append({"dates": future_date, "sales": pred})

        # Append prediction to sales series so subsequent lags use it
        sales_series.append(pred)

    forecast_df = pd.DataFrame(forecasts)
    Path(forecast_output_csv).parent.mkdir(parents=True, exist_ok=True)
    forecast_df.to_csv(forecast_output_csv, index=False)

    total = forecast_df["sales"].sum()
    daily = "\n".join(f"  {r.dates.date()}  ->  {r.sales:.2f} Kg" for _, r in forecast_df.iterrows())

    return (
        f"7-day forecast generated.\n"
        f"{daily}\n"
        f"  Total (7 days): {total:.2f} Kg\n"
        f"  Avg per day:    {forecast_df['sales'].mean():.2f} Kg\n"
        f"  Saved to: {forecast_output_csv}"
    )
```

**Notes:**

- The autoregressive roll-forward means each day's prediction influences the next day's lags and differences. This is why the loop iterates one day at a time rather than predicting all 7 at once.
- Seasonal blending (75% model / 25% historical monthly mean) acts as a regulariser, preventing the model from drifting too far from historical patterns.
- Predictions are forced to be non-negative via `abs(pred)`.

## 5. Create Forecast Plot — `create_forecast_plot`

Generates two matplotlib figures:

1. **Full view** — plots the entire sales history as a blue line and overlays the 7-day forecast as a dashed red line with markers.
2. **Zoom view** — shows only the last 7 days of actual data alongside the 7-day forecast, with a dotted grey line connecting the last actual value to the first forecast value.

Both plots are saved as high-resolution PNGs (150 dpi).

**Parameters:**

| Parameter          | Type  | Default                | Description                 |
|--------------------|-------|------------------------|-----------------------------|
| `original_csv`     | `str` | `'data/raw_sales.csv'` | Path to the raw dataset CSV |
| `forecast_csv`     | `str` | `'data/forecast.csv'`  | Path to the forecast CSV    |
| `plot_output_path` | `str` | `'plots/forecast.png'` | Path for the full-view plot |

The zoom plot is automatically saved alongside the main plot with a `_zoom` suffix (e.g., `plots/forecast_zoom.png`).

**Code:**

```python
def create_forecast_plot(
    original_csv: str = "data/raw_sales.csv",
    forecast_csv: str = "data/forecast.csv",
    plot_output_path: str = "plots/forecast.png",
) -> str:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    raw = pd.read_csv(original_csv)
    raw["dates"] = pd.to_datetime(raw["dates"])
    forecast = pd.read_csv(forecast_csv, parse_dates=["dates"])

    # --- Full history + forecast ---
    fig1, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(raw["dates"], raw["sales"],
             color="#00b4d8", linewidth=1.0, label="Sales History")
    ax1.plot(forecast["dates"], forecast["sales"],
             color="#e63946", linewidth=2,
             linestyle="--", marker="o", label="7-Day Forecast")
    ax1.set_title("Chocolate Sales — 7-Day Forecast (Full History)",
                  fontsize=14, fontweight="bold")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Sales (Kg)")
    ax1.legend(loc="upper left")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig1.autofmt_xdate()
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()

    Path(plot_output_path).parent.mkdir(parents=True, exist_ok=True)
    fig1.savefig(plot_output_path, dpi=150)
    plt.close(fig1)

    # --- Zoom view: last 7 actuals + 7 forecast ---
    zoom_path = plot_output_path.replace(".png", "_zoom.png")
    last_actual_date = raw["dates"].max()
    zoom_start = last_actual_date - timedelta(days=6)
    raw_zoom = raw[raw["dates"] >= pd.Timestamp(zoom_start)]

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(raw_zoom["dates"], raw_zoom["sales"],
             color="#00b4d8", linewidth=2, marker="s",
             label="Last 7 Days (Actual)")
    ax2.plot(forecast["dates"], forecast["sales"],
             color="#e63946", linewidth=2,
             linestyle="--", marker="o",
             label="Next 7 Days (Forecast)")
    # Bridge line
    bridge_x = [raw_zoom["dates"].iloc[-1], forecast["dates"].iloc[0]]
    bridge_y = [raw_zoom["sales"].iloc[-1], forecast["sales"].iloc[0]]
    ax2.plot(bridge_x, bridge_y, color="gray",
             linewidth=1, linestyle=":", alpha=0.6)

    ax2.set_title("Chocolate Sales — Last 7 Days vs Next 7 Days",
                  fontsize=14, fontweight="bold")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Sales (Kg)")
    ax2.legend(loc="upper left")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig2.autofmt_xdate()
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(zoom_path, dpi=150)
    plt.close(fig2)

    return f"Plots saved to '{plot_output_path}' and '{zoom_path}'."
```

**Notes:**

- Uses `matplotlib.use("Agg")` to avoid tkinter errors in headless/terminal environments.
- The zoom plot includes a **bridge line** (dotted grey) connecting the last actual data point to the first forecast point, making the transition visually clear.
- Both plots use a clean, modern colour scheme: blue for historical data, red for forecast.

## 6. Generate Final Report — `generate_final_report`

Collects all pipeline artifacts into a dated report folder (e.g., `report-2026-06-12/`) and writes a structured `report.md` that includes:

- Dataset statistics (row count, mean/min/max sales).
- A formatted table of the 7-day forecast with predicted sales per day.
- An artifact inventory table linking to each file.
- Embedded PNG plots (full history and zoom views).

**Parameters:**

| Parameter          | Type  | Default                   | Description                                        |
|--------------------|-------|---------------------------|----------------------------------------------------|
| `dataset_csv`      | `str` | `'data/raw_sales.csv'`    | Path to the raw dataset                            |
| `forecast_csv`     | `str` | `'data/forecast.csv'`     | Path to the forecast CSV                           |
| `plot_path`        | `str` | `'plots/forecast.png'`    | Path to the forecast plot                          |
| `model_path`       | `str` | `'model/forecaster.pkl'`  | Path to the trained model                          |
| `preprocessed_csv` | `str` | `'data/preprocessed.csv'` | Path to the preprocessed CSV                       |
| `report_folder`    | `str` | `'report'`                | Base folder name (date suffix added automatically) |

**Code:**

```python
def generate_final_report(
    dataset_csv: str = "data/raw_sales.csv",
    forecast_csv: str = "data/forecast.csv",
    plot_path: str = "plots/forecast.png",
    model_path: str = "model/forecaster.pkl",
    preprocessed_csv: str = "data/preprocessed.csv",
    report_folder: str = "report",
) -> str:
    import shutil

    date_str = datetime.now().strftime("%Y-%m-%d")
    folder = f"{report_folder}-{date_str}"
    os.makedirs(folder, exist_ok=True)

    # Copy all artifacts into the report folder
    zoom_path = plot_path.replace(".png", "_zoom.png")
    for src in [dataset_csv, forecast_csv, plot_path,
                zoom_path, model_path, preprocessed_csv]:
        if os.path.exists(src):
            shutil.copy2(src, folder)

    # Also copy scaler + meta if they exist
    for suffix in ["_scaler.pkl", "_meta.json"]:
        path = preprocessed_csv.replace(".csv", suffix)
        if os.path.exists(path):
            shutil.copy2(path, folder)

    # Build report.md
    forecast = pd.read_csv(forecast_csv, parse_dates=["dates"])
    raw = pd.read_csv(dataset_csv)

    lines = [
        "# Time-Series Forecast Report",
        "",
        f"**Generated:** {date_str}",
        "",
        "## Dataset",
        f"- Source: `AiresPucrs/time-series-data`",
        f"- Rows: {len(raw)}",
        f"- Mean sales: {raw['sales'].mean():.2f} Kg",
        f"- Min sales:  {raw['sales'].min():.2f} Kg",
        f"- Max sales:  {raw['sales'].max():.2f} Kg",
        "",
        "## 7-Day Forecast",
        "",
        "| Date       | Predicted Sales (Kg) |",
        "|------------|----------------------|",
    ]
    for _, row in forecast.iterrows():
        lines.append(
            f"| {row['dates'].date()} | {row['sales']:.2f} |"
        )

    total = forecast["sales"].sum()
    avg = forecast["sales"].mean()
    lines += [
        "",
        f"- **Total (7 days):** {total:.2f} Kg",
        f"- **Average per day:** {avg:.2f} Kg",
        "",
        "## Artifacts",
        "",
        "| File | Description |",
        "|------|-------------|",
        f"| `{os.path.basename(dataset_csv)}` | Raw dataset |",
        f"| `{os.path.basename(preprocessed_csv)}` | Preprocessed features |",
        f"| `{os.path.basename(model_path)}` | Trained XGBoost model |",
        f"| `{os.path.basename(forecast_csv)}` | 7-day forecast data |",
        f"| `{os.path.basename(plot_path)}` | Forecast plot — full history |",
        f"| `{os.path.basename(zoom_path)}` | Forecast plot — zoom view |",
    ]

    # Embed plot images
    # ...
    report_path = os.path.join(folder, "report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    return f"Report generated in '{folder}/'. See '{report_path}' for the summary."
```

**Notes:**

- The report folder is date-stamped automatically, so multiple runs won't overwrite each other.
- The report embeds the PNG plots using standard Markdown image syntax — they render in any Markdown viewer that supports local images.
- All intermediate artifacts (CSV, model, scaler, meta) are copied into the report folder, making it a self-contained snapshot of the run.

## Notes

- The pipeline expects a CSV with at least a `dates` column (parseable dates) and a `sales` column (numeric target). An optional `product_id` column is ignored.
- For multivariate forecasting (multiple products), the agent would need to loop over each product or extend the feature set — the current tools handle a single univariate series.
- The scaler and metadata files from preprocessing are essential for `forecast_next_7_days`. If you delete them, re-run preprocessing.

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

These packages should already be installed in the virtual environment. If any are missing, install them with `pip install datasets pandas numpy scikit-learn xgboost matplotlib`.

---

## 1. Download Dataset — `download_dataset_from_hub`

Fetches a dataset from the Hugging Face Hub using the `datasets` library and saves it as a local CSV. The output directory is created automatically — you do not need to create it beforehand.

| Parameter      | Type  | Default                | Description                                                     |
|----------------|-------|------------------------|-----------------------------------------------------------------|
| `dataset_path` | `str` | —                      | Hugging Face dataset path, e.g. `'AiresPucrs/time-series-data'` |
| `split`        | `str` | `'train'`              | Dataset split to load                                           |
| `output_csv`   | `str` | `'data/raw_sales.csv'` | Where to save the CSV                                           |

> **Tip:** Use `dataset_path='AiresPucrs/time-series-data'` for chocolate sales forecasting. Supports any dataset on the Hugging Face Hub compatible with `load_dataset()`.

---

## 2. Preprocess Time-Series — `preprocess_time_series_data`

Transforms raw sales data into a feature-rich matrix ready for XGBoost. The pipeline performs the following steps in order:

1. **Outlier capping** — values above $mean + 3\sigma$ are clamped to the cap.
2. **Lag features** — `difference_1` through `difference_7` (consecutive differences), `difference_month` (28-day), `difference_year` (366-day).
3. **Raw sales lags** — `sales_lag_1` through `sales_lag_7` (shifted raw sales).
4. **Rolling averages** — 7-day, 14-day, and 30-day moving means.
5. **Calendar features** — day-of-week, day-of-year, quarter, month, year extracted from the `dates` column.
6. **Cyclical encoding** — sine/cosine transforms for day-of-week and month to preserve circular proximity.
7. **One-hot encoding** — categorical calendar columns expanded into binary indicators.
8. **Standard scaling** — all numeric features scaled to zero mean, unit variance.

Rows with `NaN` (from lag shifts at the start of the series) are automatically dropped.

| Parameter    | Type  | Default                   | Description                        |
|--------------|-------|---------------------------|------------------------------------|
| `csv_path`   | `str` | `'data/raw_sales.csv'`    | Path to the raw CSV                |
| `output_csv` | `str` | `'data/preprocessed.csv'` | Where to save the preprocessed CSV |

**Persistent artifacts produced:**

| File                           | Purpose                                          |
|--------------------------------|--------------------------------------------------|
| `data/preprocessed.csv`        | Full preprocessed DataFrame                      |
| `data/preprocessed_scaler.pkl` | Fitted `StandardScaler` (needed by step 4)       |
| `data/preprocessed_meta.json`  | Feature column metadata (needed by step 4)       |

> **Critical:** The scaler and metadata files are required by `forecast_next_7_days`. Always run this step before forecasting, and keep the artifacts alongside the preprocessed CSV. The pipeline expects a `dates` column with parseable date strings and a numeric `sales` column. A `product_id` column is ignored (univariate series).

---

## 3. Train XGBoost — `train_xgboost_forecaster`

Trains an `XGBRegressor` on the preprocessed features using **time-series cross-validation** (`TimeSeriesSplit`, 5 splits, 60-sample test windows, gap=1). After cross-validation, a final model is fit on all available data and saved to disk via `pickle`.

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

> **Tip:** The default hyperparameters are tuned for a univariate daily sales series. For smaller datasets, consider lowering `n_estimators` and `max_depth`. Feature columns are inferred automatically — everything except `product_id`, `sales`, and `dates` is treated as a predictor. Cross-validation uses a `gap=1` to prevent data leakage between train and test folds.

---

## 4. Forecast Next 7 Days — `forecast_next_7_days`

Loads the trained model, scaler, and feature metadata, then **iteratively** predicts sales for the next 7 days using autoregressive roll-forward. Each day's prediction is appended to the sales history so subsequent lags and differences are computed correctly.

Predictions are blended with a seasonal baseline to prevent drift:

$$\text{pred} = 0.75 \times \text{model\_output} + 0.25 \times \text{historical\_month\_mean}$$

All predictions are forced non-negative.

| Parameter             | Type  | Default                   | Description                  |
|-----------------------|-------|---------------------------|------------------------------|
| `model_path`          | `str` | `'model/forecaster.pkl'`  | Path to the trained model    |
| `original_csv`        | `str` | `'data/raw_sales.csv'`    | Path to the raw dataset CSV  |
| `preprocessed_csv`    | `str` | `'data/preprocessed.csv'` | Path to the preprocessed CSV |
| `forecast_output_csv` | `str` | `'data/forecast.csv'`     | Where to save the forecast   |

> **Critical:** This step requires the scaler (`_scaler.pkl`) and metadata (`_meta.json`) files produced by preprocessing. If they are missing, re-run `preprocess_time_series_data` first. The loop predicts one day at a time (not all 7 at once) because each prediction influences the next day's lag features.

---

## 5. Create Forecast Plot — `create_forecast_plot`

Generates two publication-quality matplotlib figures:

1. **Full view** — entire sales history (blue line) with the 7-day forecast overlaid (dashed red line with markers).
2. **Zoom view** — last 7 actual days vs. next 7 forecast days, with a dotted grey bridge line connecting the last actual to the first forecast.

Both are saved as 150 dpi PNGs using the Agg backend (safe for headless environments).

| Parameter          | Type  | Default                | Description                 |
|--------------------|-------|------------------------|-----------------------------|
| `original_csv`     | `str` | `'data/raw_sales.csv'` | Path to the raw dataset CSV |
| `forecast_csv`     | `str` | `'data/forecast.csv'`  | Path to the forecast CSV    |
| `plot_output_path` | `str` | `'plots/forecast.png'` | Path for the full-view plot |

The zoom plot is automatically saved alongside the main plot with a `_zoom` suffix (e.g., `plots/forecast_zoom.png`).

---

## 6. Generate Final Report — `generate_final_report`

Collects all pipeline artifacts into a date-stamped report folder (e.g., `report-2026-06-12/`) and writes `report.md` containing:

- Dataset statistics (row count, mean/min/max sales)
- A formatted table of the 7-day forecast with daily predictions
- An artifact inventory linking to each file
- Embedded PNG plots

| Parameter          | Type  | Default                   | Description                                        |
|--------------------|-------|---------------------------|----------------------------------------------------|
| `dataset_csv`      | `str` | `'data/raw_sales.csv'`    | Path to the raw dataset                            |
| `forecast_csv`     | `str` | `'data/forecast.csv'`     | Path to the forecast CSV                           |
| `plot_path`        | `str` | `'plots/forecast.png'`    | Path to the forecast plot                          |
| `model_path`       | `str` | `'model/forecaster.pkl'`  | Path to the trained model                          |
| `preprocessed_csv` | `str` | `'data/preprocessed.csv'` | Path to the preprocessed CSV                       |
| `report_folder`    | `str` | `'report'`                | Base folder name (date suffix added automatically) |

> **Tip:** All intermediate artifacts (CSVs, model, scaler, meta, plots) are copied into the report folder, making it a self-contained snapshot of the run. The report folder is date-stamped so multiple runs won't overwrite each other.

---

## Execution Strategy

Call these tools **in order**. Each tool depends on the output of the previous one:

1. `download_dataset_from_hub(dataset_path='AiresPucrs/time-series-data')`
2. `preprocess_time_series_data()` — uses defaults from step 1
3. `train_xgboost_forecaster()` — uses defaults from step 2
4. `forecast_next_7_days()` — uses defaults from steps 2 & 3
5. `create_forecast_plot()` — uses defaults from steps 1 & 4
6. `generate_final_report()` — collects all artifacts

Use the default parameter values unless you have a specific reason to change them. Each tool returns a human-readable summary on success — read it to verify the step completed correctly before proceeding to the next one.

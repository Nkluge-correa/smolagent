"""
Custom tools for the time-series forecasting agent.

Tool pipeline:
  1. download_dataset_from_hub    ->  fetch raw CSV from Hugging Face Hub
  2. preprocess_time_series_data  ->  clean outliers, engineer features, scale, save
  3. train_xgboost_forecaster     ->  train XGBoost with cross-validation, save model
  4. forecast_next_7_days         ->  load model, predict next week, save forecast
  5. create_forecast_plot         ->  plot historical + forecast data
  6. generate_final_report        ->  collect all artifacts into a dated report folder
"""

import os
import pickle
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from smolagents import tool


@tool
def download_dataset_from_hub(dataset_path: str, output_csv: str) -> str:
    """
    Downloads a dataset from the Hugging Face Hub and saves it locally as a CSV file.

    Args:
        dataset_path: The Hugging Face Hub dataset path (e.g., 'AiresPucrs/time-series-data').
        output_csv:    The local filename to save the CSV to (e.g., 'data/sales.csv').

    Returns:
        A success message with the number of rows saved, or an error message.
    """
    from datasets import load_dataset
    import os

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)

    try:
        dataset = load_dataset(dataset_path, split="train")
        df = dataset.to_pandas()
        df.to_csv(output_csv, index=False)
        return f"Successfully downloaded '{dataset_path}' ({len(df)} rows) -> saved to '{output_csv}'."
    except Exception as e:
        return f"Error downloading '{dataset_path}': {e}"


@tool
def preprocess_time_series_data(csv_path: str, output_csv: str) -> str:
    """
    Preprocesses a raw time-series CSV for XGBoost forecasting:
      - Replaces outliers (> mean + 3*std) with the cap value.
      - Creates sales-difference features (lag 1-7, 28, 366).
      - Creates rolling-mean features (7-day, 14-day).
      - Creates calendar features (day-of-week, month, quarter, year).
      - One-hot encodes categorical features.
      - Standard-scales numerical features.
    Saves the preprocessed DataFrame to `output_csv`.

    Args:
        csv_path:   Path to the raw CSV (must have 'dates' and 'sales' columns).
        output_csv: Path where the preprocessed CSV will be saved.

    Returns:
        Summary statistics about the preprocessing.
    """
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


@tool
def train_xgboost_forecaster(
    preprocessed_csv: str,
    model_output_path: str,
    n_estimators: int = 2000,
    max_depth: int = 6,
    learning_rate: float = 0.03,
    early_stopping_rounds: int = 80,
    subsample: float = 0.85,
    colsample_bytree: float = 0.85,
    reg_alpha: float = 0.1,
) -> str:
    """
    Trains an XGBRegressor on preprocessed time-series data using TimeSeriesSplit
    cross-validation, then saves the model to disk.

    Args:
        preprocessed_csv:      Path to the CSV from `preprocess_time_series_data`.
        model_output_path:     Where to save the trained model (.pkl).
        n_estimators:          Number of boosting rounds (default 2000).
        max_depth:             Maximum tree depth (default 6).
        learning_rate:         Learning rate (default 0.03).
        early_stopping_rounds: Early stopping patience (default 80).
        subsample:             Row subsample ratio per tree (default 0.85).
        colsample_bytree:      Column subsample ratio per tree (default 0.85).
        reg_alpha:             L1 regularization (default 0.1).

    Returns:
        Cross-validation RMSE scores and final model info.
    """
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import TimeSeriesSplit
    import xgboost as xgb

    df = pd.read_csv(preprocessed_csv)

    # Identify feature columns (everything except id, target, and date)
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

    with open(model_output_path, "wb") as f:
        pickle.dump(model, f)

    avg_rmse = np.mean(scores)
    return (
        f"XGBRegressor trained successfully.\n"
        f"  Features: {len(feature_cols)}\n"
        f"  Samples: {len(df)}\n"
        f"  CV folds: 4  |  RMSE per fold: {[f'{s:.2f}' for s in scores]}\n"
        f"  Average RMSE: {avg_rmse:.2f}\n"
        f"  Model saved to: {model_output_path}"
    )


@tool
def forecast_next_7_days(
    model_path: str,
    original_csv: str,
    preprocessed_csv: str,
    forecast_output_csv: str,
) -> str:
    """
    Loads a trained XGBRegressor and generates a 7-day sales forecast,
    saving the results to a CSV.

    Args:
        model_path:          Path to the trained model (.pkl).
        original_csv:        Path to the *raw* dataset CSV (with dates + sales).
        preprocessed_csv:    Path to the preprocessed CSV (for scaler and feature structure).
        forecast_output_csv: Where to save the 7-day forecast CSV.

    Returns:
        Forecast summary (predicted sales per day, total, average).
    """
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


@tool
def create_forecast_plot(
    original_csv: str,
    forecast_csv: str,
    plot_output_path: str,
) -> str:
    """
    Creates two matplotlib PNG figures showing the forecast:

    1. **Full view** — entire sales history + 7-day forecast.
    2. **Zoom view** — last 7 days of actuals + 7-day forecast (detail).

    Args:
        original_csv:    Path to the raw dataset CSV.
        forecast_csv:    Path to the forecast CSV (from `forecast_next_7_days`).
        plot_output_path: Base path for the full-view plot (PNG).
                          The zoom view is saved alongside as '*_zoom.png'.

    Returns:
        Confirmation message with both file paths.
    """
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend — avoids tkinter errors
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    raw = pd.read_csv(original_csv)
    raw["dates"] = pd.to_datetime(raw["dates"])

    forecast = pd.read_csv(forecast_csv, parse_dates=["dates"])

    # ---- Plot 1: Full history + forecast ----
    fig1, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(raw["dates"], raw["sales"], color="#00b4d8", linewidth=1.0, label="Sales History")
    ax1.plot(forecast["dates"], forecast["sales"], color="#e63946",
             linewidth=2, linestyle="--", marker="o", label="7-Day Forecast")
    ax1.set_title("Chocolate Sales — 7-Day Forecast (Full History)", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Sales (Kg)")
    ax1.legend(loc="upper left")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig1.autofmt_xdate()
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(plot_output_path, dpi=150)
    plt.close(fig1)

    # ---- Plot 2: Zoom — last 7 days + forecast ----
    zoom_path = plot_output_path.replace(".png", "_zoom.png")
    last_actual_date = raw["dates"].max()
    zoom_start = last_actual_date - timedelta(days=6)  # 7 days back
    raw_zoom = raw[raw["dates"] >= pd.Timestamp(zoom_start)]

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(raw_zoom["dates"], raw_zoom["sales"], color="#00b4d8",
             linewidth=2, marker="s", label="Last 7 Days (Actual)")
    ax2.plot(forecast["dates"], forecast["sales"], color="#e63946",
             linewidth=2, linestyle="--", marker="o", label="Next 7 Days (Forecast)")
    # Connect last actual to first forecast with a dotted line
    bridge_x = [raw_zoom["dates"].iloc[-1], forecast["dates"].iloc[0]]
    bridge_y = [raw_zoom["sales"].iloc[-1], forecast["sales"].iloc[0]]
    ax2.plot(bridge_x, bridge_y, color="gray", linewidth=1, linestyle=":", alpha=0.6)

    ax2.set_title("Chocolate Sales — Last 7 Days vs Next 7 Days", fontsize=14, fontweight="bold")
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


@tool
def generate_final_report(
    dataset_csv: str,
    forecast_csv: str,
    plot_path: str,
    model_path: str,
    preprocessed_csv: str,
    report_folder: str = "report",
) -> str:
    """
    Collects all artifacts into a dated report folder and writes a report.md
    with the forecast summary and statistics.

    Args:
        dataset_csv:      Path to the original dataset CSV.
        forecast_csv:     Path to the forecast CSV.
        plot_path:        Path to the forecast plot HTML.
        model_path:       Path to the trained model (.pkl).
        preprocessed_csv: Path to the preprocessed CSV.
        report_folder:    Base folder name (default 'report'). A date suffix is
                          appended automatically, e.g., 'report-2026-06-09'.

    Returns:
        Path to the report folder.
    """
    import shutil

    date_str = datetime.now().strftime("%Y-%m-%d")
    folder = f"{report_folder}-{date_str}"
    os.makedirs(folder, exist_ok=True)

    # Copy artifacts (including zoom plot)
    zoom_path = plot_path.replace(".png", "_zoom.png")
    for src in [dataset_csv, forecast_csv, plot_path, zoom_path, model_path, preprocessed_csv]:
        if os.path.exists(src):
            shutil.copy2(src, folder)

    # Also copy scaler + meta if they exist
    scaler_path = preprocessed_csv.replace(".csv", "_scaler.pkl")
    if os.path.exists(scaler_path):
        shutil.copy2(scaler_path, folder)
    meta_path_ = preprocessed_csv.replace(".csv", "_meta.json")
    if os.path.exists(meta_path_):
        shutil.copy2(meta_path_, folder)

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
        lines.append(f"| {row['dates'].date()} | {row['sales']:.2f} |")

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
        f"| `{os.path.basename(zoom_path)}` | Forecast plot — last 7 vs next 7 days |",
    ]

    # Embed the plot images
    plot_basename = os.path.basename(plot_path)
    zoom_basename = os.path.basename(zoom_path)
    if os.path.exists(os.path.join(folder, plot_basename)):
        lines += [
            "",
            "## Forecast Plot — Full History",
            "",
            f"![Full History]({plot_basename})",
        ]
    if os.path.exists(os.path.join(folder, zoom_basename)):
        lines += [
            "",
            "## Forecast Plot — Zoom (Last 7 vs Next 7 Days)",
            "",
            f"![Zoom View]({zoom_basename})",
        ]

    report_path = os.path.join(folder, "report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    return f"Report generated in '{folder}/'.  See '{report_path}' for the summary."

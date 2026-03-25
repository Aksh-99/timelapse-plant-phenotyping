import os
import math
import warnings
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# ─── CONFIG ────────────────────────────────────────────────────────────────────
CSV_PATH = "data/predictions/height_dataset.csv"
MODEL_DIR = "models"

# Make sure these match your CSV header exactly
FEATURES = ["day", "height", "temperature", "stage_label"]
TARGET = "height"   # next-day target

RANDOM_SEED = 42
TEST_SIZE = 0.2

SPROUT_LABEL = 2   # 0=seed, 1=germination, 2=sprout

os.makedirs(MODEL_DIR, exist_ok=True)


# ─── 1. LOAD DATA ──────────────────────────────────────────────────────────────

def load_data(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required = ["day", "folder_name", "date", "temperature", "stage_label", "height"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")

    df = df.sort_values("day").reset_index(drop=True)
    df = df.dropna(subset=FEATURES + [TARGET])

    return df


def build_supervised(df: pd.DataFrame):
    """
    One-step forecasting:
      X[i] = [day_i, height_i, temperature_i, stage_label_i]
      y[i] = height_{i+1}
    """
    X_rows, y_rows = [], []

    for i in range(len(df) - 1):
        x_row = [df.loc[i, feat] for feat in FEATURES]
        y_val = df.loc[i + 1, TARGET]

        X_rows.append(x_row)
        y_rows.append(y_val)

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_rows, dtype=np.float32)
    return X, y


# ─── 2. METRICS ────────────────────────────────────────────────────────────────

def evaluate(name: str, y_true, y_pred) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2}


# ─── 3. MODELS ─────────────────────────────────────────────────────────────────

def train_linear(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


# ─── 4. SAVE / LOAD ────────────────────────────────────────────────────────────

def save_best(results: list, models: dict) -> str:
    best = min(results, key=lambda r: r["RMSE"])
    best_name = best["Model"]

    model_path = os.path.join(MODEL_DIR, "best_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(models[best_name], f)

    print(f"\nBest model: {best_name} (RMSE={best['RMSE']:.4f})")
    print(f"Saved model to: {model_path}")
    return best_name


def load_best_model():
    model_path = os.path.join(MODEL_DIR, "best_model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError("No saved model found. Train first.")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


# ─── 5. ONE-DAY PREDICTION ─────────────────────────────────────────────────────

def predict_next_day(day: int, height: float, temperature: float, stage_label: int) -> float:
    model = load_best_model()
    x = np.array([[day, height, temperature, stage_label]], dtype=np.float32)
    pred = model.predict(x)[0]
    return float(pred)


# ─── 6. FORECAST FROM FIRST SPROUT ─────────────────────────────────────────────

def forecast_from_first_sprout(df: pd.DataFrame) -> pd.DataFrame:
    """
    Find first row where stage_label == SPROUT_LABEL.
    Use that as the starting point.
    Predict each later day recursively and compare with actual height.
    """
    sprout_rows = df[df["stage_label"] == SPROUT_LABEL]

    if sprout_rows.empty:
        raise ValueError("No sprouting stage found in dataset.")

    first_sprout_idx = sprout_rows.index[0]

    # Need at least one later row to forecast
    if first_sprout_idx >= len(df) - 1:
        raise ValueError("Sprout is on the last row, so there are no later days to predict.")

    start_row = df.loc[first_sprout_idx]

    current_day = int(start_row["day"])
    current_height = float(start_row["height"])

    forecast_rows = []

    # predict each following row recursively
    for i in range(first_sprout_idx + 1, len(df)):
        row = df.loc[i]

        actual_day = int(row["day"])
        actual_temp = float(row["temperature"])
        actual_stage = int(row["stage_label"])
        actual_height = float(row["height"])

        predicted_height = predict_next_day(
            day=current_day,
            height=current_height,
            temperature=actual_temp,
            stage_label=actual_stage
        )

        forecast_rows.append({
            "day": actual_day,
            "date": row["date"],
            "temperature": actual_temp,
            "stage_label": actual_stage,
            "actual_height": round(actual_height, 4),
            "predicted_height": round(predicted_height, 4),
            "absolute_error": round(abs(actual_height - predicted_height), 4)
        })

        # recursive step: predicted height becomes next input height
        current_day = actual_day
        current_height = predicted_height

    return pd.DataFrame(forecast_rows)


# ─── 7. MAIN ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Plant Height Predictor")
    print("=" * 60)

    print(f"\n[1/5] Loading dataset: {CSV_PATH}")
    df = load_data(CSV_PATH)
    print(f"Rows loaded: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    if len(df) < 5:
        raise ValueError("Need at least 5 rows in the dataset.")

    X, y = build_supervised(df)

    if len(X) < 2:
        raise ValueError("Not enough supervised samples after conversion.")

    print(f"Supervised samples: {len(X)}")

    split_idx = max(1, min(len(X) - 1, int(len(X) * (1 - TEST_SIZE))))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"Train/Test split: {len(X_train)} / {len(X_test)}")

    results = []
    models = {}

    print("\n[2/5] Training Linear Regression")
    lr_model = train_linear(X_train, y_train)
    lr_preds = lr_model.predict(X_test)
    lr_metrics = evaluate("LinearRegression", y_test, lr_preds)
    results.append(lr_metrics)
    models["LinearRegression"] = lr_model
    print(lr_metrics)

    print("\n[3/5] Training Random Forest")
    rf_model = train_random_forest(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    rf_metrics = evaluate("RandomForest", y_test, rf_preds)
    results.append(rf_metrics)
    models["RandomForest"] = rf_model
    print(rf_metrics)

    print("\n[4/5] Results Summary")
    print("-" * 60)
    print(f"{'Model':<20} {'MAE':>10} {'RMSE':>10} {'R2':>10}")
    print("-" * 60)
    for r in sorted(results, key=lambda x: x["RMSE"]):
        print(f"{r['Model']:<20} {r['MAE']:>10.4f} {r['RMSE']:>10.4f} {r['R2']:>10.4f}")
    print("-" * 60)

    save_best(results, models)

    print("\n[5/5] Recursive forecast from first sprouting day")
    forecast_df = forecast_from_first_sprout(df)

    if forecast_df.empty:
        print("No forecast rows created.")
        return

    print("\nActual vs Predicted from sprouting stage:")
    print("-" * 90)
    print(
        f"{'Day':<6} {'Date':<12} {'Temp':<8} {'Stage':<8} "
        f"{'Actual':<12} {'Predicted':<12} {'Abs Error':<10}"
    )
    print("-" * 90)

    for _, row in forecast_df.iterrows():
        print(
            f"{int(row['day']):<6} "
            f"{str(row['date']):<12} "
            f"{row['temperature']:<8.2f} "
            f"{int(row['stage_label']):<8} "
            f"{row['actual_height']:<12.4f} "
            f"{row['predicted_height']:<12.4f} "
            f"{row['absolute_error']:<10.4f}"
        )

    forecast_mae = forecast_df["absolute_error"].mean()
    forecast_rmse = math.sqrt(np.mean((forecast_df["actual_height"] - forecast_df["predicted_height"]) ** 2))

    print("-" * 90)
    print(f"Forecast MAE : {forecast_mae:.4f}")
    print(f"Forecast RMSE: {forecast_rmse:.4f}")

    output_path = "output/predictions/sprout_forecast.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    forecast_df.to_csv(output_path, index=False)
    print(f"\nSaved sprout forecast to: {output_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
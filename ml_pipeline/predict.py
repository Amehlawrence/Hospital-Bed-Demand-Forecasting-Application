import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from config import MODEL_SAVE_PATH, FORECAST_START_DATE


def safe_filename(text):
    """Convert text to safe filename (replace spaces with underscores)"""
    return str(text).replace(" ", "_").replace("/", "_").replace("\\", "_")


def load_trained_model(hospital, ward):
    """Load trained model for a ward"""
    safe_hospital = safe_filename(hospital)
    safe_ward = safe_filename(ward)

    model_filename = f"{safe_hospital}_{safe_ward}_model.pkl"
    model_path = os.path.join(MODEL_SAVE_PATH, model_filename)

    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return None, None

    try:
        model = joblib.load(model_path)

        metrics_filename = f"{safe_hospital}_{safe_ward}_metrics.pkl"
        metrics_path = os.path.join(MODEL_SAVE_PATH, metrics_filename)

        if os.path.exists(metrics_path):
            metrics = joblib.load(metrics_path)
        else:
            metrics = {"mape": np.nan, "mae": np.nan}

        return model, metrics

    except Exception as e:
        print(f"Error loading model for {hospital}-{ward}: {e}")
        return None, None


def generate_forecast(hospital, ward, ts, forecast_ahead_days=14):
    """Generate forecast using trained model"""
    model, metrics = load_trained_model(hospital, ward)

    if model is None:
        print(f"No trained model available for {hospital}-{ward}")
        return None, np.nan, np.nan

    try:
        current_date = datetime.now().date()

        # If current_date is before FORECAST_START_DATE, don't go negative.
        if current_date < FORECAST_START_DATE:
            days_to_today = 0
        else:
            days_to_today = (current_date - FORECAST_START_DATE).days + 1

        total_forecast_days = days_to_today + forecast_ahead_days
        total_forecast_days = max(total_forecast_days, forecast_ahead_days)

        forecast = model.get_forecast(steps=total_forecast_days)
        forecast_df = forecast.summary_frame()

        forecast_index = pd.date_range(
            start=pd.Timestamp(FORECAST_START_DATE),
            periods=total_forecast_days,
            freq="D",
        )
        forecast_df.index = forecast_index

        mape = metrics.get("mape", np.nan)
        mae = metrics.get("mae", np.nan)

        return forecast_df, mape, mae

    except Exception as e:
        print(f"Error generating forecast for {hospital}-{ward}: {e}")
        return None, np.nan, np.nan

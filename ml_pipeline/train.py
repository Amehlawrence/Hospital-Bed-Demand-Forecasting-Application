import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error
import warnings
from database import load_all_data

warnings.filterwarnings("ignore")

from config import MODEL_SAVE_PATH


def safe_filename(text):
    """convert text to safe filename (replace spaces with underscores)"""
    return str(text).replace(" ", "_").replace("/", "_").replace("\\", "_")


def calculate_mape(actual, predicted):
    """calculate MAPE"""
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    mask = actual != 0
    if mask.sum() == 0:
        return np.nan

    actual_filtered = actual[mask]
    predicted_filtered = predicted[mask]
    ape = np.abs(actual_filtered - predicted_filtered) / actual_filtered
    return np.mean(ape) * 100


def train_model_for_ward(hospital, ward, ts):
    """Train SARIMAX model for a ward"""
    try:
        model = SARIMAX(
            ts,
            order=(2, 1, 2),
            seasonal_order=(1, 1, 1, 7),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )

        results = model.fit(disp=False, maxiter=100)

        # Test on last 14 days
        test_days = min(14, len(ts))
        if test_days > 3:
            test = ts.iloc[-test_days:]
            test_predictions = []
            train_series = ts.iloc[:-test_days].copy()

            for i in range(test_days):
                temp_model = SARIMAX(
                    train_series,
                    order=(2, 1, 2),
                    seasonal_order=(1, 1, 1, 7),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )

                temp_results = temp_model.fit(disp=False, maxiter=50)
                forecast_step = temp_results.get_forecast(steps=1)
                test_predictions.append(forecast_step.predicted_mean.iloc[0])

                if i < test_days - 1:
                    train_series = pd.concat([train_series, test.iloc[[i]]])

            test_predictions = pd.Series(test_predictions, index=test.index)
            mae = mean_absolute_error(test, test_predictions)
            mape = calculate_mape(test.values, test_predictions.values)
        else:
            mae = np.nan
            mape = np.nan

        # Create safe filenames
        safe_hospital = safe_filename(hospital)
        safe_ward = safe_filename(ward)

        # Save model
        os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
        model_filename = f"{safe_hospital}_{safe_ward}_model.pkl"
        model_path = os.path.join(MODEL_SAVE_PATH, model_filename)
        joblib.dump(results, model_path)

        # Save metrics
        metrics = {
            "hospital": hospital,
            "ward": ward,
            "mape": float(mape) if pd.notna(mape) else np.nan,
            "mae": float(mae) if pd.notna(mae) else np.nan,
            "last_trained": datetime.now().isoformat(),
            "data_points": len(ts),
        }

        metrics_filename = f"{safe_hospital}_{safe_ward}_metrics.pkl"
        metrics_path = os.path.join(MODEL_SAVE_PATH, metrics_filename)
        joblib.dump(metrics, metrics_path)

        print(f"Saved model to: {model_path}")
        return True, metrics

    except Exception as e:
        print(f"Error training model for {hospital}-{ward}: {e}")
        return False, None


def should_retrain_model(hospital, ward, retrain_threshold_days=14):
    """Check if model needs retraining"""
    safe_hospital = safe_filename(hospital)
    safe_ward = safe_filename(ward)
    metrics_filename = f"{safe_hospital}_{safe_ward}_metrics.pkl"
    metrics_path = os.path.join(MODEL_SAVE_PATH, metrics_filename)

    if not os.path.exists(metrics_path):
        print(f"No existing model found for {hospital}-{ward}")
        return True

    try:
        metrics = joblib.load(metrics_path)

        if "last_trained" not in metrics:
            print("Model exists but missing 'last_trained' field")
            return True

        last_trained = datetime.fromisoformat(metrics["last_trained"])
        days_since_training = (datetime.now() - last_trained).days

        if days_since_training >= retrain_threshold_days:
            print(
                f"Model is {days_since_training} days old "
                f"(≥ {retrain_threshold_days} threshold)"
            )
            return True
        else:
            print(
                f"Model is {days_since_training} days old "
                f"(< {retrain_threshold_days} threshold)"
            )
            return False

    except Exception as e:
        print(f"Error checking model for {hospital}-{ward}: {e}")
        return True


def train_all_models(retrain_all=False, min_data_days=28):
    """
    Main function to train all models for all hospitals and wards.

    Args:
        retrain_all: If True, retrain all models regardless of last training date
        min_data_days: Minimum number of days of data required for training

    Returns:
        int: Number of models successfully trained
    """
    trained_count = 0

    try:
        print("\n" + "=" * 60)
        print(f"STARTING MODEL TRAINING - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)

        df = load_all_data()

        if df is None or df.empty:
            print("ERROR: No data loaded")
            return 0

        print(
            f"Loaded {len(df)} records from {df['datetime'].min().date()} "
            f"to {df['datetime'].max().date()}"
        )

        combinations = df[["hospital", "ward"]].drop_duplicates()
        total_combinations = len(combinations)

        print(f"Found {total_combinations} unique hospital-ward combinations")

        if total_combinations == 0:
            print("No hospital-ward combinations found")
            return 0

        if not retrain_all:
            needs_retraining = 0
            for hospital, ward in combinations.itertuples(index=False, name=None):
                if should_retrain_model(hospital, ward):
                    needs_retraining += 1
            print(f"{needs_retraining} out of {total_combinations} need retraining")
        else:
            print("Retraining all models (forced)")

        for idx, (hospital, ward) in enumerate(
            combinations.itertuples(index=False, name=None), 1
        ):
            print(f"\n[{idx:3d}/{total_combinations:3d}] {hospital:<20} {ward:<20}", end="")

            if not retrain_all and not should_retrain_model(hospital, ward):
                print(" recently trained")
                continue

            ward_df = df[(df["hospital"] == hospital) & (df["ward"] == ward)].copy()
            ward_df = ward_df.sort_values("datetime")

            if len(ward_df) < min_data_days:
                print(f" Insufficient data ({len(ward_df)} rows < {min_data_days} minimum)")
                continue

            start_date = ward_df["datetime"].min().strftime("%Y-%m-%d")
            end_date = ward_df["datetime"].max().strftime("%Y-%m-%d")
            total_days = (ward_df["datetime"].max() - ward_df["datetime"].min()).days + 1

            print(f" Training on {len(ward_df)} records, {total_days} days: {start_date} to {end_date}")

            if len(ward_df) > total_days:
                ward_df = ward_df.set_index("datetime")
                daily_ts = ward_df["occupied_beds"].resample("D").mean()
            else:
                daily_ts = ward_df.set_index("datetime")["occupied_beds"]

            daily_ts = daily_ts.ffill().bfill()

            if len(daily_ts) < min_data_days:
                print(f" Not enough data after cleaning ({len(daily_ts)} days)")
                continue

            success, metrics = train_model_for_ward(hospital, ward, daily_ts)

            if success:
                trained_count += 1
                if metrics and pd.notna(metrics["mape"]):
                    print(f" Trained (MAPE: {metrics['mape']:.1f}%, MAE: {metrics['mae']:.2f})")
                else:
                    print(" Trained (metrics not available)")
            else:
                print(" Failed")

        print(f"\n{'='*60}")
        print("TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"Total combinations: {total_combinations}")
        print(f"Models trained: {trained_count}")
        print(f"Models skipped: {total_combinations - trained_count}")
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")

        return trained_count

    except Exception as e:
        print(f"ERROR in train_all_models: {str(e)}")
        import traceback

        traceback.print_exc()
        return trained_count

import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv(override=True)

# Dates
LAST_HISTORICAL_DATE = datetime(2025, 12, 31).date()
FORECAST_START_DATE = datetime(2026, 1, 1).date()

# Model settings
MODEL_SAVE_PATH = "ml_pipeline/models/"
MODEL_TRAINING_INTERVAL_DAYS = 14  # Train every 2 weeks
FORECAST_HORIZON_DAYS = 14

# Cache settings
CACHE_DURATION = 3600  # 1 hour

# Database
DATABASE_URL = os.getenv("DATABASE_URL")

import os
import pandas as pd
import psycopg2
from config import DATABASE_URL, LAST_HISTORICAL_DATE

# -------------------------
# Path setup (works in Docker + Windows)
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# CSV location
CSV_PATH = os.path.join(BASE_DIR, "Dataset", "bed_inventory.csv")

# Default to CSV unless you explicitly set DATA_SOURCE=db
DATA_SOURCE = os.getenv("DATA_SOURCE", "csv").lower()


def get_db_connection():
    """Get database connection (only used when DATA_SOURCE='db')."""
    return psycopg2.connect(DATABASE_URL)


def load_all_data():
    """
    Load all data from CSV or database depending on DATA_SOURCE.
    DATA_SOURCE='csv' -> reads from Dataset/bed_inventory.csv
    DATA_SOURCE='db'  -> reads from PostgreSQL
    """
    if DATA_SOURCE == "csv":
        if not os.path.exists(CSV_PATH):
            raise FileNotFoundError(
                f"CSV file not found at {CSV_PATH}. "
                "Check Dataset folder or update CSV_PATH in database.py."
            )
        df = pd.read_csv(CSV_PATH)

    elif DATA_SOURCE == "db":
        conn = get_db_connection()
        query = "SELECT * FROM bed_inventory"
        df = pd.read_sql(query, conn)
        conn.close()

    else:
        raise ValueError("DATA_SOURCE must be 'csv' or 'db'")

    # Filter data up to last historical date
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"])
    df = df[df["datetime"].dt.date <= LAST_HISTORICAL_DATE]

    return df


def get_hospitals():
    df = load_all_data()
    return sorted(df["hospital"].dropna().unique().tolist())


def get_hospital_wards(hospital_name):
    df = load_all_data()
    wards = df.loc[df["hospital"] == hospital_name, "ward"]
    return sorted(wards.dropna().unique().tolist())

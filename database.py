import pandas as pd
import psycopg2
from config import DATABASE_URL, LAST_HISTORICAL_DATE


def get_db_connection():
    """Get database connection"""
    return psycopg2.connect(DATABASE_URL)


def load_all_data():
    """Load all data from database"""
    conn = get_db_connection()
    query = "SELECT * FROM bed_inventory"
    df = pd.read_sql(query, conn)
    conn.close()

    # Filter data up to last historical date
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df[df["datetime"].dt.date <= LAST_HISTORICAL_DATE]

    return df


def get_hospitals():
    """Get list of hospitals"""
    conn = get_db_connection()
    query = "SELECT DISTINCT hospital FROM bed_inventory ORDER BY hospital"
    df = pd.read_sql(query, conn)
    conn.close()
    return df["hospital"].tolist()


def get_hospital_wards(hospital_name):
    """Get wards for a hospital"""
    conn = get_db_connection()
    query = """
        SELECT DISTINCT ward
        FROM bed_inventory
        WHERE hospital = %s
        ORDER BY ward
    """
    df = pd.read_sql(query, conn, params=(hospital_name,))
    conn.close()
    return df["ward"].tolist()

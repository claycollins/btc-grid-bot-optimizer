"""
Candle Database Module
======================
PostgreSQL storage for candle data caching.
"""

import os
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime
import pandas as pd

# Get database URL from environment (Railway provides this)
DATABASE_URL = os.environ.get('DATABASE_URL')


def get_connection():
    """Get a database connection."""
    if not DATABASE_URL:
        return None
    return psycopg2.connect(DATABASE_URL)


def init_db():
    """Initialize the database schema."""
    if not DATABASE_URL:
        print("[DB] No DATABASE_URL configured, skipping DB init")
        return False

    # Log connection attempt (mask password)
    masked_url = DATABASE_URL.split('@')[-1] if '@' in DATABASE_URL else 'configured'
    print(f"[DB] Attempting to connect to: ...@{masked_url}")

    try:
        conn = get_connection()
        if not conn:
            print("[DB] Failed to get connection")
            return False

        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS candles (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    open DOUBLE PRECISION NOT NULL,
                    high DOUBLE PRECISION NOT NULL,
                    low DOUBLE PRECISION NOT NULL,
                    close DOUBLE PRECISION NOT NULL,
                    volume DOUBLE PRECISION NOT NULL,
                    UNIQUE(symbol, timestamp)
                );
                CREATE INDEX IF NOT EXISTS idx_candles_symbol_timestamp
                ON candles(symbol, timestamp);
            """)
            conn.commit()

            # Verify table was created and get row count
            cur.execute("SELECT COUNT(*) FROM candles")
            count = cur.fetchone()[0]
            print(f"[DB] Database initialized successfully. Candles table has {count:,} rows")

        return True
    except Exception as e:
        print(f"[DB] Database init error: {e}")
        return False
    finally:
        if conn:
            conn.close()


def get_cached_candles(symbol: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
    """
    Get cached candles from the database.
    Returns a DataFrame with the cached candles, or empty DataFrame if none found.
    """
    if not DATABASE_URL:
        return pd.DataFrame()

    conn = get_connection()
    if not conn:
        return pd.DataFrame()

    try:
        query = """
            SELECT timestamp, open, high, low, close, volume
            FROM candles
            WHERE symbol = %s AND timestamp >= %s AND timestamp <= %s
            ORDER BY timestamp ASC
        """
        df = pd.read_sql_query(query, conn, params=(symbol, start_time, end_time))
        return df
    except Exception as e:
        print(f"Error fetching cached candles: {e}")
        return pd.DataFrame()
    finally:
        conn.close()


def save_candles(symbol: str, df: pd.DataFrame):
    """
    Save candles to the database.
    Uses upsert to handle duplicates gracefully.
    """
    if not DATABASE_URL or df.empty:
        return False

    conn = get_connection()
    if not conn:
        return False

    try:
        with conn.cursor() as cur:
            # Prepare data for bulk insert
            records = [
                (symbol, row['timestamp'], row['open'], row['high'],
                 row['low'], row['close'], row['volume'])
                for _, row in df.iterrows()
            ]

            # Upsert using ON CONFLICT
            execute_values(
                cur,
                """
                INSERT INTO candles (symbol, timestamp, open, high, low, close, volume)
                VALUES %s
                ON CONFLICT (symbol, timestamp) DO NOTHING
                """,
                records,
                page_size=1000
            )
        conn.commit()
        print(f"Saved {len(records)} candles to database")
        return True
    except Exception as e:
        print(f"Error saving candles: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()


def get_missing_ranges(symbol: str, start_time: datetime, end_time: datetime) -> list:
    """
    Determine which time ranges are missing from the cache.
    Returns a list of (start, end) tuples for missing ranges.
    """
    if not DATABASE_URL:
        return [(start_time, end_time)]

    conn = get_connection()
    if not conn:
        return [(start_time, end_time)]

    try:
        with conn.cursor() as cur:
            # Get min and max timestamps we have cached in the range
            cur.execute("""
                SELECT MIN(timestamp), MAX(timestamp), COUNT(*)
                FROM candles
                WHERE symbol = %s AND timestamp >= %s AND timestamp <= %s
            """, (symbol, start_time, end_time))

            result = cur.fetchone()
            cached_min, cached_max, count = result

            if count == 0:
                # No cached data at all
                return [(start_time, end_time)]

            missing_ranges = []

            # Check if we need data before our cached range
            if cached_min > start_time:
                missing_ranges.append((start_time, cached_min))

            # Check if we need data after our cached range
            if cached_max < end_time:
                missing_ranges.append((cached_max, end_time))

            return missing_ranges

    except Exception as e:
        print(f"Error checking missing ranges: {e}")
        return [(start_time, end_time)]
    finally:
        conn.close()


def get_candles_for_download(symbol: str, start_time: datetime, end_time: datetime) -> list:
    """
    Get candles for CSV download.
    Returns a list of dicts ready for CSV export.
    """
    if not DATABASE_URL:
        return []

    conn = get_connection()
    if not conn:
        return []

    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT timestamp, open, high, low, close, volume
                FROM candles
                WHERE symbol = %s AND timestamp >= %s AND timestamp <= %s
                ORDER BY timestamp ASC
            """, (symbol, start_time, end_time))

            columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            rows = cur.fetchall()

            return [
                {col: (str(val) if col == 'timestamp' else val)
                 for col, val in zip(columns, row)}
                for row in rows
            ]
    except Exception as e:
        print(f"Error fetching candles for download: {e}")
        return []
    finally:
        conn.close()

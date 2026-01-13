#!/usr/bin/env python3
"""
Candle Data Cache Module
========================
SQLite-based caching for historical candle data to avoid redundant API calls.
"""

import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Tuple
import os

# Database file path
DB_PATH = os.environ.get('CANDLE_CACHE_DB', 'candle_cache.db')


def get_connection():
    """Get a database connection with row factory."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize the database schema."""
    conn = get_connection()
    cursor = conn.cursor()

    # Create candles table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS candles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            interval TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume REAL NOT NULL,
            UNIQUE(symbol, interval, timestamp)
        )
    ''')

    # Create index for faster queries
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_candles_symbol_interval_timestamp
        ON candles(symbol, interval, timestamp)
    ''')

    # Create metadata table to track what ranges we have cached
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS cache_metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            interval TEXT NOT NULL,
            start_timestamp INTEGER NOT NULL,
            end_timestamp INTEGER NOT NULL,
            fetched_at TEXT NOT NULL,
            UNIQUE(symbol, interval)
        )
    ''')

    conn.commit()
    conn.close()


def get_cached_range(symbol: str, interval: str) -> Optional[Tuple[int, int]]:
    """
    Get the cached timestamp range for a symbol/interval pair.

    Returns:
        Tuple of (start_timestamp, end_timestamp) in milliseconds, or None if not cached
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute('''
        SELECT start_timestamp, end_timestamp
        FROM cache_metadata
        WHERE symbol = ? AND interval = ?
    ''', (symbol, interval))

    row = cursor.fetchone()
    conn.close()

    if row:
        return (row['start_timestamp'], row['end_timestamp'])
    return None


def get_cached_candles(
    symbol: str,
    interval: str,
    start_time: int,
    end_time: int
) -> pd.DataFrame:
    """
    Retrieve cached candles from the database.

    Parameters:
        symbol: Trading pair symbol
        interval: Candle interval (e.g., '1m')
        start_time: Start timestamp in milliseconds
        end_time: End timestamp in milliseconds

    Returns:
        DataFrame with candle data
    """
    conn = get_connection()

    query = '''
        SELECT timestamp, open, high, low, close, volume
        FROM candles
        WHERE symbol = ? AND interval = ?
        AND timestamp >= ? AND timestamp <= ?
        ORDER BY timestamp ASC
    '''

    df = pd.read_sql_query(
        query,
        conn,
        params=(symbol, interval, start_time, end_time)
    )

    conn.close()

    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    return df


def save_candles(
    symbol: str,
    interval: str,
    df: pd.DataFrame,
    start_time: int,
    end_time: int
):
    """
    Save candles to the database cache.

    Parameters:
        symbol: Trading pair symbol
        interval: Candle interval
        df: DataFrame with columns [timestamp, open, high, low, close, volume]
        start_time: Start timestamp of the fetched range (ms)
        end_time: End timestamp of the fetched range (ms)
    """
    if df.empty:
        return

    conn = get_connection()
    cursor = conn.cursor()

    # Convert timestamp to milliseconds if it's datetime
    df_copy = df.copy()
    if pd.api.types.is_datetime64_any_dtype(df_copy['timestamp']):
        df_copy['timestamp'] = df_copy['timestamp'].astype('int64') // 10**6

    # Insert candles (ignore duplicates)
    for _, row in df_copy.iterrows():
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO candles
                (symbol, interval, timestamp, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                symbol,
                interval,
                int(row['timestamp']),
                float(row['open']),
                float(row['high']),
                float(row['low']),
                float(row['close']),
                float(row['volume'])
            ))
        except Exception as e:
            print(f"Error inserting candle: {e}")
            continue

    # Update metadata
    existing_range = get_cached_range(symbol, interval)

    if existing_range:
        # Extend the existing range
        new_start = min(existing_range[0], start_time)
        new_end = max(existing_range[1], end_time)
        cursor.execute('''
            UPDATE cache_metadata
            SET start_timestamp = ?, end_timestamp = ?, fetched_at = ?
            WHERE symbol = ? AND interval = ?
        ''', (new_start, new_end, datetime.now().isoformat(), symbol, interval))
    else:
        # Create new metadata entry
        cursor.execute('''
            INSERT INTO cache_metadata
            (symbol, interval, start_timestamp, end_timestamp, fetched_at)
            VALUES (?, ?, ?, ?, ?)
        ''', (symbol, interval, start_time, end_time, datetime.now().isoformat()))

    conn.commit()
    conn.close()


def get_missing_ranges(
    symbol: str,
    interval: str,
    requested_start: int,
    requested_end: int
) -> list:
    """
    Determine which time ranges need to be fetched from the API.

    Returns:
        List of (start, end) tuples representing missing ranges
    """
    cached_range = get_cached_range(symbol, interval)

    if cached_range is None:
        # No cache, need to fetch everything
        return [(requested_start, requested_end)]

    cached_start, cached_end = cached_range
    missing_ranges = []

    # Check if we need data before the cached range
    if requested_start < cached_start:
        missing_ranges.append((requested_start, cached_start - 1))

    # Check if we need data after the cached range
    if requested_end > cached_end:
        missing_ranges.append((cached_end + 1, requested_end))

    return missing_ranges


def get_cache_stats() -> dict:
    """Get statistics about the cache."""
    conn = get_connection()
    cursor = conn.cursor()

    # Count total candles
    cursor.execute('SELECT COUNT(*) as count FROM candles')
    total_candles = cursor.fetchone()['count']

    # Get symbols cached
    cursor.execute('''
        SELECT symbol, interval, start_timestamp, end_timestamp, fetched_at,
        (SELECT COUNT(*) FROM candles c WHERE c.symbol = m.symbol AND c.interval = m.interval) as candle_count
        FROM cache_metadata m
    ''')

    symbols = []
    for row in cursor.fetchall():
        symbols.append({
            'symbol': row['symbol'],
            'interval': row['interval'],
            'start': datetime.fromtimestamp(row['start_timestamp'] / 1000).isoformat(),
            'end': datetime.fromtimestamp(row['end_timestamp'] / 1000).isoformat(),
            'fetched_at': row['fetched_at'],
            'candle_count': row['candle_count']
        })

    # Get database file size
    db_size = os.path.getsize(DB_PATH) if os.path.exists(DB_PATH) else 0

    conn.close()

    return {
        'total_candles': total_candles,
        'symbols': symbols,
        'db_size_mb': round(db_size / (1024 * 1024), 2)
    }


def clear_cache(symbol: str = None):
    """Clear the cache, optionally for a specific symbol only."""
    conn = get_connection()
    cursor = conn.cursor()

    if symbol:
        cursor.execute('DELETE FROM candles WHERE symbol = ?', (symbol,))
        cursor.execute('DELETE FROM cache_metadata WHERE symbol = ?', (symbol,))
    else:
        cursor.execute('DELETE FROM candles')
        cursor.execute('DELETE FROM cache_metadata')

    conn.commit()
    conn.close()


# Initialize database on module import
init_db()

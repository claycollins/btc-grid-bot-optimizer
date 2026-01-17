#!/usr/bin/env python3
"""
ASTeRDEX Long Grid Bot Optimizer
================================
Fetches historical 1-minute candle data from ASTeRDEX and runs a vectorized
backtest to find the optimal number of grids for a Long-Only Grid Bot strategy.

Author: Quantitative Trading Bot
Version: 1.0.0
"""

import numpy as np
import pandas as pd
import requests
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings

# Import database module for caching
try:
    import candle_db
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_URL = "https://fapi.asterdex.com"
KLINES_ENDPOINT = "/fapi/v1/klines"

# Default parameters
DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_INTERVAL = "1m"
DEFAULT_LOOKBACK_DAYS = 30
DEFAULT_CAPITAL = 10000
MAX_LIMIT_PER_REQUEST = 1500

# =============================================================================
# DATA FETCHING MODULE
# =============================================================================

def fetch_klines_batch(
    symbol: str,
    interval: str,
    start_time: int,
    end_time: int,
    limit: int = MAX_LIMIT_PER_REQUEST
) -> List[List]:
    """
    Fetch a single batch of kline data from ASTeRDEX API.

    Parameters:
    -----------
    symbol : str
        Trading pair symbol (e.g., 'BTCUSDT', 'WBTCUSDT')
    interval : str
        Candlestick interval (e.g., '1m', '5m', '1h')
    start_time : int
        Start timestamp in milliseconds
    end_time : int
        End timestamp in milliseconds
    limit : int
        Maximum number of klines to fetch (max 1500)

    Returns:
    --------
    List[List] : Raw kline data from API
    """
    url = f"{BASE_URL}{KLINES_ENDPOINT}"
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_time,
        "endTime": end_time,
        "limit": limit
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] API request failed: {e}")
        return []


def get_available_symbols() -> List[str]:
    """
    Fetch list of available trading symbols from ASTeRDEX.

    Returns:
    --------
    List[str] : List of available trading pair symbols
    """
    url = f"{BASE_URL}/fapi/v1/exchangeInfo"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        symbols = [s['symbol'] for s in data.get('symbols', []) if s.get('status') == 'TRADING']
        return sorted(symbols)
    except Exception as e:
        # Fallback to common symbols if API fails
        return ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT', 'ADAUSDT']


def fetch_historical_klines(
    symbol: str = DEFAULT_SYMBOL,
    interval: str = DEFAULT_INTERVAL,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    verbose: bool = True,
    progress_callback=None
) -> pd.DataFrame:
    """
    Fetch historical kline data from ASTeRDEX with pagination support.

    Parameters:
    -----------
    symbol : str
        Trading pair symbol
    interval : str
        Candlestick interval
    lookback_days : int
        Number of days to look back
    verbose : bool
        Print progress information
    progress_callback : callable, optional
        Callback function for progress updates (receives percent, message)

    Returns:
    --------
    pd.DataFrame : Cleaned OHLCV data with columns:
                   ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Fetching {symbol} {interval} data from ASTeRDEX...")
        print(f"Lookback period: {lookback_days} days")
        print(f"{'='*60}\n")

    # Calculate time range
    end_time = int(datetime.now().timestamp() * 1000)
    start_time = int((datetime.now() - timedelta(days=lookback_days)).timestamp() * 1000)

    # For 1-minute candles, each candle is 60000 ms
    interval_ms = 60000  # 1 minute in milliseconds
    total_candles_needed = lookback_days * 24 * 60

    if verbose:
        print(f"Expected candles: ~{total_candles_needed:,}")

    all_klines = []
    current_start = start_time
    batch_num = 0
    total_batches_estimate = max(1, (end_time - start_time) // (MAX_LIMIT_PER_REQUEST * interval_ms) + 1)

    while current_start < end_time:
        batch_num += 1
        batch_end = min(current_start + (MAX_LIMIT_PER_REQUEST * interval_ms), end_time)

        if verbose:
            print(f"  Batch {batch_num}: Fetching from {datetime.fromtimestamp(current_start/1000).strftime('%Y-%m-%d %H:%M')}...")

        if progress_callback:
            progress = min(40, int((batch_num / total_batches_estimate) * 40))
            progress_callback(progress, f"Fetching data batch {batch_num}...")

        klines = fetch_klines_batch(symbol, interval, current_start, batch_end)

        if not klines:
            if verbose:
                print(f"  [WARNING] Empty response for batch {batch_num}")
            break

        all_klines.extend(klines)

        # Move start time to after the last received candle
        if klines:
            last_open_time = klines[-1][0]
            current_start = last_open_time + interval_ms
        else:
            break

        # Rate limiting - be respectful to the API
        time.sleep(0.1)

    if not all_klines:
        print("[ERROR] No kline data received from API")
        return pd.DataFrame()

    if verbose:
        print(f"\nTotal raw klines fetched: {len(all_klines):,}")

    # Parse kline data into DataFrame
    # Response format: [open_time, open, high, low, close, volume, close_time,
    #                   quote_volume, trades, taker_buy_base, taker_buy_quote, ignore]
    df = pd.DataFrame(all_klines, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])

    # Clean and convert data types
    df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
    df['open'] = pd.to_numeric(df['open'], errors='coerce')
    df['high'] = pd.to_numeric(df['high'], errors='coerce')
    df['low'] = pd.to_numeric(df['low'], errors='coerce')
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce')

    # Keep only essential columns
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

    # Sort by time and remove duplicates
    df = df.sort_values('timestamp').drop_duplicates(subset='timestamp').reset_index(drop=True)

    # Drop any rows with NaN values
    initial_len = len(df)
    df = df.dropna()
    dropped = initial_len - len(df)

    if verbose:
        print(f"After cleaning: {len(df):,} candles")
        if dropped > 0:
            print(f"  (Dropped {dropped} rows with missing data)")
        print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"Price range: ${df['low'].min():,.2f} - ${df['high'].max():,.2f}")

    return df


def fetch_historical_klines_cached(
    symbol: str = DEFAULT_SYMBOL,
    interval: str = DEFAULT_INTERVAL,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    verbose: bool = True,
    progress_callback=None
) -> tuple:
    """
    Fetch historical kline data with PostgreSQL caching.
    Checks DB first, only fetches missing data from API.

    Parameters are same as fetch_historical_klines.

    Returns:
        tuple: (DataFrame, cache_stats dict)
        cache_stats contains: cached_count, api_count, total_count, cache_percent, source
    """
    cache_stats = {
        'cached_count': 0,
        'api_count': 0,
        'total_count': 0,
        'cache_percent': 0,
        'source': 'api'  # 'api', 'cache', 'mixed', or 'api_only'
    }

    if not DB_AVAILABLE or not candle_db.DATABASE_URL:
        # Fallback to direct API fetch if DB not available
        if verbose:
            print("[INFO] Database not configured, fetching from API...")
        df = fetch_historical_klines(symbol, interval, lookback_days, verbose, progress_callback)
        cache_stats['api_count'] = len(df)
        cache_stats['total_count'] = len(df)
        cache_stats['source'] = 'api_only'
        return df, cache_stats

    # Calculate time range
    end_time = datetime.now()
    start_time = end_time - timedelta(days=lookback_days)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Fetching {symbol} {interval} data (with caching)...")
        print(f"Lookback period: {lookback_days} days")
        print(f"{'='*60}\n")

    if progress_callback:
        progress_callback(5, "Checking cached data...")

    # Check what we have cached
    cached_df = candle_db.get_cached_candles(symbol, start_time, end_time)
    cached_count = len(cached_df)

    if verbose:
        print(f"Found {cached_count:,} cached candles")

    # Calculate expected candles
    expected_candles = lookback_days * 24 * 60

    # Always fetch recent data (last 2 hours) to ensure fresh candles
    # This ensures we get new candles even if cache is mostly full
    recent_cutoff = end_time - timedelta(hours=2)

    all_new_data = []
    api_fetched = 0

    # If we have most of the data cached (>95%), only fetch recent data
    if cached_count >= expected_candles * 0.95:
        if verbose:
            print(f"Cache is >95% full, fetching only recent data (last 2 hours)")
        if progress_callback:
            progress_callback(10, "Fetching recent candles...")

        # Fetch last 2 hours from API to get fresh data
        recent_df = fetch_historical_klines(
            symbol=symbol,
            interval=interval,
            lookback_days=1,  # Fetch 1 day to ensure we get the recent data
            verbose=False,
            progress_callback=None
        )

        if not recent_df.empty:
            # Filter to just the recent period
            recent_df = recent_df[recent_df['timestamp'] >= recent_cutoff]
            if not recent_df.empty:
                all_new_data.append(recent_df)
                api_fetched = len(recent_df)
                # Save new candles to database
                candle_db.save_candles(symbol, recent_df)
                if verbose:
                    print(f"Fetched {len(recent_df)} recent candles from API")
    else:
        # Find missing ranges and fetch from API
        missing_ranges = candle_db.get_missing_ranges(symbol, start_time, end_time)

        if verbose:
            print(f"Missing ranges: {len(missing_ranges)}")

        for i, (range_start, range_end) in enumerate(missing_ranges):
            range_days = (range_end - range_start).days + 1
            if verbose:
                print(f"Fetching missing range {i+1}: {range_start} to {range_end}")

            if progress_callback:
                progress_callback(10 + int((i / max(len(missing_ranges), 1)) * 30),
                                f"Fetching missing data ({i+1}/{len(missing_ranges)})...")

            # Fetch the missing range
            new_df = fetch_historical_klines(
                symbol=symbol,
                interval=interval,
                lookback_days=range_days,
                verbose=False,
                progress_callback=None
            )

            if not new_df.empty:
                # Filter to just the range we need
                new_df = new_df[(new_df['timestamp'] >= range_start) &
                               (new_df['timestamp'] <= range_end)]
                all_new_data.append(new_df)
                api_fetched += len(new_df)

                # Save to database
                candle_db.save_candles(symbol, new_df)

    # Combine cached and new data
    if all_new_data:
        new_combined = pd.concat(all_new_data, ignore_index=True)
        if cached_count > 0:
            final_df = pd.concat([cached_df, new_combined], ignore_index=True)
        else:
            final_df = new_combined
    else:
        final_df = cached_df

    # Clean up: sort and deduplicate
    final_df = final_df.sort_values('timestamp').drop_duplicates(
        subset='timestamp').reset_index(drop=True)

    if verbose:
        print(f"Total candles after merge: {len(final_df):,}")
        if len(final_df) > 0:
            print(f"Date range: {final_df['timestamp'].min()} to {final_df['timestamp'].max()}")

    # Update cache stats
    total = len(final_df)
    cache_stats['cached_count'] = cached_count
    cache_stats['api_count'] = api_fetched
    cache_stats['total_count'] = total
    cache_stats['cache_percent'] = round((cached_count / total * 100), 1) if total > 0 else 0
    cache_stats['source'] = 'cache' if api_fetched == 0 else ('mixed' if cached_count > 0 else 'api')

    return final_df, cache_stats


def get_current_price(symbol: str = DEFAULT_SYMBOL) -> float:
    """
    Get the current price for a symbol from ASTeRDEX.

    Parameters:
    -----------
    symbol : str
        Trading pair symbol

    Returns:
    --------
    float : Current price
    """
    url = f"{BASE_URL}/fapi/v1/ticker/price"
    params = {"symbol": symbol}

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        return float(data.get('price', 0))
    except Exception as e:
        print(f"[WARNING] Could not fetch current price: {e}")
        return 0.0


# =============================================================================
# BACKTEST ENGINE
# =============================================================================

def run_backtest_grids(
    df: pd.DataFrame,
    num_grids: int,
    lower: float,
    upper: float,
    capital: float = 10000
) -> Dict:
    """
    Run a vectorized backtest for a Long Grid Bot strategy.

    The strategy:
    - Divides the price range [lower, upper] into num_grids levels
    - Buys at each grid level when price drops to it
    - Sells when price rises by one grid spacing
    - Capital is equally distributed among all grid levels

    Parameters:
    -----------
    df : pd.DataFrame
        OHLCV data with columns ['open', 'high', 'low', 'close']
    num_grids : int
        Number of grid levels
    lower : float
        Lower price boundary
    upper : float
        Upper price boundary
    capital : float
        Total trading capital

    Returns:
    --------
    Dict : Backtest results including trades, profit, and ROI
    """
    if num_grids < 2:
        return {
            'num_grids': num_grids,
            'spacing': 0,
            'total_trades': 0,
            'total_profit': 0,
            'roi_percent': 0
        }

    # 1. Setup Grid Levels
    rng = upper - lower
    spacing = rng / num_grids

    # Buy levels are strictly lower, lower+spacing, ... up to N-1
    buy_levels = np.array([lower + i * spacing for i in range(num_grids)])
    sell_levels = buy_levels + spacing

    per_grid_capital = capital / num_grids

    # 2. State Initialization
    # Positions[i] = True means we hold inventory for grid level i
    positions = np.zeros(num_grids, dtype=bool)

    # If price starts below a grid, we assume we bought it immediately at open
    start_price = df.iloc[0]['open']
    positions[buy_levels < start_price] = True

    total_profit = 0.0
    trade_count = 0

    # Pre-calculate profit per successful swing for each level
    # Profit = Investment * (Sell - Buy) / Buy
    level_profits = per_grid_capital * (spacing / buy_levels)

    # 3. Vectorized Simulation Loop
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values

    for i in range(len(df)):
        O, H, L, C = opens[i], highs[i], lows[i], closes[i]

        # Determine intra-candle path (High/Low processing order) based on candle color
        if C >= O:
            # Green Candle: Open -> Low -> High -> Close
            # A. Check Buys at Low
            buy_mask = (~positions) & (buy_levels >= L - 1e-9)
            positions[buy_mask] = True

            # B. Check Sells at High
            sell_mask = (positions) & (sell_levels <= H + 1e-9)
            if np.any(sell_mask):
                executed = np.sum(sell_mask)
                trade_count += executed
                total_profit += np.sum(level_profits[sell_mask])
                positions[sell_mask] = False

        else:
            # Red Candle: Open -> High -> Low -> Close
            # A. Check Sells at High
            sell_mask = (positions) & (sell_levels <= H + 1e-9)
            if np.any(sell_mask):
                executed = np.sum(sell_mask)
                trade_count += executed
                total_profit += np.sum(level_profits[sell_mask])
                positions[sell_mask] = False

            # B. Check Buys at Low
            buy_mask = (~positions) & (buy_levels >= L - 1e-9)
            positions[buy_mask] = True

    return {
        'num_grids': num_grids,
        'spacing': spacing,
        'total_trades': trade_count,
        'total_profit': total_profit,
        'roi_percent': (total_profit / capital) * 100
    }


def run_backtest_with_trades(
    df: pd.DataFrame,
    num_grids: int,
    lower: float,
    upper: float,
    capital: float = 10000
) -> Dict:
    """
    Run a detailed backtest that logs each individual trade with timestamps.
    Use this only for the optimal configuration (not during optimization).

    Returns:
    --------
    Dict : Backtest results including detailed trade_log
    """
    if num_grids < 2:
        return {
            'num_grids': num_grids,
            'spacing': 0,
            'total_trades': 0,
            'total_profit': 0,
            'roi_percent': 0,
            'trade_log': []
        }

    # Setup Grid Levels
    rng = upper - lower
    spacing = rng / num_grids

    buy_levels = np.array([lower + i * spacing for i in range(num_grids)])
    sell_levels = buy_levels + spacing

    per_grid_capital = capital / num_grids

    # State Initialization
    positions = np.zeros(num_grids, dtype=bool)

    # Track buy prices for each position to calculate profit on sell
    position_buy_prices = np.zeros(num_grids)

    # If price starts below a grid, we assume we bought it immediately at open
    start_price = df.iloc[0]['open']
    initial_positions = buy_levels < start_price
    positions[initial_positions] = True
    position_buy_prices[initial_positions] = buy_levels[initial_positions]

    total_profit = 0.0
    trade_count = 0
    trade_log = []

    # Pre-calculate profit per successful swing for each level
    level_profits = per_grid_capital * (spacing / buy_levels)

    # Get timestamps
    timestamps = df['timestamp'].values if 'timestamp' in df.columns else df.index.values
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values

    for i in range(len(df)):
        O, H, L, C = opens[i], highs[i], lows[i], closes[i]
        ts = timestamps[i]

        # Convert timestamp to string if it's a datetime
        if hasattr(ts, 'strftime'):
            ts_str = ts.strftime('%Y-%m-%d %H:%M')
        else:
            ts_str = str(ts)

        if C >= O:
            # Green Candle: Open -> Low -> High -> Close
            # A. Check Buys at Low
            buy_mask = (~positions) & (buy_levels >= L - 1e-9)
            for level_idx in np.where(buy_mask)[0]:
                positions[level_idx] = True
                position_buy_prices[level_idx] = buy_levels[level_idx]
                trade_log.append({
                    'timestamp': ts_str,
                    'type': 'BUY',
                    'level': int(level_idx + 1),
                    'price': round(float(buy_levels[level_idx]), 2),
                    'profit': None
                })

            # B. Check Sells at High
            sell_mask = (positions) & (sell_levels <= H + 1e-9)
            for level_idx in np.where(sell_mask)[0]:
                profit = float(level_profits[level_idx])
                trade_count += 1
                total_profit += profit
                positions[level_idx] = False
                trade_log.append({
                    'timestamp': ts_str,
                    'type': 'SELL',
                    'level': int(level_idx + 1),
                    'price': round(float(sell_levels[level_idx]), 2),
                    'profit': round(profit, 4)
                })

        else:
            # Red Candle: Open -> High -> Low -> Close
            # A. Check Sells at High
            sell_mask = (positions) & (sell_levels <= H + 1e-9)
            for level_idx in np.where(sell_mask)[0]:
                profit = float(level_profits[level_idx])
                trade_count += 1
                total_profit += profit
                positions[level_idx] = False
                trade_log.append({
                    'timestamp': ts_str,
                    'type': 'SELL',
                    'level': int(level_idx + 1),
                    'price': round(float(sell_levels[level_idx]), 2),
                    'profit': round(profit, 4)
                })

            # B. Check Buys at Low
            buy_mask = (~positions) & (buy_levels >= L - 1e-9)
            for level_idx in np.where(buy_mask)[0]:
                positions[level_idx] = True
                position_buy_prices[level_idx] = buy_levels[level_idx]
                trade_log.append({
                    'timestamp': ts_str,
                    'type': 'BUY',
                    'level': int(level_idx + 1),
                    'price': round(float(buy_levels[level_idx]), 2),
                    'profit': None
                })

    return {
        'num_grids': num_grids,
        'spacing': spacing,
        'total_trades': trade_count,
        'total_profit': total_profit,
        'roi_percent': (total_profit / capital) * 100,
        'trade_log': trade_log
    }


def run_optimization(
    df: pd.DataFrame,
    lower: float,
    upper: float,
    capital: float = 10000,
    min_grids: int = 2,
    max_grids: int = 200,
    verbose: bool = True,
    progress_callback=None
) -> pd.DataFrame:
    """
    Run grid optimization across a range of grid counts.

    Parameters:
    -----------
    df : pd.DataFrame
        OHLCV data
    lower : float
        Lower price boundary
    upper : float
        Upper price boundary
    capital : float
        Total trading capital
    min_grids : int
        Minimum number of grids to test
    max_grids : int
        Maximum number of grids to test
    verbose : bool
        Print progress information
    progress_callback : callable, optional
        Callback function for progress updates (receives percent, message)

    Returns:
    --------
    pd.DataFrame : Results for each grid configuration
    """
    if verbose:
        print(f"\n{'='*60}")
        print("Running Grid Optimization Backtest")
        print(f"{'='*60}")
        print(f"Price Range: ${lower:,.2f} - ${upper:,.2f}")
        print(f"Range Width: ${upper - lower:,.2f} ({((upper - lower) / lower) * 100:.2f}%)")
        print(f"Capital: ${capital:,.2f}")
        print(f"Testing grids: {min_grids} to {max_grids}")
        print(f"Data points: {len(df):,} candles")
        print(f"{'='*60}\n")

    results = []
    total_grids = max_grids - min_grids + 1

    for num_grids in range(min_grids, max_grids + 1):
        result = run_backtest_grids(df, num_grids, lower, upper, capital)
        results.append(result)

        if verbose and num_grids % 20 == 0:
            print(f"  Progress: {num_grids}/{max_grids} grids tested...")

        if progress_callback:
            progress = 50 + int(((num_grids - min_grids + 1) / total_grids) * 50)
            progress_callback(progress, f"Testing {num_grids} grids...")

    results_df = pd.DataFrame(results)

    # Add additional metrics
    results_df['profit_per_trade'] = results_df['total_profit'] / results_df['total_trades'].replace(0, np.nan)
    results_df['trades_per_day'] = results_df['total_trades'] / (len(df) / 1440)  # 1440 minutes per day
    results_df['daily_roi'] = results_df['roi_percent'] / (len(df) / 1440)

    return results_df


def print_results(results_df: pd.DataFrame, top_n: int = 5) -> None:
    """
    Print the optimization results in a formatted manner.

    Parameters:
    -----------
    results_df : pd.DataFrame
        Optimization results
    top_n : int
        Number of top results to display
    """
    print(f"\n{'='*80}")
    print(f"{'OPTIMIZATION RESULTS':^80}")
    print(f"{'='*80}\n")

    # Sort by total profit (descending)
    sorted_df = results_df.sort_values('total_profit', ascending=False)

    print(f"{'TOP ' + str(top_n) + ' CONFIGURATIONS BY TOTAL PROFIT':^80}")
    print("-" * 80)
    print(f"{'Rank':<6}{'Grids':<8}{'Spacing':<14}{'Trades':<10}{'Profit ($)':<14}{'ROI (%)':<12}{'$/Trade':<12}")
    print("-" * 80)

    for rank, (idx, row) in enumerate(sorted_df.head(top_n).iterrows(), 1):
        print(f"{rank:<6}{int(row['num_grids']):<8}${row['spacing']:>10,.2f}  {int(row['total_trades']):<10}"
              f"${row['total_profit']:>10,.2f}  {row['roi_percent']:>10.2f}%  "
              f"${row['profit_per_trade']:>8.4f}")

    print("-" * 80)

    # Highlight the optimal configuration
    optimal = sorted_df.iloc[0]

    print(f"\n{'*'*80}")
    print(f"{'OPTIMAL CONFIGURATION':^80}")
    print(f"{'*'*80}")
    print(f"""
    Number of Grids:     {int(optimal['num_grids'])}
    Grid Spacing:        ${optimal['spacing']:,.2f}
    Total Trades:        {int(optimal['total_trades']):,}
    Total Profit:        ${optimal['total_profit']:,.2f}
    ROI:                 {optimal['roi_percent']:.2f}%
    Profit per Trade:    ${optimal['profit_per_trade']:.4f}
    Trades per Day:      {optimal['trades_per_day']:.1f}
    Daily ROI:           {optimal['daily_roi']:.4f}%
    """)
    print(f"{'*'*80}\n")

    # Also show top by trade count (for comparison)
    print(f"\n{'ALTERNATIVE ANALYSIS - TOP 5 BY TRADE COUNT':^80}")
    print("-" * 80)
    by_trades = results_df.sort_values('total_trades', ascending=False).head(top_n)
    print(f"{'Rank':<6}{'Grids':<8}{'Spacing':<14}{'Trades':<10}{'Profit ($)':<14}{'ROI (%)':<12}")
    print("-" * 80)
    for rank, (idx, row) in enumerate(by_trades.iterrows(), 1):
        print(f"{rank:<6}{int(row['num_grids']):<8}${row['spacing']:>10,.2f}  {int(row['total_trades']):<10}"
              f"${row['total_profit']:>10,.2f}  {row['roi_percent']:>10.2f}%")
    print("-" * 80)


def get_user_input(prompt: str, default: float) -> float:
    """
    Get user input with a default value.

    Parameters:
    -----------
    prompt : str
        Input prompt to display
    default : float
        Default value if user presses Enter

    Returns:
    --------
    float : User input or default value
    """
    try:
        user_input = input(f"{prompt} [default: {default:,.2f}]: ").strip()
        if user_input == "":
            return default
        return float(user_input)
    except ValueError:
        print(f"Invalid input, using default: {default}")
        return default


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main execution function for the Grid Bot Optimizer.
    """
    print("\n" + "=" * 80)
    print("ASTeRDEX LONG GRID BOT OPTIMIZER".center(80))
    print("=" * 80)
    print("\nThis tool finds the mathematically optimal number of grids for a")
    print("Long-Only Grid Bot strategy using historical 1-minute candle data.\n")

    # Get symbol from user
    symbol_input = input(f"Enter symbol [default: {DEFAULT_SYMBOL}]: ").strip().upper()
    symbol = symbol_input if symbol_input else DEFAULT_SYMBOL

    # Ensure symbol format (remove slash if present)
    symbol = symbol.replace("/", "")

    # Fetch current price for defaults
    print(f"\nFetching current price for {symbol}...")
    current_price = get_current_price(symbol)

    if current_price > 0:
        print(f"Current {symbol} price: ${current_price:,.2f}")
        default_lower = round(current_price * 0.9, 2)
        default_upper = round(current_price * 1.1, 2)
    else:
        print("[WARNING] Could not fetch current price. Using manual defaults.")
        # Fallback defaults for BTC
        default_lower = 90000.0
        default_upper = 110000.0

    # Get user inputs
    print("\n" + "-" * 40)
    print("Configure Grid Parameters")
    print("-" * 40)

    lower_limit = get_user_input("Enter LOWER price limit", default_lower)
    upper_limit = get_user_input("Enter UPPER price limit", default_upper)

    # Validate price range
    if lower_limit >= upper_limit:
        print("[ERROR] Lower limit must be less than upper limit!")
        return

    capital = get_user_input("Enter trading CAPITAL", DEFAULT_CAPITAL)

    lookback_input = input(f"Enter lookback period in days [default: {DEFAULT_LOOKBACK_DAYS}]: ").strip()
    lookback_days = int(lookback_input) if lookback_input else DEFAULT_LOOKBACK_DAYS

    # Fetch historical data
    df = fetch_historical_klines(
        symbol=symbol,
        interval=DEFAULT_INTERVAL,
        lookback_days=lookback_days,
        verbose=True
    )

    if df.empty:
        print("[ERROR] Failed to fetch historical data. Exiting.")
        return

    # Check if price range is valid for the data
    data_low = df['low'].min()
    data_high = df['high'].max()

    print(f"\n[INFO] Historical price range: ${data_low:,.2f} - ${data_high:,.2f}")

    if lower_limit > data_high or upper_limit < data_low:
        print("[WARNING] Your grid range is outside the historical price range!")
        print("          This may result in very few or no trades.")
        proceed = input("Continue anyway? (y/n): ").strip().lower()
        if proceed != 'y':
            return

    # Run optimization
    results_df = run_optimization(
        df=df,
        lower=lower_limit,
        upper=upper_limit,
        capital=capital,
        min_grids=2,
        max_grids=200,
        verbose=True
    )

    # Display results
    print_results(results_df, top_n=5)

    # Save results to CSV
    output_file = f"grid_optimization_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n[INFO] Full results saved to: {output_file}")

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS".center(80))
    print("=" * 80)
    print(f"""
    Symbol:              {symbol}
    Data Period:         {df['timestamp'].min().strftime('%Y-%m-%d')} to {df['timestamp'].max().strftime('%Y-%m-%d')}
    Total Candles:       {len(df):,}
    Grid Range:          ${lower_limit:,.2f} - ${upper_limit:,.2f}
    Capital:             ${capital:,.2f}

    Best Grid Count:     {int(results_df.loc[results_df['total_profit'].idxmax(), 'num_grids'])}
    Best Grid Spacing:   ${results_df.loc[results_df['total_profit'].idxmax(), 'spacing']:,.2f}
    Maximum Profit:      ${results_df['total_profit'].max():,.2f}
    Maximum ROI:         {results_df['roi_percent'].max():.2f}%
    """)
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

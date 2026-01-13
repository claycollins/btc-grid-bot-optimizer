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
    progress_callback=None,
    use_cache: bool = True
) -> pd.DataFrame:
    """
    Fetch historical kline data from ASTeRDEX with caching and pagination support.

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
    use_cache : bool
        Whether to use the local cache (default True)

    Returns:
    --------
    pd.DataFrame : Cleaned OHLCV data with columns:
                   ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    """
    # Import cache module
    try:
        import candle_cache as cache
    except ImportError:
        use_cache = False

    if verbose:
        print(f"\n{'='*60}")
        print(f"Fetching {symbol} {interval} data from ASTeRDEX...")
        print(f"Lookback period: {lookback_days} days")
        print(f"Cache: {'enabled' if use_cache else 'disabled'}")
        print(f"{'='*60}\n")

    # Calculate time range
    end_time = int(datetime.now().timestamp() * 1000)
    start_time = int((datetime.now() - timedelta(days=lookback_days)).timestamp() * 1000)

    # For 1-minute candles, each candle is 60000 ms
    interval_ms = 60000  # 1 minute in milliseconds

    # Check cache first
    cached_df = pd.DataFrame()
    missing_ranges = [(start_time, end_time)]

    if use_cache:
        if progress_callback:
            progress_callback(5, "Checking cache...")

        # Get what we already have cached
        cached_df = cache.get_cached_candles(symbol, interval, start_time, end_time)

        if not cached_df.empty:
            if verbose:
                print(f"Found {len(cached_df):,} cached candles")

        # Determine what ranges we need to fetch
        missing_ranges = cache.get_missing_ranges(symbol, interval, start_time, end_time)

        if not missing_ranges:
            if verbose:
                print("All data available from cache!")
            if progress_callback:
                progress_callback(40, "Loaded from cache")
            return cached_df

        if verbose:
            print(f"Need to fetch {len(missing_ranges)} range(s) from API")

    # Fetch missing ranges from API
    all_klines = []
    total_ranges = len(missing_ranges)

    for range_idx, (range_start, range_end) in enumerate(missing_ranges):
        current_start = range_start
        batch_num = 0
        total_batches_estimate = max(1, (range_end - range_start) // (MAX_LIMIT_PER_REQUEST * interval_ms) + 1)

        while current_start < range_end:
            batch_num += 1
            batch_end = min(current_start + (MAX_LIMIT_PER_REQUEST * interval_ms), range_end)

            if verbose:
                print(f"  Range {range_idx + 1}/{total_ranges}, Batch {batch_num}: Fetching from {datetime.fromtimestamp(current_start/1000).strftime('%Y-%m-%d %H:%M')}...")

            if progress_callback:
                overall_progress = (range_idx / total_ranges) + (batch_num / total_batches_estimate / total_ranges)
                progress = 5 + min(35, int(overall_progress * 35))
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

    # Parse newly fetched kline data
    new_df = pd.DataFrame()
    if all_klines:
        if verbose:
            print(f"\nTotal raw klines fetched from API: {len(all_klines):,}")

        # Response format: [open_time, open, high, low, close, volume, close_time,
        #                   quote_volume, trades, taker_buy_base, taker_buy_quote, ignore]
        new_df = pd.DataFrame(all_klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])

        # Clean and convert data types
        new_df['timestamp'] = pd.to_datetime(new_df['open_time'], unit='ms')
        new_df['open'] = pd.to_numeric(new_df['open'], errors='coerce')
        new_df['high'] = pd.to_numeric(new_df['high'], errors='coerce')
        new_df['low'] = pd.to_numeric(new_df['low'], errors='coerce')
        new_df['close'] = pd.to_numeric(new_df['close'], errors='coerce')
        new_df['volume'] = pd.to_numeric(new_df['volume'], errors='coerce')

        # Keep only essential columns
        new_df = new_df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

        # Save to cache
        if use_cache and not new_df.empty:
            if verbose:
                print("Saving to cache...")
            cache.save_candles(symbol, interval, new_df, start_time, end_time)

    # Combine cached and new data
    if not cached_df.empty and not new_df.empty:
        df = pd.concat([cached_df, new_df], ignore_index=True)
    elif not cached_df.empty:
        df = cached_df
    elif not new_df.empty:
        df = new_df
    else:
        if verbose:
            print("[ERROR] No kline data available")
        return pd.DataFrame()

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

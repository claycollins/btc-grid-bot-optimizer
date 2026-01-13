#!/usr/bin/env python3
"""
ASTeRDEX Grid Bot Optimizer - Web Application
==============================================
Flask backend serving REST API for the grid optimization engine.
"""

import os
import uuid
import threading
import io
import csv
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify, request, Response

from flask_cors import CORS

# Import the optimizer module
import asterdex_grid_optimizer as optimizer

# Import database module
try:
    import candle_db
    DB_AVAILABLE = True
    print("[APP] Database module loaded successfully")
except ImportError as e:
    DB_AVAILABLE = False
    print(f"[APP] Database module not available: {e}")

app = Flask(__name__)
CORS(app)

# Initialize database on startup
if DB_AVAILABLE:
    if candle_db.DATABASE_URL:
        print("[APP] DATABASE_URL is set, initializing database...")
        db_init_success = candle_db.init_db()
        if db_init_success:
            print("[APP] Database ready for caching")
        else:
            print("[APP] Database initialization failed - falling back to API-only mode")
    else:
        print("[APP] No DATABASE_URL found - running in API-only mode (no caching)")
else:
    print("[APP] Running without database support")

# In-memory job storage (for local use)
jobs = {}

# =============================================================================
# API ROUTES
# =============================================================================

@app.route('/')
def index():
    """Serve the main application page."""
    return render_template('index.html')


@app.route('/api/symbols', methods=['GET'])
def get_symbols():
    """Get list of available trading symbols."""
    try:
        symbols = optimizer.get_available_symbols()
        return jsonify({
            'success': True,
            'symbols': symbols
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/price/<symbol>', methods=['GET'])
def get_price(symbol):
    """Get current price for a symbol."""
    try:
        symbol = symbol.upper().replace('/', '')
        price = optimizer.get_current_price(symbol)

        if price > 0:
            return jsonify({
                'success': True,
                'symbol': symbol,
                'price': price,
                'default_lower': 85000,
                'default_upper': 95000
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Could not fetch price'
            }), 404
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/optimize', methods=['POST'])
def start_optimization():
    """Start a grid optimization job."""
    try:
        data = request.get_json()

        # Extract parameters
        symbol = data.get('symbol', 'BTCUSDT').upper().replace('/', '')
        lower_limit = float(data.get('lower_limit', 0))
        upper_limit = float(data.get('upper_limit', 0))
        capital = float(data.get('capital', 10000))
        lookback_days = int(data.get('lookback_days', 30))
        max_grids = int(data.get('max_grids', 200))

        # Validate inputs
        if lower_limit >= upper_limit:
            return jsonify({
                'success': False,
                'error': 'Lower limit must be less than upper limit'
            }), 400

        if capital <= 0:
            return jsonify({
                'success': False,
                'error': 'Capital must be positive'
            }), 400

        if lookback_days < 1 or lookback_days > 365:
            return jsonify({
                'success': False,
                'error': 'Lookback days must be between 1 and 365'
            }), 400

        if max_grids < 2 or max_grids > 500:
            return jsonify({
                'success': False,
                'error': 'Max grids must be between 2 and 500'
            }), 400

        # Create job
        job_id = str(uuid.uuid4())[:8]
        jobs[job_id] = {
            'status': 'running',
            'progress': 0,
            'message': 'Starting optimization...',
            'params': {
                'symbol': symbol,
                'lower_limit': lower_limit,
                'upper_limit': upper_limit,
                'capital': capital,
                'lookback_days': lookback_days,
                'max_grids': max_grids
            },
            'result': None,
            'error': None,
            'started_at': datetime.now().isoformat()
        }

        # Run optimization in background thread
        thread = threading.Thread(
            target=run_optimization_job,
            args=(job_id, symbol, lower_limit, upper_limit, capital, lookback_days, max_grids)
        )
        thread.daemon = True
        thread.start()

        return jsonify({
            'success': True,
            'job_id': job_id,
            'message': 'Optimization started'
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


def run_optimization_job(job_id, symbol, lower_limit, upper_limit, capital, lookback_days, max_grids):
    """Background worker for optimization job."""
    try:
        def progress_callback(percent, message):
            if job_id in jobs:
                jobs[job_id]['progress'] = percent
                jobs[job_id]['message'] = message

        # Fetch historical data (use cached version if DB available)
        progress_callback(5, 'Fetching historical data...')

        # Default cache stats for API-only mode
        cache_stats = {
            'cached_count': 0,
            'api_count': 0,
            'total_count': 0,
            'cache_percent': 0,
            'source': 'api_only'
        }

        if DB_AVAILABLE and candle_db.DATABASE_URL:
            df, cache_stats = optimizer.fetch_historical_klines_cached(
                symbol=symbol,
                interval='1m',
                lookback_days=lookback_days,
                verbose=False,
                progress_callback=progress_callback
            )
        else:
            df = optimizer.fetch_historical_klines(
                symbol=symbol,
                interval='1m',
                lookback_days=lookback_days,
                verbose=False,
                progress_callback=progress_callback
            )
            cache_stats['api_count'] = len(df)
            cache_stats['total_count'] = len(df)

        if df.empty:
            jobs[job_id]['status'] = 'failed'
            jobs[job_id]['error'] = 'Failed to fetch historical data'
            return

        # Get data info including cache stats
        data_info = {
            'total_candles': len(df),
            'date_start': df['timestamp'].min().strftime('%Y-%m-%d %H:%M'),
            'date_end': df['timestamp'].max().strftime('%Y-%m-%d %H:%M'),
            'price_low': float(df['low'].min()),
            'price_high': float(df['high'].max()),
            'cache_stats': cache_stats
        }

        progress_callback(45, 'Running optimization...')

        # Run optimization
        results_df = optimizer.run_optimization(
            df=df,
            lower=lower_limit,
            upper=upper_limit,
            capital=capital,
            min_grids=2,
            max_grids=max_grids,
            verbose=False,
            progress_callback=progress_callback
        )

        progress_callback(95, 'Processing results...')

        # Sort by profit and get top results
        results_df = results_df.sort_values('total_profit', ascending=False)

        # Get optimal configuration
        optimal = results_df.iloc[0]

        # Prepare response data
        all_results = results_df.to_dict('records')

        # Clean up NaN values
        for r in all_results:
            for k, v in r.items():
                if isinstance(v, float) and (v != v):  # NaN check
                    r[k] = 0

        jobs[job_id]['status'] = 'completed'
        jobs[job_id]['progress'] = 100
        jobs[job_id]['message'] = 'Optimization complete!'
        jobs[job_id]['result'] = {
            'data_info': data_info,
            'optimal': {
                'num_grids': int(optimal['num_grids']),
                'spacing': float(optimal['spacing']),
                'total_trades': int(optimal['total_trades']),
                'total_profit': float(optimal['total_profit']),
                'roi_percent': float(optimal['roi_percent']),
                'profit_per_trade': float(optimal['profit_per_trade']) if optimal['profit_per_trade'] == optimal['profit_per_trade'] else 0,
                'trades_per_day': float(optimal['trades_per_day']) if optimal['trades_per_day'] == optimal['trades_per_day'] else 0,
                'daily_roi': float(optimal['daily_roi']) if optimal['daily_roi'] == optimal['daily_roi'] else 0
            },
            'all_results': all_results,
            'top_results': all_results[:20],
            'candle_data': df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].assign(
                timestamp=df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            ).to_dict('records')
        }

    except Exception as e:
        jobs[job_id]['status'] = 'failed'
        jobs[job_id]['error'] = str(e)
        jobs[job_id]['progress'] = 0


@app.route('/api/status/<job_id>', methods=['GET'])
def get_job_status(job_id):
    """Get status of an optimization job."""
    if job_id not in jobs:
        return jsonify({
            'success': False,
            'error': 'Job not found'
        }), 404

    job = jobs[job_id]

    response = {
        'success': True,
        'job_id': job_id,
        'status': job['status'],
        'progress': job['progress'],
        'message': job['message']
    }

    if job['status'] == 'completed':
        response['result'] = job['result']
    elif job['status'] == 'failed':
        response['error'] = job['error']

    return jsonify(response)


@app.route('/api/download/candles', methods=['GET'])
def download_candles():
    """Download candle data as CSV from the database (legacy endpoint)."""
    symbol = request.args.get('symbol', 'BTCUSDT').upper().replace('/', '')
    lookback_days = int(request.args.get('lookback_days', 30))

    if not DB_AVAILABLE or not candle_db.DATABASE_URL:
        return jsonify({
            'success': False,
            'error': 'CSV download is now handled client-side. Please run an optimization and use the Download button on the results page.',
            'hint': 'If you are seeing this in the browser, please refresh the page to get the latest version.'
        }), 400

    # Calculate time range
    end_time = datetime.now()
    start_time = end_time - timedelta(days=lookback_days)

    # Get candles from database
    candles = candle_db.get_candles_for_download(symbol, start_time, end_time)

    if not candles:
        return jsonify({
            'success': False,
            'error': 'No candle data found. Please run an optimization first.'
        }), 404

    # Generate CSV
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    writer.writeheader()
    writer.writerows(candles)

    # Create response
    csv_content = output.getvalue()
    filename = f"candles_{symbol}_{datetime.now().strftime('%Y%m%d')}.csv"

    return Response(
        csv_content,
        mimetype='text/csv',
        headers={'Content-Disposition': f'attachment; filename={filename}'}
    )


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint with database diagnostics."""
    db_status = {
        'configured': bool(DB_AVAILABLE and candle_db.DATABASE_URL),
        'connected': False,
        'table_exists': False,
        'row_count': 0
    }

    if db_status['configured']:
        try:
            conn = candle_db.get_connection()
            if conn:
                db_status['connected'] = True
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables
                            WHERE table_name = 'candles'
                        )
                    """)
                    db_status['table_exists'] = cur.fetchone()[0]

                    if db_status['table_exists']:
                        cur.execute("SELECT COUNT(*) FROM candles")
                        db_status['row_count'] = cur.fetchone()[0]
                conn.close()
        except Exception as e:
            db_status['error'] = str(e)

    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'database': db_status
    })


@app.route('/api/db/init', methods=['POST'])
def init_database():
    """Manually initialize/reinitialize the database."""
    if not DB_AVAILABLE:
        return jsonify({
            'success': False,
            'error': 'Database module not available'
        }), 500

    if not candle_db.DATABASE_URL:
        # List environment variables that might contain DB URL (for debugging)
        db_vars = ['DATABASE_URL', 'DATABASE_PRIVATE_URL', 'POSTGRES_URL', 'POSTGRESQL_URL', 'PGHOST']
        found_vars = {var: bool(os.environ.get(var)) for var in db_vars}
        return jsonify({
            'success': False,
            'error': 'No database URL configured',
            'env_vars_checked': found_vars
        }), 500

    try:
        success = candle_db.init_db()
        if success:
            # Get row count
            conn = candle_db.get_connection()
            row_count = 0
            if conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) FROM candles")
                    row_count = cur.fetchone()[0]
                conn.close()

            return jsonify({
                'success': True,
                'message': 'Database initialized successfully',
                'row_count': row_count
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Database initialization failed - check server logs'
            }), 500
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    import sys
    # Use PORT env var for cloud deployment, fallback to 5001 for local
    port = int(os.environ.get('PORT', sys.argv[1] if len(sys.argv) > 1 else 5001))
    debug = os.environ.get('FLASK_DEBUG', 'true').lower() == 'true'

    print("\n" + "=" * 60)
    print("ASTeRDEX Grid Bot Optimizer - Web Interface")
    print("=" * 60)
    print(f"\nStarting server at http://localhost:{port}")
    print("Press Ctrl+C to stop\n")

    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug,
        threaded=True
    )

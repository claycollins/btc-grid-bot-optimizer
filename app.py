#!/usr/bin/env python3
"""
ASTeRDEX Grid Bot Optimizer - Web Application
==============================================
Flask backend serving REST API for the grid optimization engine.
"""

import os
import uuid
import threading
from datetime import datetime
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS

# Import the optimizer module
import asterdex_grid_optimizer as optimizer

app = Flask(__name__)
CORS(app)

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
                'default_lower': round(price * 0.9, 2),
                'default_upper': round(price * 1.1, 2)
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

        # Fetch historical data
        progress_callback(5, 'Fetching historical data...')
        df = optimizer.fetch_historical_klines(
            symbol=symbol,
            interval='1m',
            lookback_days=lookback_days,
            verbose=False,
            progress_callback=progress_callback
        )

        if df.empty:
            jobs[job_id]['status'] = 'failed'
            jobs[job_id]['error'] = 'Failed to fetch historical data'
            return

        # Get data info
        data_info = {
            'total_candles': len(df),
            'date_start': df['timestamp'].min().strftime('%Y-%m-%d %H:%M'),
            'date_end': df['timestamp'].max().strftime('%Y-%m-%d %H:%M'),
            'price_low': float(df['low'].min()),
            'price_high': float(df['high'].max())
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
            'top_results': all_results[:20]
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


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    })


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

/**
 * ASTeRDEX Grid Bot Optimizer - Frontend Application
 * ===================================================
 */

// =============================================================================
// STATE & CONFIGURATION
// =============================================================================

const state = {
    currentJobId: null,
    pollInterval: null,
    profitChart: null,
    tradesChart: null,
    allResults: [],
    candleData: [],             // Store candle data for CSV download
    tradeLog: [],               // Store trade log for display and download
    tradeLogFilter: 'all',      // Current trade log filter
    lastSymbol: 'BTCUSDT',      // Track last optimization params for CSV download
    lastLookbackDays: 30,
    sortColumn: 'total_profit',
    sortDirection: 'desc',
    selectedGridCount: null,    // Track selected row's grid count
    lowerLimit: 0,              // Store price range for grid calculations
    upperLimit: 0,
    capital: 0
};

const API_BASE = '';

// =============================================================================
// INITIALIZATION
// =============================================================================

document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
});

async function initializeApp() {
    // Load available symbols
    await loadSymbols();

    // Set up event listeners
    setupEventListeners();

    // Fetch initial price
    await fetchCurrentPrice();
}

// =============================================================================
// EVENT LISTENERS
// =============================================================================

function setupEventListeners() {
    // Form submission
    document.getElementById('optimizer-form').addEventListener('submit', handleFormSubmit);

    // Symbol change
    document.getElementById('symbol').addEventListener('change', fetchCurrentPrice);

    // Lookback slider
    document.getElementById('lookback-days').addEventListener('input', (e) => {
        document.getElementById('lookback-value').textContent = e.target.value;
    });

    // Table sorting
    document.querySelectorAll('.results-table th[data-sort]').forEach(th => {
        th.addEventListener('click', () => handleSort(th.dataset.sort));
    });

    // Add thousands separators to numeric inputs
    const numericInputs = ['lower-limit', 'upper-limit', 'capital'];
    numericInputs.forEach(id => {
        const input = document.getElementById(id);
        input.addEventListener('focus', handleNumericFocus);
        input.addEventListener('blur', handleNumericBlur);
    });
}

function handleNumericFocus(e) {
    // Remove formatting on focus to allow editing
    const value = e.target.value.replace(/,/g, '');
    e.target.value = value;
    e.target.type = 'number';
}

function handleNumericBlur(e) {
    // Add formatting on blur
    const value = parseFloat(e.target.value);
    if (!isNaN(value)) {
        e.target.type = 'text';
        e.target.value = value.toLocaleString('en-US', {
            minimumFractionDigits: 0,
            maximumFractionDigits: 2
        });
    }
}

// =============================================================================
// API FUNCTIONS
// =============================================================================

async function loadSymbols() {
    try {
        const response = await fetch(`${API_BASE}/api/symbols`);
        const data = await response.json();

        if (data.success && data.symbols.length > 0) {
            const select = document.getElementById('symbol');
            select.innerHTML = data.symbols
                .map(s => `<option value="${s}"${s === 'BTCUSDT' ? ' selected' : ''}>${s}</option>`)
                .join('');
        }
    } catch (error) {
        console.error('Failed to load symbols:', error);
    }
}

async function fetchCurrentPrice() {
    const symbol = document.getElementById('symbol').value;
    const priceDisplay = document.getElementById('current-price');

    priceDisplay.textContent = 'Loading...';
    priceDisplay.style.color = 'var(--text-secondary)';

    try {
        const response = await fetch(`${API_BASE}/api/price/${symbol}`);
        const data = await response.json();

        if (data.success) {
            priceDisplay.textContent = `Current: $${formatNumber(data.price, 2)}`;
            priceDisplay.style.color = 'var(--accent-success)';

            // Update default price range
            document.getElementById('lower-limit').placeholder = formatNumber(data.default_lower, 2);
            document.getElementById('upper-limit').placeholder = formatNumber(data.default_upper, 2);

            // Set values if empty (with formatting)
            const lowerInput = document.getElementById('lower-limit');
            const upperInput = document.getElementById('upper-limit');
            if (!lowerInput.value) {
                lowerInput.type = 'text';
                lowerInput.value = formatNumberInput(data.default_lower);
            }
            if (!upperInput.value) {
                upperInput.type = 'text';
                upperInput.value = formatNumberInput(data.default_upper);
            }

            // Format capital on initial load
            const capitalInput = document.getElementById('capital');
            if (capitalInput.value && capitalInput.type !== 'text') {
                const capValue = parseFloat(capitalInput.value);
                capitalInput.type = 'text';
                capitalInput.value = formatNumberInput(capValue);
            }
        } else {
            priceDisplay.textContent = 'Price unavailable';
            priceDisplay.style.color = 'var(--accent-danger)';
        }
    } catch (error) {
        priceDisplay.textContent = 'Connection error';
        priceDisplay.style.color = 'var(--accent-danger)';
    }
}

function parseFormattedNumber(value) {
    // Remove commas and parse as float
    return parseFloat(String(value).replace(/,/g, ''));
}

async function handleFormSubmit(e) {
    e.preventDefault();

    const formData = {
        symbol: document.getElementById('symbol').value,
        lower_limit: parseFormattedNumber(document.getElementById('lower-limit').value),
        upper_limit: parseFormattedNumber(document.getElementById('upper-limit').value),
        capital: parseFormattedNumber(document.getElementById('capital').value),
        lookback_days: parseInt(document.getElementById('lookback-days').value),
        max_grids: parseInt(document.getElementById('max-grids').value)
    };

    // Validation
    if (formData.lower_limit >= formData.upper_limit) {
        showError('Lower limit must be less than upper limit');
        return;
    }

    if (formData.capital <= 0) {
        showError('Capital must be positive');
        return;
    }

    // Save params for CSV download
    state.lastSymbol = formData.symbol;
    state.lastLookbackDays = formData.lookback_days;

    // Start optimization
    setLoadingState(true);

    try {
        const response = await fetch(`${API_BASE}/api/optimize`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(formData)
        });

        const data = await response.json();

        if (data.success) {
            state.currentJobId = data.job_id;
            startPolling();
        } else {
            showError(data.error || 'Failed to start optimization');
            setLoadingState(false);
        }
    } catch (error) {
        showError('Connection error. Please try again.');
        setLoadingState(false);
    }
}

function startPolling() {
    // Clear any existing interval
    if (state.pollInterval) {
        clearInterval(state.pollInterval);
    }

    // Poll every 500ms
    state.pollInterval = setInterval(checkJobStatus, 500);
}

async function checkJobStatus() {
    if (!state.currentJobId) return;

    try {
        const response = await fetch(`${API_BASE}/api/status/${state.currentJobId}`);
        const data = await response.json();

        if (data.success) {
            updateProgress(data.progress, data.message);

            if (data.status === 'completed') {
                clearInterval(state.pollInterval);
                state.pollInterval = null;
                try {
                    displayResults(data.result);
                } catch (displayError) {
                    console.error('Error displaying results:', displayError);
                    showError('Error displaying results: ' + displayError.message);
                }
                setLoadingState(false);
            } else if (data.status === 'failed') {
                clearInterval(state.pollInterval);
                state.pollInterval = null;
                showError(data.error || 'Optimization failed');
                setLoadingState(false);
            }
        }
    } catch (error) {
        console.error('Polling error:', error);
    }
}

// =============================================================================
// UI FUNCTIONS
// =============================================================================

function setLoadingState(isLoading) {
    const btn = document.getElementById('run-btn');
    const btnText = btn.querySelector('.btn-text');
    const btnLoader = btn.querySelector('.btn-loader');
    const progressContainer = document.getElementById('progress-container');

    btn.disabled = isLoading;
    btnText.textContent = isLoading ? 'Running...' : 'Run Optimization';
    btnLoader.style.display = isLoading ? 'block' : 'none';
    progressContainer.style.display = isLoading ? 'block' : 'none';

    if (isLoading) {
        document.getElementById('results-placeholder').style.display = 'none';
        document.getElementById('results-content').style.display = 'none';
        document.getElementById('error-message').style.display = 'none';
    }
}

function updateProgress(percent, message) {
    document.getElementById('progress-fill').style.width = `${percent}%`;
    document.getElementById('progress-text').textContent = message || `${percent}% complete`;
}

function showError(message) {
    document.getElementById('results-placeholder').style.display = 'none';
    document.getElementById('results-content').style.display = 'none';
    document.getElementById('error-message').style.display = 'flex';
    document.getElementById('error-text').textContent = message;
}

function resetUI() {
    document.getElementById('results-placeholder').style.display = 'flex';
    document.getElementById('results-content').style.display = 'none';
    document.getElementById('error-message').style.display = 'none';
    document.getElementById('progress-container').style.display = 'none';
}

function displayResults(result) {
    // Store all results for sorting
    state.allResults = result.all_results;

    // Store candle data for CSV download
    state.candleData = result.candle_data || [];

    // Store parameters for grid calculations
    const optimal = result.optimal;
    state.lowerLimit = optimal.lower_limit;
    state.upperLimit = optimal.upper_limit;
    state.capital = parseFormattedNumber(document.getElementById('capital').value);
    state.selectedGridCount = optimal.num_grids;

    // Show results panel
    document.getElementById('results-placeholder').style.display = 'none';
    document.getElementById('error-message').style.display = 'none';
    document.getElementById('results-content').style.display = 'block';

    // Update summary cards
    document.getElementById('optimal-grids').textContent = optimal.num_grids;
    document.getElementById('grid-spacing').textContent = `$${formatNumber(optimal.spacing, 0)}`;
    document.getElementById('total-profit').textContent = `$${formatNumber(optimal.total_profit, 0)}`;
    document.getElementById('roi-percent').textContent = `${formatNumber(optimal.roi_percent, 2)}%`;
    document.getElementById('total-trades').textContent = formatNumber(optimal.total_trades, 0);
    document.getElementById('profit-per-trade').textContent = `$${formatNumber(optimal.profit_per_trade, 0)}`;

    // Update data info
    const info = result.data_info;
    document.getElementById('data-range').textContent = `${info.date_start} to ${info.date_end}`;
    document.getElementById('data-candles').textContent = formatNumber(info.total_candles, 0);
    document.getElementById('data-prices').textContent = `$${formatNumber(info.price_low, 2)} - $${formatNumber(info.price_high, 2)}`;

    // Update cache stats display
    const cacheStatsItem = document.getElementById('cache-stats-item');
    const dataSourceEl = document.getElementById('data-source');
    if (info.cache_stats) {
        const stats = info.cache_stats;
        let sourceText = '';
        let sourceClass = '';

        if (stats.source === 'cache') {
            sourceText = `100% from cache (${formatNumber(stats.cached_count, 0)} candles)`;
            sourceClass = 'source-cache';
        } else if (stats.source === 'api_only') {
            sourceText = `100% from API (no database configured)`;
            sourceClass = 'source-api';
        } else if (stats.source === 'api') {
            sourceText = `100% from API (${formatNumber(stats.api_count, 0)} candles fetched)`;
            sourceClass = 'source-api';
        } else if (stats.source === 'mixed') {
            sourceText = `${stats.cache_percent}% cached, ${100 - stats.cache_percent}% from API`;
            sourceClass = 'source-mixed';
        }

        dataSourceEl.textContent = sourceText;
        dataSourceEl.className = sourceClass;
        cacheStatsItem.style.display = 'inline';
    } else {
        cacheStatsItem.style.display = 'none';
    }

    // Populate grid levels table
    const gridLevelsContainer = document.getElementById('grid-levels-container');
    if (gridLevelsContainer) {
        if (optimal.grid_levels && optimal.grid_levels.length > 0) {
            // Update headers with selected grid count
            document.getElementById('selected-grids-count').textContent = optimal.num_grids;
            const gridRangeLabel = document.getElementById('grid-range-label');
            if (gridRangeLabel) {
                gridRangeLabel.textContent = `($${formatNumber(optimal.lower_limit, 0)} - $${formatNumber(optimal.upper_limit, 0)})`;
            }

            const gridLevelsTbody = document.getElementById('grid-levels-tbody');
            if (gridLevelsTbody) {
                gridLevelsTbody.innerHTML = optimal.grid_levels.map(level => `
                    <tr>
                        <td>${level.level}</td>
                        <td class="buy-price">$${formatNumber(level.buy_price, 2)}</td>
                        <td class="sell-price">$${formatNumber(level.sell_price, 2)}</td>
                        <td>$${formatNumber(level.profit_per_trade, 0)}</td>
                    </tr>
                `).join('');
            }

            gridLevelsContainer.style.display = 'block';
        } else {
            gridLevelsContainer.style.display = 'none';
        }
    }

    // Populate trade log
    if (result.trade_log && result.trade_log.length > 0) {
        state.tradeLog = result.trade_log;
        state.tradeLogFilter = 'all';
        document.getElementById('trade-log-grids').textContent = optimal.num_grids;
        document.getElementById('trade-log-count').textContent = `(${result.trade_log.length} trades)`;
        populateTradeLog(result.trade_log);
        document.getElementById('trade-log-container').style.display = 'block';

        // Set up filter button listeners
        document.querySelectorAll('.btn-filter').forEach(btn => {
            btn.addEventListener('click', (e) => {
                document.querySelectorAll('.btn-filter').forEach(b => b.classList.remove('active'));
                e.target.classList.add('active');
                state.tradeLogFilter = e.target.dataset.filter;
                filterTradeLog();
            });
        });
    } else {
        document.getElementById('trade-log-container').style.display = 'none';
    }

    // Create charts
    createCharts(result.all_results, optimal.num_grids);

    // Populate table
    populateTable(result.top_results);
}

// =============================================================================
// CHART FUNCTIONS
// =============================================================================

function createCharts(results, optimalGrids) {
    // Destroy existing charts
    if (state.profitChart) state.profitChart.destroy();
    if (state.tradesChart) state.tradesChart.destroy();

    const chartOptions = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: { display: false },
            tooltip: {
                backgroundColor: '#1c2128',
                titleColor: '#e6edf3',
                bodyColor: '#8b949e',
                borderColor: '#30363d',
                borderWidth: 1
            }
        },
        scales: {
            x: {
                grid: { color: '#21262d' },
                ticks: { color: '#8b949e' }
            },
            y: {
                grid: { color: '#21262d' },
                ticks: { color: '#8b949e' }
            }
        }
    };

    // Profit Chart
    const profitCtx = document.getElementById('profit-chart').getContext('2d');
    const profitData = results.map(r => ({ x: r.num_grids, y: r.total_profit }));

    // Find optimal point index
    const optimalIndex = results.findIndex(r => r.num_grids === optimalGrids);

    // Create point colors array
    const pointColors = results.map((r, i) =>
        i === optimalIndex ? '#f85149' : '#58a6ff'
    );
    const pointRadius = results.map((r, i) =>
        i === optimalIndex ? 8 : 2
    );

    state.profitChart = new Chart(profitCtx, {
        type: 'line',
        data: {
            labels: results.map(r => r.num_grids),
            datasets: [{
                label: 'Total Profit',
                data: results.map(r => r.total_profit),
                borderColor: '#58a6ff',
                backgroundColor: 'rgba(88, 166, 255, 0.1)',
                fill: true,
                tension: 0.3,
                pointBackgroundColor: pointColors,
                pointRadius: pointRadius,
                pointHoverRadius: 6
            }]
        },
        options: {
            ...chartOptions,
            scales: {
                ...chartOptions.scales,
                x: {
                    ...chartOptions.scales.x,
                    title: {
                        display: true,
                        text: 'Number of Grids',
                        color: '#8b949e'
                    }
                },
                y: {
                    ...chartOptions.scales.y,
                    title: {
                        display: true,
                        text: 'Profit ($)',
                        color: '#8b949e'
                    }
                }
            },
            plugins: {
                ...chartOptions.plugins,
                annotation: {
                    annotations: {
                        optimal: {
                            type: 'point',
                            xValue: optimalGrids,
                            yValue: results[optimalIndex]?.total_profit || 0,
                            backgroundColor: '#f85149',
                            radius: 8
                        }
                    }
                }
            }
        }
    });

    // Trades Chart
    const tradesCtx = document.getElementById('trades-chart').getContext('2d');

    state.tradesChart = new Chart(tradesCtx, {
        type: 'bar',
        data: {
            labels: results.map(r => r.num_grids),
            datasets: [{
                label: 'Trade Count',
                data: results.map(r => r.total_trades),
                backgroundColor: results.map((r, i) =>
                    i === optimalIndex ? '#3fb950' : 'rgba(63, 185, 80, 0.5)'
                ),
                borderColor: '#3fb950',
                borderWidth: 1
            }]
        },
        options: {
            ...chartOptions,
            scales: {
                ...chartOptions.scales,
                x: {
                    ...chartOptions.scales.x,
                    title: {
                        display: true,
                        text: 'Number of Grids',
                        color: '#8b949e'
                    }
                },
                y: {
                    ...chartOptions.scales.y,
                    title: {
                        display: true,
                        text: 'Trade Count',
                        color: '#8b949e'
                    }
                }
            }
        }
    });
}

// =============================================================================
// TABLE FUNCTIONS
// =============================================================================

function populateTable(results) {
    const tbody = document.getElementById('results-tbody');
    tbody.innerHTML = results.map((r, i) => `
        <tr data-grids="${r.num_grids}" class="${r.num_grids === state.selectedGridCount ? 'selected' : ''}">
            <td>${r.num_grids}</td>
            <td>$${formatNumber(r.spacing, 0)}</td>
            <td>${formatNumber(r.total_trades, 0)}</td>
            <td>$${formatNumber(r.total_profit, 0)}</td>
            <td>${formatNumber(r.roi_percent, 2)}%</td>
            <td>$${formatNumber(r.profit_per_trade || 0, 0)}</td>
            <td>${formatNumber(r.trades_per_day || 0, 1)}</td>
        </tr>
    `).join('');

    // Add click handlers to rows
    tbody.querySelectorAll('tr').forEach(row => {
        row.addEventListener('click', () => {
            const numGrids = parseInt(row.dataset.grids);
            selectConfiguration(numGrids);
        });
    });
}

function handleSort(column) {
    // Toggle direction if same column
    if (state.sortColumn === column) {
        state.sortDirection = state.sortDirection === 'asc' ? 'desc' : 'asc';
    } else {
        state.sortColumn = column;
        state.sortDirection = 'desc';
    }

    // Sort results
    const sorted = [...state.allResults].sort((a, b) => {
        const aVal = a[column] || 0;
        const bVal = b[column] || 0;
        return state.sortDirection === 'asc' ? aVal - bVal : bVal - aVal;
    });

    // Update table
    populateTable(sorted.slice(0, 20));

    // Update header indicators
    document.querySelectorAll('.results-table th').forEach(th => {
        th.classList.remove('sorted-asc', 'sorted-desc');
        if (th.dataset.sort === column) {
            th.classList.add(state.sortDirection === 'asc' ? 'sorted-asc' : 'sorted-desc');
        }
    });
}

// =============================================================================
// CONFIGURATION SELECTION
// =============================================================================

async function selectConfiguration(numGrids) {
    // Update state
    state.selectedGridCount = numGrids;

    // Update row highlighting
    document.querySelectorAll('#results-tbody tr').forEach(row => {
        row.classList.remove('selected');
        if (parseInt(row.dataset.grids) === numGrids) {
            row.classList.add('selected');
        }
    });

    // Update grid levels (calculate client-side)
    updateGridLevels(numGrids);

    // Update trade log header to show loading
    document.getElementById('trade-log-grids').textContent = numGrids;
    document.getElementById('trade-log-count').textContent = '(loading...)';
    document.getElementById('trade-log-loading').style.display = 'block';
    document.getElementById('trade-log-tbody').innerHTML = '';

    // Fetch trade log from API
    try {
        const response = await fetch(`${API_BASE}/api/backtest/${state.currentJobId}/${numGrids}`);
        const data = await response.json();

        if (data.success) {
            state.tradeLog = data.trade_log;
            state.tradeLogFilter = 'all';
            document.getElementById('trade-log-count').textContent = `(${data.trade_log.length} trades)`;
            populateTradeLog(data.trade_log);

            // Reset filter buttons
            document.querySelectorAll('.btn-filter').forEach(btn => {
                btn.classList.remove('active');
                if (btn.dataset.filter === 'all') {
                    btn.classList.add('active');
                }
            });
        } else {
            document.getElementById('trade-log-count').textContent = '(error loading)';
        }
    } catch (error) {
        console.error('Error fetching trade log:', error);
        document.getElementById('trade-log-count').textContent = '(error loading)';
    }

    document.getElementById('trade-log-loading').style.display = 'none';

    // Show the containers
    document.getElementById('grid-levels-container').style.display = 'block';
    document.getElementById('trade-log-container').style.display = 'block';
}

function updateGridLevels(numGrids) {
    const spacing = (state.upperLimit - state.lowerLimit) / numGrids;

    // Update header
    document.getElementById('selected-grids-count').textContent = numGrids;
    document.getElementById('grid-range-label').textContent =
        `($${formatNumber(state.lowerLimit, 0)} - $${formatNumber(state.upperLimit, 0)})`;

    // Calculate and display grid levels
    const gridLevels = [];
    for (let i = 0; i < numGrids; i++) {
        const buyPrice = state.lowerLimit + i * spacing;
        const sellPrice = buyPrice + spacing;
        const profitPerTrade = (state.capital / numGrids) * (spacing / buyPrice);
        gridLevels.push({
            level: i + 1,
            buy_price: buyPrice,
            sell_price: sellPrice,
            profit_per_trade: profitPerTrade
        });
    }

    const tbody = document.getElementById('grid-levels-tbody');
    tbody.innerHTML = gridLevels.map(level => `
        <tr>
            <td>${level.level}</td>
            <td class="buy-price">$${formatNumber(level.buy_price, 2)}</td>
            <td class="sell-price">$${formatNumber(level.sell_price, 2)}</td>
            <td>$${formatNumber(level.profit_per_trade, 0)}</td>
        </tr>
    `).join('');
}

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

function formatNumber(num, decimals = 2) {
    if (num === null || num === undefined || isNaN(num)) return '--';
    return num.toLocaleString('en-US', {
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals
    });
}

function formatNumberInput(num) {
    if (num === null || num === undefined || isNaN(num)) return '';
    return num.toLocaleString('en-US', {
        minimumFractionDigits: 0,
        maximumFractionDigits: 2
    });
}

function downloadCandles() {
    if (!state.candleData || state.candleData.length === 0) {
        alert('No candle data available. Please run an optimization first.');
        return;
    }

    // Generate CSV content client-side
    const headers = ['timestamp', 'open', 'high', 'low', 'close', 'volume'];
    const csvRows = [headers.join(',')];

    for (const candle of state.candleData) {
        const row = headers.map(h => candle[h] ?? '').join(',');
        csvRows.push(row);
    }

    const csvContent = csvRows.join('\n');

    // Create blob and trigger download
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    const filename = `candles_${state.lastSymbol}_${new Date().toISOString().split('T')[0]}.csv`;
    link.setAttribute('href', url);
    link.setAttribute('download', filename);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
}

// =============================================================================
// TRADE LOG FUNCTIONS
// =============================================================================

function populateTradeLog(trades) {
    const tbody = document.getElementById('trade-log-tbody');
    tbody.innerHTML = trades.map(trade => `
        <tr>
            <td>${trade.timestamp}</td>
            <td class="${trade.type === 'BUY' ? 'trade-buy' : 'trade-sell'}">${trade.type}</td>
            <td>${trade.level}</td>
            <td>$${formatNumber(trade.price, 2)}</td>
            <td class="${trade.profit ? 'profit-cell' : ''}">${trade.profit ? '$' + formatNumber(trade.profit, 0) : '-'}</td>
        </tr>
    `).join('');
}

function filterTradeLog() {
    const filter = state.tradeLogFilter;
    let filtered = state.tradeLog;

    if (filter !== 'all') {
        filtered = state.tradeLog.filter(t => t.type === filter);
    }

    document.getElementById('trade-log-count').textContent = `(${filtered.length} trades)`;
    populateTradeLog(filtered);
}

function downloadTradeLog() {
    if (!state.tradeLog || state.tradeLog.length === 0) {
        alert('No trade log available. Please run an optimization first.');
        return;
    }

    // Generate CSV content
    const headers = ['timestamp', 'type', 'level', 'price', 'profit'];
    const csvRows = [headers.join(',')];

    for (const trade of state.tradeLog) {
        const row = [
            trade.timestamp,
            trade.type,
            trade.level,
            trade.price,
            trade.profit || ''
        ].join(',');
        csvRows.push(row);
    }

    const csvContent = csvRows.join('\n');

    // Create blob and trigger download
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    const filename = `trade_log_${state.lastSymbol}_${new Date().toISOString().split('T')[0]}.csv`;
    link.setAttribute('href', url);
    link.setAttribute('download', filename);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
}

// Global functions
window.resetUI = resetUI;
window.downloadCandles = downloadCandles;
window.downloadTradeLog = downloadTradeLog;

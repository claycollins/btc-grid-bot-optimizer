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
    sortColumn: 'total_profit',
    sortDirection: 'desc'
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

            // Set values if empty
            if (!document.getElementById('lower-limit').value) {
                document.getElementById('lower-limit').value = data.default_lower;
            }
            if (!document.getElementById('upper-limit').value) {
                document.getElementById('upper-limit').value = data.default_upper;
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

async function handleFormSubmit(e) {
    e.preventDefault();

    const formData = {
        symbol: document.getElementById('symbol').value,
        lower_limit: parseFloat(document.getElementById('lower-limit').value),
        upper_limit: parseFloat(document.getElementById('upper-limit').value),
        capital: parseFloat(document.getElementById('capital').value),
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
                displayResults(data.result);
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

    // Show results panel
    document.getElementById('results-placeholder').style.display = 'none';
    document.getElementById('error-message').style.display = 'none';
    document.getElementById('results-content').style.display = 'block';

    // Update summary cards
    const optimal = result.optimal;
    document.getElementById('optimal-grids').textContent = optimal.num_grids;
    document.getElementById('grid-spacing').textContent = `$${formatNumber(optimal.spacing, 2)}`;
    document.getElementById('total-profit').textContent = `$${formatNumber(optimal.total_profit, 2)}`;
    document.getElementById('roi-percent').textContent = `${formatNumber(optimal.roi_percent, 2)}%`;
    document.getElementById('total-trades').textContent = formatNumber(optimal.total_trades, 0);
    document.getElementById('profit-per-trade').textContent = `$${formatNumber(optimal.profit_per_trade, 4)}`;

    // Update data info
    const info = result.data_info;
    document.getElementById('data-range').textContent = `${info.date_start} to ${info.date_end}`;
    document.getElementById('data-candles').textContent = formatNumber(info.total_candles, 0);
    document.getElementById('data-prices').textContent = `$${formatNumber(info.price_low, 2)} - $${formatNumber(info.price_high, 2)}`;

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
        <tr>
            <td>${r.num_grids}</td>
            <td>$${formatNumber(r.spacing, 2)}</td>
            <td>${formatNumber(r.total_trades, 0)}</td>
            <td>$${formatNumber(r.total_profit, 2)}</td>
            <td>${formatNumber(r.roi_percent, 2)}%</td>
            <td>$${formatNumber(r.profit_per_trade || 0, 4)}</td>
            <td>${formatNumber(r.trades_per_day || 0, 1)}</td>
        </tr>
    `).join('');
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
// UTILITY FUNCTIONS
// =============================================================================

function formatNumber(num, decimals = 2) {
    if (num === null || num === undefined || isNaN(num)) return '--';
    return num.toLocaleString('en-US', {
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals
    });
}

function downloadCandles() {
    if (!state.currentJobId) {
        alert('No optimization data available');
        return;
    }

    window.location.href = `${API_BASE}/api/download/${state.currentJobId}/candles`;
}

// Global functions
window.resetUI = resetUI;
window.downloadCandles = downloadCandles;

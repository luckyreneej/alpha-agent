# Alpha-Agent System User Guide - Part 2: Usage and Examples

## Step-by-Step Usage Guide

### Initializing the System

To initialize the Alpha-Agent system, you need to create instances of the main components:

```python
# Import necessary modules
from utils.data.polygon_api import PolygonAPI
from backtest.historical_data_fetcher import HistoricalDataFetcher
from backtest.backtest_engine import BacktestEngine
from backtest.strategy import MovingAverageCrossStrategy

# Initialize API and data fetcher
api = PolygonAPI()  # Will use POLYGON_API_KEY environment variable
data_fetcher = HistoricalDataFetcher(api_key=None)  # None means use environment variable

# Create a strategy
strategy = MovingAverageCrossStrategy(fast_period=50, slow_period=200)

# Initialize backtest engine
backtest = BacktestEngine(
    initial_capital=100000,
    data_fetcher=data_fetcher,
    strategy=strategy
)
```

### Running Stock Prediction Analysis

To analyze and predict stock price movements:

```python
# Fetch historical data for analysis
start_date = "2022-01-01"
end_date = "2022-12-31"
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]

# Get the data
historical_data = data_fetcher.fetch_stock_history(
    tickers=tickers,
    start_date=start_date,
    end_date=end_date
)

# Format for prediction
analysis_data = data_fetcher.prepare_backtest_data(
    dataset={'stocks': historical_data},
    format_type='panel'
)

# Run analysis (example using machine learning)
from utils.analysis.ml_predictor import MLPredictor

predictor = MLPredictor()
predictor.train(analysis_data, target_column='close', horizon=5)
predictions = predictor.predict()

print("5-day price predictions:")
print(predictions)
```

### Generating Trading Signals

To generate trading signals using a strategy:

```python
# Initialize a strategy
from backtest.strategy import RSIStrategy

rsi_strategy = RSIStrategy(period=14, oversold=30, overbought=70)

# Get latest data point
latest_date = historical_data['AAPL'].index[-1]
positions = {'AAPL': 0, 'MSFT': 0, 'GOOGL': 0, 'AMZN': 0, 'META': 0}

# Generate signals for each stock
for ticker, data in historical_data.items():
    signals = rsi_strategy.generate_signals(data, latest_date, positions)
    signal_value = signals[ticker]
    signal_type = "BUY" if signal_value > 0 else ("SELL" if signal_value < 0 else "HOLD")
    print(f"{ticker}: {signal_type} signal (value: {signal_value})")
```

### Conducting Backtests

To backtest a trading strategy:

```python
# Set up backtest parameters
backtest_params = {
    "initial_capital": 100000,
    "start_date": "2020-01-01",
    "end_date": "2022-12-31",
    "tickers": ["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
    "position_size": 0.2,  # Allocate 20% of portfolio to each position
    "commission": 0.001,   # 0.1% commission per trade
    "slippage": 0.001     # 0.1% slippage per trade
}

# Configure and run backtest
backtest = BacktestEngine(
    initial_capital=backtest_params["initial_capital"],
    data_fetcher=data_fetcher,
    strategy=rsi_strategy
)

results = backtest.run(
    tickers=backtest_params["tickers"],
    start_date=backtest_params["start_date"],
    end_date=backtest_params["end_date"],
    position_size=backtest_params["position_size"],
    commission=backtest_params["commission"],
    slippage=backtest_params["slippage"]
)

print("Backtest completed!")
print(f"Final portfolio value: ${results['final_portfolio_value']:.2f}")
print(f"Total return: {results['total_return']:.2%}")
print(f"Sharpe ratio: {results['metrics']['sharpe_ratio']:.2f}")
```

### Visualizing and Analyzing Results

To visualize and analyze backtest results:

```python
from backtest.performance_metrics import PerformanceAnalyzer

# Create performance analyzer
analyzer = PerformanceAnalyzer()

# Plot returns analysis
analyzer.plot_returns_analysis(
    returns=results['returns'],
    benchmark_returns=results.get('benchmark_returns'),
    save_path="reports/returns_analysis.png"
)

# Calculate detailed metrics
detailed_metrics = analyzer.calculate_metrics(
    returns=results['returns'],
    benchmark_returns=results.get('benchmark_returns')
)

# Print key metrics
for metric_name, value in detailed_metrics.items():
    if isinstance(value, float):
        if 'return' in metric_name or 'drawdown' in metric_name:
            print(f"{metric_name}: {value:.2%}")
        else:
            print(f"{metric_name}: {value:.4f}")
```

## Code Examples

### Basic Usage Examples

#### Running a Simple Backtest

```python
from backtest.historical_data_fetcher import HistoricalDataFetcher
from backtest.backtest_engine import BacktestEngine
from backtest.strategy import MovingAverageCrossStrategy

# Initialize components
data_fetcher = HistoricalDataFetcher()
strategy = MovingAverageCrossStrategy(fast_period=50, slow_period=200)
backtest = BacktestEngine(initial_capital=100000, data_fetcher=data_fetcher, strategy=strategy)

# Run backtest
results = backtest.run(
    tickers=["SPY"],
    start_date="2018-01-01",
    end_date="2022-12-31"
)

# Print results summary
print(f"Total Return: {results['total_return']:.2%}")
print(f"Annualized Return: {results['metrics']['annualized_return']:.2%}")
print(f"Max Drawdown: {results['metrics']['max_drawdown']:.2%}")
print(f"Sharpe Ratio: {results['metrics']['sharpe_ratio']:.2f}")
```

#### Comparing Multiple Strategies

```python
from backtest.strategy import MovingAverageCrossStrategy, RSIStrategy, MACDStrategy
from backtest.backtest_engine import BacktestEngine
from backtest.historical_data_fetcher import HistoricalDataFetcher
from backtest.performance_metrics import PerformanceAnalyzer
import matplotlib.pyplot as plt

# Initialize data fetcher
data_fetcher = HistoricalDataFetcher()

# Define backtest parameters
backtest_params = {
    "tickers": ["SPY"],
    "start_date": "2018-01-01",
    "end_date": "2022-12-31",
    "initial_capital": 100000
}

# Define strategies to test
strategies = [
    MovingAverageCrossStrategy(fast_period=50, slow_period=200),
    RSIStrategy(period=14, oversold=30, overbought=70),
    MACDStrategy(fast_period=12, slow_period=26, signal_period=9)
]

# Run backtests and collect results
results = []
strategy_names = []

for strategy in strategies:
    backtest = BacktestEngine(
        initial_capital=backtest_params["initial_capital"],
        data_fetcher=data_fetcher,
        strategy=strategy
    )
    
    result = backtest.run(
        tickers=backtest_params["tickers"],
        start_date=backtest_params["start_date"],
        end_date=backtest_params["end_date"]
    )
    
    results.append(result)
    strategy_names.append(strategy.name)

# Compare performance
fig, ax = plt.subplots(figsize=(12, 6))

for i, result in enumerate(results):
    equity_curve = (1 + result['returns']).cumprod()
    ax.plot(equity_curve.index, equity_curve.values, label=strategy_names[i])

ax.set_title('Strategy Comparison')
ax.set_xlabel('Date')
ax.set_ylabel('Portfolio Value (normalized)')
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.savefig('reports/strategy_comparison.png')
```
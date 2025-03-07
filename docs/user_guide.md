# Alpha-Agent System User Guide

This guide provides comprehensive instructions for using the Alpha-Agent backtesting framework.

## Table of Contents

- [Getting Started](#getting-started)
- [System Architecture](#system-architecture)
- [Backtesting Framework](#backtesting-framework)
- [Performance Analysis](#performance-analysis)
- [Strategy Development](#strategy-development)
- [Troubleshooting](#troubleshooting)

## Getting Started

### Installation

1. Clone the repository:
```bash
git clone https://github.com/your-repo/alpha-agent.git
cd alpha-agent
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Main dependencies:
- pandas
- numpy
- matplotlib
- seaborn
- scipy
- requests
- polygon-api-client

### API Key Setup

This system requires a Polygon.io API key for historical data retrieval:

1. Register for an API key at [Polygon.io](https://polygon.io/)
2. Set as an environment variable:
   ```bash
   export POLYGON_API_KEY="your_api_key_here"
   ```

## System Architecture

The Alpha-Agent framework consists of these primary components:

1. **Data Fetching**: `historical_data_fetcher.py` handles retrieving and caching market data
2. **Strategy Implementation**: `strategy.py` contains base strategy classes and implementations
3. **Backtesting Engine**: `backtest_engine.py` orchestrates backtest execution
4. **Performance Analysis**: `performance_metrics.py` calculates performance metrics
5. **Portfolio Optimization**: `portfolio_optimizer.py` handles position sizing and allocation

## Backtesting Framework

### Basic Backtest

```python
from backtest.historical_data_fetcher import HistoricalDataFetcher
from backtest.backtest_engine import BacktestEngine
from backtest.strategy import MovingAverageCrossStrategy
from backtest.performance_metrics import PerformanceAnalyzer

# Initialize components
data_fetcher = HistoricalDataFetcher()
strategy = MovingAverageCrossStrategy(fast_period=50, slow_period=200)
backtest = BacktestEngine(initial_capital=100000, data_fetcher=data_fetcher, strategy=strategy)

# Run backtest
results = backtest.run(
    tickers=["AAPL", "MSFT", "GOOGL"],
    start_date="2020-01-01",
    end_date="2022-12-31",
    position_size=0.3,  # 30% of portfolio per position
    commission=0.001,   # 0.1% commission
    slippage=0.001      # 0.1% slippage
)

# Analyze results
analyzer = PerformanceAnalyzer()
metrics = analyzer.calculate_metrics(returns=results['returns'])

print(f"Total Return: {metrics['total_return']:.2%}")
print(f"Annualized Return: {metrics['annualized_return']:.2%}")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")

# Visualize results
analyzer.plot_returns_analysis(
    returns=results['returns'],
    save_path="backtest_results.png"
)
```

### Available Strategy Types

The framework includes several built-in strategy classes:

1. **MovingAverageCrossStrategy**: Generates signals based on fast and slow moving average crossovers
2. **RSIStrategy**: Trades based on Relative Strength Index overbought/oversold levels
3. **MACDStrategy**: Uses Moving Average Convergence/Divergence indicator crossovers

You can create custom strategies by:
1. Inheriting from the base `Strategy` class
2. Using the `create_strategy_from_function()` helper for simpler strategies
3. Combining multiple strategies with `create_combined_strategy()`

## Performance Analysis

The `PerformanceAnalyzer` class calculates key performance metrics:

- Total and annualized returns
- Volatility and drawdown metrics
- Sharpe, Sortino, and Calmar ratios
- Alpha, beta, and other risk-adjusted metrics

```python
from backtest.performance_metrics import PerformanceAnalyzer

analyzer = PerformanceAnalyzer()

# Calculate metrics
metrics = analyzer.calculate_metrics(
    returns=results['returns'],
    benchmark_returns=benchmark_returns
)

# Generate visual analysis
analyzer.plot_returns_analysis(
    returns=results['returns'],
    benchmark_returns=benchmark_returns,
    save_path="performance_analysis.png"
)
```

## Strategy Development

### Creating a Custom Strategy

Extend the base Strategy class:

```python
from backtest.strategy import Strategy
import pandas as pd

class MyCustomStrategy(Strategy):
    def __init__(self, param1=10, param2=20):
        super().__init__()
        self.set_parameters(param1=param1, param2=param2)
        self.set_description(f"My Custom Strategy ({param1}, {param2})")
    
    def generate_signals(self, data, current_date, positions):
        signals = {}
        # Your strategy logic here
        # ...
        return signals
```

### Using a Function-Based Strategy

For simpler strategies:

```python
from backtest.strategy import create_strategy_from_function

def my_simple_strategy(data, current_date, positions):
    signals = {}
    # Strategy logic here
    # ...
    return signals

# Create a strategy object from the function
strategy = create_strategy_from_function(
    my_simple_strategy, 
    name="MySimpleStrategy"
)
```

## Troubleshooting

### Common Issues

#### API Rate Limiting

If encountering rate limit errors with Polygon.io:

```python
from backtest.historical_data_fetcher import HistoricalDataFetcher

# Use caching to reduce API calls
data_fetcher = HistoricalDataFetcher(use_cache=True, cache_dir="data_cache")

# Or adjust the retry settings
data_fetcher.set_rate_limit_params(max_retries=5, retry_delay=2.0)
```

#### Memory Issues with Large Datasets

Use date chunking for large historical datasets:

```python
from datetime import datetime, timedelta

start_date = datetime.strptime("2018-01-01", "%Y-%m-%d")
end_date = datetime.strptime("2022-12-31", "%Y-%m-%d")
chunk_size = timedelta(days=90)

# Process in chunks
chunks = []
current_date = start_date

while current_date < end_date:
    chunk_end = min(current_date + chunk_size, end_date)
    
    # Process this date range
    chunk_data = data_fetcher.fetch_stock_history(
        tickers=["AAPL"],
        start_date=current_date.strftime("%Y-%m-%d"),
        end_date=chunk_end.strftime("%Y-%m-%d")
    )
    
    # Do something with the chunk
    # ...
    
    current_date = chunk_end + timedelta(days=1)
```

#### Missing Data

For handling missing data points:

```python
# Fill missing values in historical data
for ticker, df in historical_data.items():
    # Forward fill missing values (up to 5 consecutive NaN values)
    df.fillna(method='ffill', limit=5, inplace=True)
    
    # Fill any remaining NaN values with interpolation
    df.interpolate(method='linear', inplace=True)
```

### Performance Optimization Tips

1. Use vectorized operations instead of loops
2. Enable caching for historical data
3. Use the appropriate time frame for your strategy (daily vs minute data)
4. Profile memory usage for large backtests
5. Consider parallel processing for parameter optimization
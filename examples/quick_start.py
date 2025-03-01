# -*- coding: utf-8 -*-

"""
Alpha-Agent Quick Start Example

This script demonstrates basic usage of the Alpha-Agent backtesting framework
with a simple moving average crossover strategy.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime, timedelta

# Ensure the alpha-agent package is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backtest.historical_data_fetcher import HistoricalDataFetcher
from backtest.backtest_engine import BacktestEngine
from backtest.strategy import MovingAverageCrossStrategy
from backtest.performance_metrics import PerformanceAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create output directory if it doesn't exist
output_dir = "./results"
os.makedirs(output_dir, exist_ok=True)


# Define a simple mock API client for demonstration purposes
class SimpleMockAPIClient:
    def get_stock_bars(self, ticker, timespan, from_date, to_date):
        """Generate sample stock data for backtesting demonstration."""
        # Create date range
        start_date = pd.to_datetime(from_date)
        end_date = pd.to_datetime(to_date)
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days

        # Set seed for reproducibility
        np.random.seed(42 + hash(ticker) % 100)

        # Generate mock price data
        if ticker == 'AAPL':
            base_price = 150.0
            trend = 0.0002  # slight upward trend
            volatility = 0.015
        elif ticker == 'MSFT':
            base_price = 250.0
            trend = 0.0003
            volatility = 0.012
        elif ticker == 'GOOGL':
            base_price = 2500.0
            trend = 0.0001
            volatility = 0.018
        elif ticker == 'SPY':  # Benchmark
            base_price = 400.0
            trend = 0.0001
            volatility = 0.010
        else:
            base_price = 100.0
            trend = 0.0
            volatility = 0.02

        # Generate price series with random walk
        closes = [base_price]
        for i in range(1, len(date_range)):
            # Random walk with slight trend
            daily_return = np.random.normal(trend, volatility)
            new_price = closes[-1] * (1 + daily_return)
            closes.append(new_price)

        # Create full OHLCV dataframe
        df = pd.DataFrame({
            'date': date_range,
            'open': [p * (1 - np.random.uniform(0, 0.005)) for p in closes],
            'high': [p * (1 + np.random.uniform(0, 0.01)) for p in closes],
            'low': [p * (1 - np.random.uniform(0, 0.01)) for p in closes],
            'close': closes,
            'volume': np.random.randint(1000000, 10000000, size=len(date_range))
        })

        return df


# Parameters
tickers = ["AAPL", "MSFT", "GOOGL"]
start_date = "2020-01-01"
end_date = "2022-12-31"
initial_capital = 100000.0

print(f"\n{'=' * 50}")
print(f"Alpha-Agent Quick Start Example")
print(f"{'=' * 50}")
print(f"Ticker(s): {', '.join(tickers)}")
print(f"Period: {start_date} to {end_date}")
print(f"Initial capital: ${initial_capital:,.2f}")

try:
    # Initialize data fetcher with mock API client
    # In a real application, replace this with your actual API client
    api_client = SimpleMockAPIClient()
    data_fetcher = HistoricalDataFetcher(api_client=api_client)

    # Fetch historical data for our tickers
    print("\nFetching historical data...")
    stock_data = data_fetcher.fetch_stock_history(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date
    )

    # Fetch benchmark data
    benchmark_ticker = "SPY"
    benchmark_data = data_fetcher.fetch_stock_history(
        tickers=[benchmark_ticker],
        start_date=start_date,
        end_date=end_date
    )

    # Prepare data for backtesting
    print("Preparing data for backtesting...")
    dataset = {'stocks': stock_data}
    backtest_data = data_fetcher.prepare_backtest_data(dataset, format_type='panel')

    # Create a strategy
    strategy = MovingAverageCrossStrategy(fast_period=50, slow_period=200)
    print(f"Using strategy: {strategy.name} - {strategy.description}")

    # Initialize backtest engine
    backtest_engine = BacktestEngine(
        initial_capital=initial_capital,
        commission=0.001,  # 0.1% commission
        slippage=0.001  # 0.1% slippage
    )

    # Run the backtest
    print("\nRunning backtest...")
    results = backtest_engine.run_backtest(
        data=backtest_data,
        strategy=strategy.generate_signals
    )

    # Extract results
    portfolio_history = results['portfolio_history']
    trades = results['trades']
    metrics = results['metrics']

    # Convert portfolio history to proper format for analysis
    portfolio_df = pd.DataFrame(portfolio_history)
    portfolio_df['date'] = pd.to_datetime(portfolio_df['date'])
    portfolio_df.set_index('date', inplace=True)

    # Calculate returns series
    returns = portfolio_df['portfolio_value'].pct_change().dropna()

    # Prepare benchmark returns for comparison
    benchmark_returns = None
    if benchmark_ticker in benchmark_data:
        benchmark_df = benchmark_data[benchmark_ticker]
        benchmark_df['date'] = pd.to_datetime(benchmark_df['date'])
        benchmark_df.set_index('date', inplace=True)
        benchmark_returns = benchmark_df['close'].pct_change().dropna()

    # Calculate comprehensive performance metrics
    analyzer = PerformanceAnalyzer()
    performance_metrics = analyzer.calculate_metrics(
        returns=returns,
        benchmark_returns=benchmark_returns,
        risk_free_rate=0.02  # 2% annual risk-free rate
    )

    # Print results summary
    print(f"\n{'=' * 50}")
    print(f"BACKTEST RESULTS SUMMARY")
    print(f"{'=' * 50}")
    print(f"Final Portfolio Value: ${portfolio_df['portfolio_value'].iloc[-1]:,.2f}")
    print(f"Total Return: {metrics['total_return']:.2%}")
    print(f"Annualized Return: {metrics['annualized_return']:.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"Sortino Ratio: {performance_metrics.get('sortino_ratio', 'N/A'):.2f}")

    if benchmark_returns is not None and 'alpha' in performance_metrics and 'beta' in performance_metrics:
        print(f"\nBENCHMARK COMPARISON:")
        print(f"Alpha: {performance_metrics['alpha']:.4f}")
        print(f"Beta: {performance_metrics['beta']:.4f}")
        print(f"Correlation: {performance_metrics['correlation']:.4f}")

    # Print trade statistics
    if not trades.empty:
        print(f"\nTRADE STATISTICS:")
        print(f"Total Trades: {len(trades)}")

        buy_trades = trades[trades['type'] == 'buy']
        sell_trades = trades[trades['type'] == 'sell']
        print(f"Buy Trades: {len(buy_trades)}")
        print(f"Sell Trades: {len(sell_trades)}")

        if 'value' in trades.columns:
            winning_trades = trades[trades['value'] > 0]
            losing_trades = trades[trades['value'] < 0]
            print(f"Winning Trades: {len(winning_trades)}")
            print(f"Losing Trades: {len(losing_trades)}")
            print(f"Win Rate: {len(winning_trades) / len(trades):.2%}")

    # Plot portfolio performance
    print("\nGenerating performance chart...")
    output_path = os.path.join(output_dir, "quick_start_results.png")
    analyzer.plot_returns_analysis(
        returns=returns,
        benchmark_returns=benchmark_returns,
        save_path=output_path
    )

    # Also generate the built-in performance chart
    backtest_engine.plot_portfolio_performance(
        save_path=os.path.join(output_dir, "portfolio_performance.png")
    )

    print(f"\nResults visualizations saved to: {output_dir}")
    print(f"\nBacktest completed successfully!")

except Exception as e:
    logger.error(f"Error during backtest: {e}", exc_info=True)
    print(f"\nBacktest failed. See error details above.")

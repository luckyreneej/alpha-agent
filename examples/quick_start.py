#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Alpha-Agent Quick Start Example

This script demonstrates basic usage of the Alpha-Agent backtesting framework
with a simple moving average crossover strategy.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import logging

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

# Parameters
tickers = ["AAPL", "MSFT", "GOOGL"]
start_date = "2020-01-01"
end_date = "2022-12-31"
initial_capital = 100000.0

# Initialize components
data_fetcher = HistoricalDataFetcher()
strategy = MovingAverageCrossStrategy(fast_period=50, slow_period=200)
backtest_engine = BacktestEngine(
    initial_capital=initial_capital,
    data_fetcher=data_fetcher,
    strategy=strategy
)

print(f"\n{'='*50}")
print(f"Starting backtest with {strategy.name} strategy")
print(f"Ticker(s): {', '.join(tickers)}")
print(f"Period: {start_date} to {end_date}")
print(f"Initial capital: ${initial_capital:,.2f}")
print(f"{'='*50}\n")

# Run the backtest
try:
    results = backtest_engine.run(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        benchmark="SPY"  # Use SPY as benchmark
    )
    
    # Calculate performance metrics
    analyzer = PerformanceAnalyzer()
    metrics = analyzer.calculate_metrics(
        returns=results['returns'],
        benchmark_returns=results.get('benchmark_returns')
    )
    
    # Print results summary
    print(f"\n{'='*50}")
    print(f"BACKTEST RESULTS SUMMARY")
    print(f"{'='*50}")
    print(f"Final Portfolio Value: ${results['final_portfolio_value']:,.2f}")
    print(f"Total Return: {metrics['total_return']:.2%}")
    print(f"Annualized Return: {metrics['annualized_return']:.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"Sortino Ratio: {metrics['sortino_ratio']:.2f}")
    
    if 'alpha' in metrics and 'beta' in metrics:
        print(f"\nBENCHMARK COMPARISON:")
        print(f"Alpha: {metrics['alpha']:.4f}")
        print(f"Beta: {metrics['beta']:.4f}")
        print(f"Correlation: {metrics['correlation']:.4f}")
    
    # Plot the results
    output_path = os.path.join(output_dir, "quick_start_results.png")
    analyzer.plot_returns_analysis(
        returns=results['returns'],
        benchmark_returns=results.get('benchmark_returns'),
        save_path=output_path
    )
    
    # Print trade statistics
    trades = results.get('trades', [])
    if trades:
        print(f"\nTRADE STATISTICS:")
        print(f"Total Trades: {len(trades)}")
        
        buy_trades = [t for t in trades if t.get('type') == 'buy']
        sell_trades = [t for t in trades if t.get('type') == 'sell']
        print(f"Buy Trades: {len(buy_trades)}")
        print(f"Sell Trades: {len(sell_trades)}")
        
    print(f"\nResults visualization saved to: {output_path}")
    print(f"\nBacktest completed successfully!")
    
except Exception as e:
    logger.error(f"Error during backtest: {e}", exc_info=True)
    print(f"\nBacktest failed. See error details above.")

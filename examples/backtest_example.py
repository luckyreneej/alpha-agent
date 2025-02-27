#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Backtest Example Script

This example demonstrates how to use the Alpha-Agent backtesting framework
to evaluate trading strategies on historical stock data.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
import logging

# Ensure the alpha-agent package is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backtest.historical_data_fetcher import HistoricalDataFetcher
from backtest.backtest_engine import BacktestEngine
from backtest.strategy import MovingAverageCrossStrategy, RSIStrategy, MACDStrategy
from backtest.strategy import create_combined_strategy
from backtest.performance_metrics import PerformanceAnalyzer
from backtest.portfolio_optimizer import PortfolioOptimizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Backtest trading strategies')
    
    parser.add_argument('--tickers', type=str, default="SPY,AAPL,MSFT",
                        help='Comma-separated list of ticker symbols')
    parser.add_argument('--start-date', type=str, default="2020-01-01",
                        help='Start date for backtest (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default="2022-12-31",
                        help='End date for backtest (YYYY-MM-DD)')
    parser.add_argument('--capital', type=float, default=100000.0,
                        help='Initial capital amount')
    parser.add_argument('--strategy', type=str, default="ma_cross",
                        choices=['ma_cross', 'rsi', 'macd', 'combined'],
                        help='Strategy to backtest')
    parser.add_argument('--benchmark', type=str, default="SPY",
                        help='Benchmark ticker symbol')
    parser.add_argument('--optimize', action='store_true',
                        help='Use portfolio optimization')
    parser.add_argument('--commission', type=float, default=0.001,
                        help='Commission rate (as a fraction)')
    parser.add_argument('--output-dir', type=str, default="./results",
                        help='Directory for output files')
    
    return parser.parse_args()

def create_strategy(strategy_name):
    """Create a strategy instance based on the name."""
    if strategy_name == 'ma_cross':
        return MovingAverageCrossStrategy(fast_period=50, slow_period=200)
    elif strategy_name == 'rsi':
        return RSIStrategy(period=14, oversold=30, overbought=70)
    elif strategy_name == 'macd':
        return MACDStrategy(fast_period=12, slow_period=26, signal_period=9)
    elif strategy_name == 'combined':
        # Create a combined strategy with equal weights
        strategies = [
            MovingAverageCrossStrategy(fast_period=50, slow_period=200),
            RSIStrategy(period=14, oversold=30, overbought=70),
            MACDStrategy(fast_period=12, slow_period=26, signal_period=9)
        ]
        return create_combined_strategy(strategies)
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

def ensure_directory_exists(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def run_backtest(args):
    """Run backtest with the specified parameters."""
    logger.info(f"Starting backtest with {args.strategy} strategy")
    
    # Parse ticker list
    tickers = args.tickers.split(',')
    
    # Create output directory
    ensure_directory_exists(args.output_dir)
    
    # Initialize components
    data_fetcher = HistoricalDataFetcher()
    strategy = create_strategy(args.strategy)
    backtest = BacktestEngine(
        initial_capital=args.capital,
        data_fetcher=data_fetcher,
        strategy=strategy
    )
    
    # Run backtest
    logger.info(f"Running backtest for tickers: {tickers} from {args.start_date} to {args.end_date}")
    results = backtest.run(
        tickers=tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        commission=args.commission,
        benchmark=args.benchmark if args.benchmark else None
    )
    
    # Log basic results
    logger.info(f"Backtest completed.")
    logger.info(f"Final portfolio value: ${results['final_portfolio_value']:.2f}")
    logger.info(f"Total return: {results['total_return']:.2%}")
    
    # Analyze results
    analyzer = PerformanceAnalyzer()
    metrics = analyzer.calculate_metrics(
        returns=results['returns'],
        benchmark_returns=results.get('benchmark_returns')
    )
    
    # Print key metrics
    print("\nPerformance Metrics:")
    print(f"Total Return: {metrics['total_return']:.2%}")
    print(f"Annualized Return: {metrics['annualized_return']:.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    
    if 'alpha' in metrics:
        print(f"Alpha: {metrics['alpha']:.4f}")
        print(f"Beta: {metrics['beta']:.4f}")
    
    # Generate performance charts
    output_path = os.path.join(args.output_dir, f"{args.strategy}_performance.png")
    analyzer.plot_returns_analysis(
        returns=results['returns'],
        benchmark_returns=results.get('benchmark_returns'),
        save_path=output_path
    )
    logger.info(f"Performance chart saved to {output_path}")
    
    # Portfolio optimization (if requested)
    if args.optimize and len(tickers) > 1:
        optimize_portfolio(results, tickers, args.output_dir)
    
    return results, metrics

def optimize_portfolio(results, tickers, output_dir):
    """Perform portfolio optimization."""
    logger.info("Performing portfolio optimization")
    
    # Extract individual asset returns
    assets_data = results.get('asset_returns', {})
    if not assets_data or len(assets_data) <= 1:
        logger.warning("Insufficient asset data for portfolio optimization")
        return
    
    # Create a DataFrame with returns for each asset
    returns_df = pd.DataFrame()
    for ticker, returns in assets_data.items():
        if isinstance(returns, pd.Series):
            returns_df[ticker] = returns
    
    if len(returns_df.columns) <= 1:
        logger.warning("Not enough assets with return data for optimization")
        return
    
    # Initialize optimizer
    optimizer = PortfolioOptimizer(risk_free_rate=0.02)  # Assuming 2% risk-free rate
    
    # Run different optimization methods
    optimization_methods = ['equal_weight', 'min_variance', 'max_sharpe']
    results_dict = {}
    
    for method in optimization_methods:
        result = optimizer.optimize(returns_df, method=method)
        results_dict[method] = result
        
        # Print optimization results
        print(f"\n{method.upper()} Portfolio:")
        print("Weights:")
        for asset, weight in result['weights'].items():
            print(f"  {asset}: {weight:.4f}")
        print(f"Expected Return: {result['metrics']['return']:.2%}")
        print(f"Volatility: {result['metrics']['volatility']:.2%}")
        print(f"Sharpe Ratio: {result['metrics']['sharpe_ratio']:.4f}")
    
    # Generate efficient frontier
    ef_df = optimizer.efficient_frontier(returns_df, points=30)
    
    # Plot efficient frontier and optimal portfolios
    plt.figure(figsize=(10, 6))
    plt.scatter(ef_df['volatility'], ef_df['return'], c=ef_df['sharpe_ratio'],
                cmap='viridis', s=10, alpha=0.7)
    
    # Plot optimization results
    for method, result in results_dict.items():
        plt.scatter(
            result['metrics']['volatility'],
            result['metrics']['return'],
            marker='*',
            s=200,
            label=f"{method.replace('_', ' ').title()}"
        )
    
    plt.colorbar(label='Sharpe Ratio')
    plt.xlabel('Annualized Volatility')
    plt.ylabel('Annualized Return')
    plt.title('Efficient Frontier and Optimal Portfolios')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    output_path = os.path.join(output_dir, "portfolio_optimization.png")
    plt.savefig(output_path)
    logger.info(f"Portfolio optimization chart saved to {output_path}")

def main():
    args = parse_arguments()
    results, metrics = run_backtest(args)
    
    print("\nBacktest completed successfully!")
    print(f"Results saved in {args.output_dir}")

if __name__ == "__main__":
    main()

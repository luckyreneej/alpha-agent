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


# Mock API client for demonstration
class MockAPIClient:
    """Mock API client that returns sample stock data for testing."""

    def get_stock_bars(self, ticker, timespan, from_date, to_date):
        """Generate mock stock price data for testing."""
        # Create date range
        date_range = pd.date_range(start=from_date, end=to_date, freq='B')

        # Generate mock price data with some randomness
        np.random.seed(42)  # for reproducibility

        # Base price and volatility vary by ticker
        if ticker == 'SPY':
            base_price = 400
            volatility = 0.01
        elif ticker == 'AAPL':
            base_price = 150
            volatility = 0.015
        elif ticker == 'MSFT':
            base_price = 300
            volatility = 0.012
        else:
            base_price = 200
            volatility = 0.02

        # Generate price series with random walk
        closes = [base_price]
        for _ in range(1, len(date_range)):
            prev_close = closes[-1]
            change = np.random.normal(0, volatility)
            new_close = prev_close * (1 + change)
            closes.append(new_close)

        # Create DataFrame
        df = pd.DataFrame({
            'date': date_range,
            'open': closes,
            'high': [c * (1 + np.random.uniform(0, 0.01)) for c in closes],
            'low': [c * (1 - np.random.uniform(0, 0.01)) for c in closes],
            'close': closes,
            'volume': np.random.randint(1000000, 10000000, size=len(date_range))
        })

        return df


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

    # Initialize API client and data fetcher
    api_client = MockAPIClient()  # In production, use your actual API client
    data_fetcher = HistoricalDataFetcher(api_client=api_client)

    # Fetch historical data
    logger.info(f"Fetching historical data for {tickers} from {args.start_date} to {args.end_date}")
    stock_data = data_fetcher.fetch_stock_history(
        tickers=tickers,
        start_date=args.start_date,
        end_date=args.end_date
    )

    # Create dataset and prepare for backtesting
    dataset = {'stocks': stock_data}
    backtest_data = data_fetcher.prepare_backtest_data(dataset, format_type='panel')

    # Add benchmark data for comparison if requested
    benchmark_data = None
    if args.benchmark and args.benchmark not in tickers:
        benchmark_stock_data = data_fetcher.fetch_stock_history(
            tickers=[args.benchmark],
            start_date=args.start_date,
            end_date=args.end_date
        )
        if benchmark_stock_data:
            benchmark_data = benchmark_stock_data[args.benchmark]

    # Create strategy
    strategy = create_strategy(args.strategy)
    logger.info(f"Created {strategy.name} strategy: {strategy.description}")

    # Initialize backtest engine
    backtest_engine = BacktestEngine(
        initial_capital=args.capital,
        commission=args.commission,
        slippage=args.commission  # Using same value for slippage as commission for simplicity
    )

    # Run backtest
    results = backtest_engine.run_backtest(
        data=backtest_data,
        strategy=strategy.generate_signals
    )

    # Extract results
    portfolio_history = results['portfolio_history']
    trades = results['trades']
    metrics = results['metrics']

    # Calculate returns as a series for further analysis
    portfolio_history['date'] = pd.to_datetime(portfolio_history['date'])
    portfolio_history.set_index('date', inplace=True)
    returns = portfolio_history['portfolio_value'].pct_change().dropna()

    # Create benchmark returns if available
    benchmark_returns = None
    if benchmark_data is not None:
        benchmark_data.set_index('date', inplace=True)
        benchmark_returns = benchmark_data['close'].pct_change().dropna()

    # Analyze results with PerformanceAnalyzer
    analyzer = PerformanceAnalyzer()
    performance_metrics = analyzer.calculate_metrics(
        returns=returns,
        benchmark_returns=benchmark_returns,
        risk_free_rate=0.02  # 2% annual risk-free rate
    )

    # Print key metrics
    print("\nPerformance Metrics:")
    print(f"Total Return: {metrics['total_return']:.2%}")
    print(f"Annualized Return: {metrics['annualized_return']:.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"Sortino Ratio: {performance_metrics.get('sortino_ratio', 'N/A')}")

    if benchmark_returns is not None and 'alpha' in performance_metrics:
        print(f"Alpha: {performance_metrics['alpha']:.4f}")
        print(f"Beta: {performance_metrics['beta']:.4f}")

    # Generate performance charts
    output_path = os.path.join(args.output_dir, f"{args.strategy}_performance.png")
    analyzer.plot_returns_analysis(
        returns=returns,
        benchmark_returns=benchmark_returns,
        save_path=output_path
    )
    logger.info(f"Performance chart saved to {output_path}")

    # Portfolio optimization (if requested)
    if args.optimize and len(tickers) > 1:
        # Create asset returns for optimization
        asset_returns = {}
        for ticker in tickers:
            ticker_data = stock_data[ticker]
            ticker_returns = ticker_data['close'].pct_change().dropna()
            asset_returns[ticker] = ticker_returns

        optimize_portfolio(asset_returns, tickers, args.output_dir)

    # Combined results for return
    combined_results = {
        'portfolio_history': portfolio_history,
        'trades': trades,
        'metrics': metrics,
        'returns': returns,
        'benchmark_returns': benchmark_returns,
        'performance_metrics': performance_metrics,
        'final_portfolio_value': portfolio_history['portfolio_value'].iloc[-1]
    }

    return combined_results


def optimize_portfolio(asset_returns, tickers, output_dir):
    """Perform portfolio optimization."""
    logger.info("Performing portfolio optimization")

    # Check if we have enough assets for optimization
    if len(asset_returns) <= 1:
        logger.warning("Insufficient asset data for portfolio optimization")
        return

    # Create a DataFrame with returns for each asset
    returns_df = pd.DataFrame(asset_returns)

    if returns_df.empty or len(returns_df.columns) <= 1:
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
    results = run_backtest(args)

    print("\nBacktest completed successfully!")
    print(f"Final Portfolio Value: ${results['final_portfolio_value']:.2f}")
    print(f"Results saved in {args.output_dir}")


if __name__ == "__main__":
    main()

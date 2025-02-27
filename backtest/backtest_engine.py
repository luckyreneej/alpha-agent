import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import json
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BacktestEngine:
    """
    Backtesting engine for evaluating trading strategies using historical data.
    Supports multiple asset types, position sizing, and detailed performance metrics.
    """
    
    def __init__(self, 
                 initial_capital: float = 100000.0,
                 commission: float = 0.001,  # 10 basis points
                 slippage: float = 0.001,   # 10 basis points
                 lot_size: int = 100,
                 leverage: float = 1.0,
                 data_frequency: str = 'daily',
                 risk_free_rate: float = 0.02):  # 2% annual risk-free rate
        """
        Initialize the backtesting engine.
        
        Args:
            initial_capital: Initial portfolio value
            commission: Commission rate per trade (percentage)
            slippage: Slippage per trade (percentage)
            lot_size: Size of one trading lot
            leverage: Maximum leverage allowed
            data_frequency: Frequency of data ('daily', 'hourly', etc.)
            risk_free_rate: Annual risk-free rate for Sharpe ratio calculation
        """
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.lot_size = lot_size
        self.leverage = leverage
        self.data_frequency = data_frequency
        self.risk_free_rate = risk_free_rate
        
        # Portfolio tracking
        self.positions = {}  # Current positions {symbol: quantity}
        self.trades = []     # List of executed trades
        self.portfolio_history = []  # Daily portfolio values
        self.returns_history = []    # Daily returns
        
        # Performance metrics
        self.metrics = {}
    
    def reset(self):
        """
        Reset the backtesting engine to initial state.
        """
        self.capital = self.initial_capital
        self.positions = {}
        self.trades = []
        self.portfolio_history = []
        self.returns_history = []
        self.metrics = {}
    
    def run_backtest(self, 
                    data: pd.DataFrame, 
                    strategy: Callable, 
                    strategy_params: Optional[Dict[str, Any]] = None):
        """
        Run a backtest on the provided data using the specified strategy.
        
        Args:
            data: DataFrame with historical price data (must include 'date', 'open', 'high', 'low', 'close', 'volume')
            strategy: Function that generates trading signals
            strategy_params: Additional parameters for the strategy function
            
        Returns:
            Dictionary with backtest results and performance metrics
        """
        if strategy_params is None:
            strategy_params = {}
        
        # Ensure data is sorted by date
        if 'date' not in data.columns:
            raise ValueError("Data must include a 'date' column")
            
        data = data.sort_values('date')
        
        # Reset the backtest state
        self.reset()
        
        # Initial portfolio value
        self.portfolio_history.append({
            'date': data['date'].iloc[0],
            'portfolio_value': self.capital,
            'cash': self.capital,
            'positions_value': 0.0
        })
        
        # Main backtest loop
        for i in range(1, len(data)):
            current_day = data.iloc[i]
            previous_day = data.iloc[i-1]
            
            # Calculate current portfolio value before applying new trades
            portfolio_value_pre = self._calculate_portfolio_value(current_day)
            
            # Get strategy signals for the current day
            signals = strategy(data.iloc[:i], current_day, **strategy_params)
            
            # Execute trades based on signals
            for symbol, signal in signals.items():
                if signal != 0 and symbol in current_day:
                    self._execute_trade(symbol, signal, current_day)
            
            # Calculate portfolio value after trades
            portfolio_value_post = self._calculate_portfolio_value(current_day)
            
            # Calculate daily return
            daily_return = (portfolio_value_post / portfolio_value_pre) - 1 if portfolio_value_pre > 0 else 0
            self.returns_history.append(daily_return)
            
            # Update portfolio history
            self.portfolio_history.append({
                'date': current_day['date'],
                'portfolio_value': portfolio_value_post,
                'cash': self.capital,
                'positions_value': portfolio_value_post - self.capital
            })
        
        # Calculate performance metrics
        self._calculate_metrics()
        
        return {
            'portfolio_history': pd.DataFrame(self.portfolio_history),
            'trades': pd.DataFrame(self.trades) if self.trades else pd.DataFrame(),
            'metrics': self.metrics
        }
    
    def _execute_trade(self, symbol: str, signal: float, day_data: pd.Series):
        """
        Execute a trade based on the signal.
        
        Args:
            symbol: Trading symbol
            signal: Trading signal (-1.0 to 1.0 where -1 is full short, 1 is full long)
            day_data: Current day's price data
        """
        # Determine current position and price
        current_position = self.positions.get(symbol, 0)
        current_price = day_data[f'{symbol}_close'] if f'{symbol}_close' in day_data else day_data['close']
        
        # Calculate target position based on signal and available capital
        # Signal is between -1 and 1, indicating position size as percentage of portfolio
        max_position_value = self.capital * self.leverage
        target_position_value = max_position_value * signal
        target_position = int(target_position_value / current_price / self.lot_size) * self.lot_size
        
        # Calculate position delta
        position_delta = target_position - current_position
        
        if position_delta == 0:
            return  # No trade needed
        
        # Calculate transaction costs
        price_with_slippage = current_price * (1 + self.slippage) if position_delta > 0 else current_price * (1 - self.slippage)
        trade_value = abs(position_delta) * price_with_slippage
        commission_cost = trade_value * self.commission
        
        # Check if we have enough capital
        if position_delta > 0 and trade_value + commission_cost > self.capital:
            # Adjust position based on available capital
            affordable_delta = int((self.capital - commission_cost) / price_with_slippage / self.lot_size) * self.lot_size
            if affordable_delta <= 0:
                return  # Can't afford any shares
            position_delta = affordable_delta
            trade_value = position_delta * price_with_slippage
            commission_cost = trade_value * self.commission
        
        # Execute the trade
        self.capital -= position_delta * price_with_slippage  # Subtract cost for buys, add proceeds for sells
        self.capital -= commission_cost  # Subtract commission
        
        # Update position
        if symbol not in self.positions:
            self.positions[symbol] = position_delta
        else:
            self.positions[symbol] += position_delta
        
        # Remove position if quantity is zero
        if self.positions[symbol] == 0:
            del self.positions[symbol]
        
        # Record the trade
        self.trades.append({
            'date': day_data['date'],
            'symbol': symbol,
            'quantity': position_delta,
            'price': price_with_slippage,
            'value': trade_value,
            'commission': commission_cost,
            'type': 'buy' if position_delta > 0 else 'sell'
        })
    
    def _calculate_portfolio_value(self, day_data: pd.Series) -> float:
        """
        Calculate the current portfolio value including cash and positions.
        
        Args:
            day_data: Current day's price data
            
        Returns:
            Total portfolio value
        """
        positions_value = 0.0
        
        for symbol, quantity in self.positions.items():
            symbol_price = day_data[f'{symbol}_close'] if f'{symbol}_close' in day_data else day_data['close']
            positions_value += quantity * symbol_price
        
        return self.capital + positions_value
    
    def _calculate_metrics(self):
        """
        Calculate performance metrics for the backtest.
        """
        if not self.portfolio_history or len(self.portfolio_history) < 2:
            logger.warning("Insufficient data for metrics calculation")
            return
        
        # Convert portfolio history to DataFrame for analysis
        portfolio_df = pd.DataFrame(self.portfolio_history)
        portfolio_df['date'] = pd.to_datetime(portfolio_df['date'])
        portfolio_df.set_index('date', inplace=True)
        
        # Calculate returns
        portfolio_df['return'] = portfolio_df['portfolio_value'].pct_change()
        portfolio_df['cum_return'] = (1 + portfolio_df['return']).cumprod() - 1
        
        # Trading period in years
        start_date = portfolio_df.index[0]
        end_date = portfolio_df.index[-1]
        trading_days = (end_date - start_date).days
        years = trading_days / 365.25
        
        # Basic metrics
        total_return = (portfolio_df['portfolio_value'].iloc[-1] / self.initial_capital) - 1
        self.metrics['total_return'] = total_return
        self.metrics['annualized_return'] = (1 + total_return) ** (1 / max(years, 0.01)) - 1 if years > 0 else 0
        
        # Calculate drawdown
        portfolio_df['peak'] = portfolio_df['portfolio_value'].cummax()
        portfolio_df['drawdown'] = (portfolio_df['portfolio_value'] - portfolio_df['peak']) / portfolio_df['peak']
        self.metrics['max_drawdown'] = portfolio_df['drawdown'].min()
        
        # Risk metrics
        returns = portfolio_df['return'].dropna().values
        self.metrics['volatility'] = np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0
        
        # Sharpe ratio
        excess_returns = returns - (self.risk_free_rate / 252)  # Daily excess returns
        sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if np.std(excess_returns) > 0 else 0
        self.metrics['sharpe_ratio'] = sharpe
        
        # Sortino ratio (using downside deviation)
        downside_returns = returns[returns < 0]
        downside_deviation = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino = np.mean(excess_returns) / downside_deviation * np.sqrt(252) if downside_deviation > 0 else 0
        self.metrics['sortino_ratio'] = sortino
        
        # Calmar ratio (annualized return / max drawdown)
        calmar = abs(self.metrics['annualized_return'] / self.metrics['max_drawdown']) if self.metrics['max_drawdown'] != 0 else 0
        self.metrics['calmar_ratio'] = calmar
        
        # Trading metrics
        self.metrics['total_trades'] = len(self.trades)
        
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            self.metrics['win_rate'] = len(trades_df[trades_df['value'] > 0]) / len(trades_df) if len(trades_df) > 0 else 0
            self.metrics['profit_factor'] = abs(trades_df[trades_df['value'] > 0]['value'].sum() / trades_df[trades_df['value'] < 0]['value'].sum()) if trades_df[trades_df['value'] < 0]['value'].sum() != 0 else 0
            self.metrics['avg_trade_pnl'] = trades_df['value'].mean()
            self.metrics['max_trade_pnl'] = trades_df['value'].max()
            self.metrics['min_trade_pnl'] = trades_df['value'].min()
            self.metrics['total_commission'] = trades_df['commission'].sum()
    
    def plot_portfolio_performance(self, benchmark_data=None, save_path=None):
        """
        Plot portfolio performance, returns, and drawdowns.
        
        Args:
            benchmark_data: Optional DataFrame with benchmark performance data
            save_path: Path to save the plot (if None, display the plot)
        """
        if not self.portfolio_history:
            logger.warning("No portfolio history to plot")
            return
        
        portfolio_df = pd.DataFrame(self.portfolio_history)
        portfolio_df['date'] = pd.to_datetime(portfolio_df['date'])
        portfolio_df.set_index('date', inplace=True)
        
        # Create the plot with 3 subplots
        fig, axes = plt.subplots(3, 1, figsize=(12, 15), gridspec_kw={'height_ratios': [3, 1, 1]})
        
        # Plot portfolio value
        axes[0].plot(portfolio_df.index, portfolio_df['portfolio_value'], label='Portfolio', linewidth=2)
        if benchmark_data is not None:
            # Normalize benchmark to same starting value
            benchmark_scaled = benchmark_data['close'] * (self.initial_capital / benchmark_data['close'].iloc[0])
            axes[0].plot(benchmark_data.index, benchmark_scaled, label='Benchmark', linewidth=1, alpha=0.7)
        
        axes[0].set_title('Portfolio Value Over Time', fontsize=14)
        axes[0].set_ylabel('Value ($)', fontsize=12)
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot returns
        returns = portfolio_df['portfolio_value'].pct_change()
        axes[1].plot(portfolio_df.index[1:], returns[1:], label='Daily Returns', linewidth=1)
        axes[1].set_title('Daily Returns', fontsize=14)
        axes[1].set_ylabel('Return (%)', fontsize=12)
        axes[1].grid(True)
        
        # Plot drawdowns
        peak = portfolio_df['portfolio_value'].cummax()
        drawdown = (portfolio_df['portfolio_value'] - peak) / peak
        axes[2].fill_between(portfolio_df.index, drawdown, 0, alpha=0.3, color='red', label='Drawdown')
        axes[2].set_title('Drawdown', fontsize=14)
        axes[2].set_ylabel('Drawdown (%)', fontsize=12)
        axes[2].set_ylim(min(drawdown) * 1.1, 0.01)
        axes[2].grid(True)
        
        # Format x-axis dates
        for ax in axes:
            ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Add performance metrics as text
        textstr = f"Total Return: {self.metrics['total_return']:.2%}\n"
        textstr += f"Annualized Return: {self.metrics['annualized_return']:.2%}\n"
        textstr += f"Sharpe Ratio: {self.metrics['sharpe_ratio']:.2f}\n"
        textstr += f"Max Drawdown: {self.metrics['max_drawdown']:.2%}"
        
        axes[0].text(0.02, 0.95, textstr, transform=axes[0].transAxes,
                     fontsize=10, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close(fig)
        else:
            plt.show()
    
    def plot_trade_analysis(self, save_path=None):
        """
        Plot trade analysis charts (trade distribution, equity curve, etc.)
        
        Args:
            save_path: Path to save the plot (if None, display the plot)
        """
        if not self.trades:
            logger.warning("No trades to analyze")
            return
        
        trades_df = pd.DataFrame(self.trades)
        trades_df['date'] = pd.to_datetime(trades_df['date'])
        
        # Calculate PnL per trade
        trades_df['pnl'] = np.where(trades_df['type'] == 'buy', 0, trades_df['value'])
        
        # Create the plot with 2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Plot trade PnL distribution
        sns.histplot(trades_df['pnl'], kde=True, ax=axes[0, 0])
        axes[0, 0].set_title('Trade PnL Distribution', fontsize=14)
        axes[0, 0].set_xlabel('PnL ($)', fontsize=12)
        axes[0, 0].axvline(x=0, color='r', linestyle='--', alpha=0.7)
        
        # Plot cumulative PnL
        trades_df['cum_pnl'] = trades_df['pnl'].cumsum()
        trades_df.plot(x='date', y='cum_pnl', ax=axes[0, 1], legend=False)
        axes[0, 1].set_title('Cumulative PnL', fontsize=14)
        axes[0, 1].set_ylabel('Cumulative PnL ($)', fontsize=12)
        
        # Plot trade size distribution
        sns.histplot(trades_df['quantity'].abs(), kde=True, ax=axes[1, 0])
        axes[1, 0].set_title('Trade Size Distribution', fontsize=14)
        axes[1, 0].set_xlabel('Trade Size (shares)', fontsize=12)
        
        # Plot trade count by symbol
        if 'symbol' in trades_df.columns:
            symbol_counts = trades_df['symbol'].value_counts()
            symbol_counts.plot(kind='bar', ax=axes[1, 1])
            axes[1, 1].set_title('Trades by Symbol', fontsize=14)
            axes[1, 1].set_ylabel('Number of Trades', fontsize=12)
            plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45)
        else:
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close(fig)
        else:
            plt.show()
    
    def export_results(self, output_dir: str):
        """
        Export backtest results to files for further analysis.
        
        Args:
            output_dir: Directory to save output files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Export portfolio history to CSV
        if self.portfolio_history:
            portfolio_df = pd.DataFrame(self.portfolio_history)
            portfolio_df.to_csv(os.path.join(output_dir, 'portfolio_history.csv'), index=False)
        
        # Export trades to CSV
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            trades_df.to_csv(os.path.join(output_dir, 'trades.csv'), index=False)
        
        # Export metrics to JSON
        with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
            json.dump(self.metrics, f, indent=4)
        
        logger.info(f"Backtest results exported to {output_dir}")
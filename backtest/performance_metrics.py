import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import Dict, List, Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PerformanceAnalyzer:
    """
    Analyzes trading performance and calculates various metrics.
    """

    def calculate_metrics(self,
                        trades: List[Dict],
                        portfolio_history: List[Dict],
                        benchmark_data: Optional[pd.DataFrame] = None) -> Dict:
        """
        Calculate performance metrics from trading history.

        Args:
            trades: List of trade dictionaries
            portfolio_history: List of portfolio state dictionaries
            benchmark_data: Optional benchmark data for comparison

        Returns:
            Dictionary containing performance metrics
        """
        if not portfolio_history:
            return {
                'error': 'No portfolio history available for analysis'
            }

        # Convert portfolio history to DataFrame
        portfolio_df = pd.DataFrame(portfolio_history)
        portfolio_df.set_index('timestamp', inplace=True)

        # Calculate returns
        portfolio_returns = self._calculate_returns(portfolio_df['portfolio_value'])
        
        # Calculate benchmark returns if available
        benchmark_returns = None
        if benchmark_data is not None:
            benchmark_returns = self._calculate_returns(benchmark_data['close'])

        # Calculate metrics
        metrics = {
            'total_return': self._calculate_total_return(portfolio_df),
            'annualized_return': self._calculate_annualized_return(portfolio_returns),
            'sharpe_ratio': self._calculate_sharpe_ratio(portfolio_returns),
            'max_drawdown': self._calculate_max_drawdown(portfolio_df),
            'volatility': self._calculate_volatility(portfolio_returns),
            'trade_metrics': self._calculate_trade_metrics(trades),
        }

        # Add benchmark comparison if available
        if benchmark_returns is not None:
            metrics.update({
                'benchmark_return': self._calculate_total_return(benchmark_data),
                'alpha': self._calculate_alpha(portfolio_returns, benchmark_returns),
                'beta': self._calculate_beta(portfolio_returns, benchmark_returns),
            })

        return metrics

    def _calculate_returns(self, values: pd.Series) -> pd.Series:
        """Calculate percentage returns from a series of values."""
        return values.pct_change().fillna(0)

    def _calculate_total_return(self, df: pd.DataFrame) -> float:
        """Calculate total return percentage."""
        if len(df) < 2:
            return 0.0
        initial_value = df.iloc[0]['portfolio_value']
        final_value = df.iloc[-1]['portfolio_value']
        return (final_value - initial_value) / initial_value * 100

    def _calculate_annualized_return(self, returns: pd.Series) -> float:
        """Calculate annualized return percentage."""
        if len(returns) < 2:
            return 0.0
        
        # Assuming daily data
        days = len(returns)
        total_return = (1 + returns).prod()
        annualized_return = (total_return ** (252/days) - 1) * 100
        return annualized_return

    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) < 2:
            return 0.0
        
        # Assuming daily data, annualize
        excess_returns = returns - risk_free_rate/252
        return np.sqrt(252) * excess_returns.mean() / returns.std()

    def _calculate_max_drawdown(self, df: pd.DataFrame) -> float:
        """Calculate maximum drawdown percentage."""
        if len(df) < 2:
            return 0.0
            
        portfolio_values = df['portfolio_value']
        rolling_max = portfolio_values.expanding().max()
        drawdowns = (portfolio_values - rolling_max) / rolling_max * 100
        return abs(drawdowns.min())

    def _calculate_volatility(self, returns: pd.Series) -> float:
        """Calculate annualized volatility."""
        if len(returns) < 2:
            return 0.0
        return returns.std() * np.sqrt(252) * 100

    def _calculate_trade_metrics(self, trades: List[Dict]) -> Dict:
        """Calculate trade-specific metrics."""
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'avg_profit': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0
            }

        # Convert trades to DataFrame
        trades_df = pd.DataFrame(trades)
        
        # Calculate profits/losses
        trades_df['pnl'] = trades_df.apply(
            lambda x: x['proceeds'] - x['cost'] if x['action'] == 'sell' else 0,
            axis=1
        )
        
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] < 0]
        
        metrics = {
            'total_trades': len(trades_df[trades_df['action'] == 'sell']),
            'win_rate': len(winning_trades) / len(trades_df) * 100 if len(trades_df) > 0 else 0,
            'avg_profit': winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0,
            'avg_loss': abs(losing_trades['pnl'].mean()) if len(losing_trades) > 0 else 0,
            'profit_factor': (
                abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum())
                if len(losing_trades) > 0 and losing_trades['pnl'].sum() != 0
                else 0
            )
        }
        
        return metrics

    def _calculate_alpha(self, portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate Jensen's Alpha."""
        if len(portfolio_returns) < 2 or len(benchmark_returns) < 2:
            return 0.0
            
        # Assuming daily data, annualize
        risk_free_rate = 0.02  # Assuming 2% risk-free rate
        beta = self._calculate_beta(portfolio_returns, benchmark_returns)
        
        portfolio_return = (1 + portfolio_returns).prod() ** (252/len(portfolio_returns)) - 1
        benchmark_return = (1 + benchmark_returns).prod() ** (252/len(benchmark_returns)) - 1
        
        alpha = portfolio_return - (risk_free_rate + beta * (benchmark_return - risk_free_rate))
        return alpha * 100

    def _calculate_beta(self, portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate portfolio beta."""
        if len(portfolio_returns) < 2 or len(benchmark_returns) < 2:
            return 0.0
            
        # Align dates
        aligned_returns = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
        
        if len(aligned_returns) < 2:
            return 0.0
            
        covariance = aligned_returns.iloc[:, 0].cov(aligned_returns.iloc[:, 1])
        variance = aligned_returns.iloc[:, 1].var()
        
        return covariance / variance if variance != 0 else 0.0

    def plot_returns_analysis(self, returns: pd.Series, benchmark_returns: Optional[pd.Series] = None,
                              save_path: Optional[str] = None):
        """
        Create comprehensive return analysis plot.

        Args:
            returns: Series of returns
            benchmark_returns: Optional benchmark returns for comparison
            save_path: Path to save the plot (if None, display the plot)
        """
        if returns.empty:
            logger.warning("Cannot plot returns analysis: empty returns data")
            return

        # Calculate cumulative returns
        cumulative_returns = (1 + returns).cumprod()

        # Create figure with 2 subplots
        fig, axes = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})

        # Plot cumulative returns
        cumulative_returns.plot(ax=axes[0], color='blue', linewidth=2, label='Strategy')

        if benchmark_returns is not None and not benchmark_returns.empty:
            # Align benchmark returns with strategy returns
            benchmark_returns = benchmark_returns.reindex(returns.index, method='ffill')
            cumulative_benchmark = (1 + benchmark_returns.fillna(0)).cumprod()
            cumulative_benchmark.plot(ax=axes[0], color='gray', linewidth=1.5, label='Benchmark', alpha=0.7)

        axes[0].set_title('Cumulative Returns', fontsize=14)
        axes[0].set_ylabel('Growth of $1')
        axes[0].legend()
        axes[0].grid(True)

        # Display key metrics on the plot
        metrics = self.calculate_metrics(returns, benchmark_returns)
        metrics_text = (f"Total Return: {metrics['total_return']:.2%}\n"
                        f"Ann. Return: {metrics['annualized_return']:.2%}\n"
                        f"Volatility: {metrics['volatility']:.2%}\n"
                        f"Sharpe: {metrics['sharpe_ratio']:.2f}\n"
                        f"Max DD: {metrics['max_drawdown']:.2%}")

        # Add text box with metrics
        axes[0].text(0.02, 0.97, metrics_text, transform=axes[0].transAxes,
                     fontsize=10, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Plot drawdown chart
        cumulative = (1 + returns).cumprod()
        peak = cumulative.expanding(min_periods=1).max()
        drawdown = (cumulative / peak) - 1

        drawdown.plot(ax=axes[1], color='red', alpha=0.5, linewidth=1, label='Drawdown')
        axes[1].fill_between(drawdown.index, drawdown.values, 0, color='red', alpha=0.3)
        axes[1].set_title('Drawdown', fontsize=14)
        axes[1].set_ylabel('Drawdown')
        axes[1].set_ylim(min(drawdown.min() * 1.1, -0.01), 0.01)  # Ensure y-axis includes 0
        axes[1].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            plt.close(fig)
        else:
            plt.show()

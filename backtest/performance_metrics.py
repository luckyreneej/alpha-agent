import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import Dict, Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PerformanceAnalyzer:
    """
    Performance analysis tool for backtesting results.
    Calculates standard performance metrics and generates visualization.
    """

    def __init__(self, annualization_factor: int = 252):
        """
        Initialize the performance analyzer.

        Args:
            annualization_factor: Factor for annualizing returns (252 for daily, 52 for weekly, 12 for monthly)
        """
        self.annualization_factor = annualization_factor

    def calculate_metrics(self, returns: pd.Series, benchmark_returns: Optional[pd.Series] = None,
                          risk_free_rate: float = 0.0) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics.

        Args:
            returns: Series of returns
            benchmark_returns: Optional benchmark returns for comparison
            risk_free_rate: Annual risk-free rate

        Returns:
            Dictionary of performance metrics
        """
        if returns.empty:
            logger.warning("Empty returns series provided")
            return {}

        metrics = {}

        # Basic return metrics
        metrics['total_return'] = self._calculate_total_return(returns)
        metrics['annualized_return'] = self._calculate_annualized_return(returns)
        metrics['volatility'] = self._calculate_volatility(returns)

        # Risk-adjusted metrics
        metrics['sharpe_ratio'] = self._calculate_sharpe_ratio(returns, risk_free_rate)
        metrics['sortino_ratio'] = self._calculate_sortino_ratio(returns, risk_free_rate)
        metrics['max_drawdown'] = self._calculate_max_drawdown(returns)
        metrics['calmar_ratio'] = self._calculate_calmar_ratio(returns)

        # Information metrics (if benchmark provided)
        if benchmark_returns is not None and not benchmark_returns.empty:
            # Align benchmark returns with strategy returns
            aligned_returns, aligned_benchmark = returns.align(benchmark_returns, join='inner')

            if not aligned_returns.empty and len(aligned_returns) > 1:
                metrics['beta'] = self._calculate_beta(aligned_returns, aligned_benchmark)
                metrics['alpha'] = self._calculate_alpha(aligned_returns, aligned_benchmark, risk_free_rate)
                metrics['correlation'] = aligned_returns.corr(aligned_benchmark)

        return metrics

    def calculate_trade_metrics(self, trades_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate trading-specific performance metrics.

        Args:
            trades_df: DataFrame with trade information (must include 'value', 'type')

        Returns:
            Dictionary of trade metrics
        """
        metrics = {}

        if trades_df.empty:
            return {"total_trades": 0}

        # Basic trade metrics
        metrics['total_trades'] = len(trades_df)

        # Calculate buy and sell trades
        if 'type' in trades_df.columns:
            buy_trades = trades_df[trades_df['type'] == 'buy']
            sell_trades = trades_df[trades_df['type'] == 'sell']
            metrics['buy_trades'] = len(buy_trades)
            metrics['sell_trades'] = len(sell_trades)

        # Win/loss analysis
        if 'value' in trades_df.columns:
            winning_trades = trades_df[trades_df['value'] > 0]
            losing_trades = trades_df[trades_df['value'] < 0]

            metrics['winning_trades'] = len(winning_trades)
            metrics['losing_trades'] = len(losing_trades)
            metrics['win_rate'] = len(winning_trades) / len(trades_df) if len(trades_df) > 0 else 0

            # Calculate profit metrics
            if not winning_trades.empty:
                metrics['avg_win'] = winning_trades['value'].mean()
                metrics['max_win'] = winning_trades['value'].max()

            if not losing_trades.empty:
                metrics['avg_loss'] = losing_trades['value'].mean()
                metrics['max_loss'] = losing_trades['value'].min()  # Note: this is negative

            # Calculate profit factor (ratio of gross profits to gross losses)
            total_win = winning_trades['value'].sum() if len(winning_trades) > 0 else 0
            total_loss = abs(losing_trades['value'].sum()) if len(losing_trades) > 0 else 0
            metrics['profit_factor'] = total_win / total_loss if total_loss > 0 else float('inf')

            # Net profit
            metrics['net_profit'] = trades_df['value'].sum()

        # Calculate commission costs if available
        if 'commission' in trades_df.columns:
            metrics['total_commission'] = trades_df['commission'].sum()

        return metrics

    def _calculate_total_return(self, returns: pd.Series) -> float:
        """Calculate the total return."""
        return (1 + returns).prod() - 1

    def _calculate_annualized_return(self, returns: pd.Series) -> float:
        """Calculate annualized return."""
        if returns.empty:
            return 0.0

        total_return = self._calculate_total_return(returns)
        n_periods = len(returns)
        years = n_periods / self.annualization_factor
        return (1 + total_return) ** (1 / max(years, 0.001)) - 1

    def _calculate_volatility(self, returns: pd.Series) -> float:
        """Calculate annualized volatility."""
        if returns.empty or len(returns) < 2:
            return 0.0
        return returns.std() * np.sqrt(self.annualization_factor)

    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        """Calculate the Sharpe ratio."""
        if returns.empty or len(returns) < 2:
            return 0.0

        # Convert annual risk-free rate to per-period
        rf_per_period = (1 + risk_free_rate) ** (1 / self.annualization_factor) - 1
        excess_returns = returns - rf_per_period
        vol = returns.std()

        if vol == 0:
            return 0

        return (excess_returns.mean() / vol) * np.sqrt(self.annualization_factor)

    def _calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        """Calculate the Sortino ratio using downside deviation."""
        if returns.empty or len(returns) < 2:
            return 0.0

        # Convert annual risk-free rate to per-period
        rf_per_period = (1 + risk_free_rate) ** (1 / self.annualization_factor) - 1

        excess_returns = returns - rf_per_period
        downside_returns = returns[returns < 0]
        downside_deviation = np.std(downside_returns) * np.sqrt(self.annualization_factor) if len(
            downside_returns) > 0 else 0

        if downside_deviation == 0:
            return 0

        return excess_returns.mean() * self.annualization_factor / downside_deviation

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate the maximum drawdown."""
        if returns.empty:
            return 0.0

        cumulative = (1 + returns).cumprod()
        peak = cumulative.expanding(min_periods=1).max()
        drawdown = (cumulative / peak) - 1
        return abs(drawdown.min())

    def _calculate_calmar_ratio(self, returns: pd.Series) -> float:
        """Calculate the Calmar ratio (annualized return / max drawdown)."""
        ann_return = self._calculate_annualized_return(returns)
        max_dd = self._calculate_max_drawdown(returns)

        if max_dd == 0:
            return 0.0  # Avoid division by zero

        return ann_return / max_dd

    def _calculate_beta(self, returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate beta (market sensitivity)."""
        if returns.empty or benchmark_returns.empty or len(returns) < 2:
            return 0.0

        # Calculate covariance and variance
        cov = returns.cov(benchmark_returns)
        var = benchmark_returns.var()

        if var == 0:
            return 0.0  # Avoid division by zero

        return cov / var

    def _calculate_alpha(self, returns: pd.Series, benchmark_returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        """Calculate Jensen's alpha."""
        if returns.empty or benchmark_returns.empty or len(returns) < 2:
            return 0.0

        # Calculate per-period risk-free rate
        rf_per_period = (1 + risk_free_rate) ** (1 / self.annualization_factor) - 1

        # Calculate beta
        beta = self._calculate_beta(returns, benchmark_returns)

        # Calculate alpha (annualized)
        alpha_per_period = returns.mean() - (rf_per_period + beta * (benchmark_returns.mean() - rf_per_period))
        alpha_annualized = (1 + alpha_per_period) ** self.annualization_factor - 1

        return alpha_annualized

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

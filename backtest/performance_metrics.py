import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
import os

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
            # Align benchmark returns
            aligned_returns, aligned_benchmark = returns.align(benchmark_returns, join='inner')
            if not aligned_returns.empty and len(aligned_returns) > 1:
                metrics['alpha'] = self._calculate_alpha(aligned_returns, aligned_benchmark, risk_free_rate)
                metrics['beta'] = self._calculate_beta(aligned_returns, aligned_benchmark)
                metrics['information_ratio'] = self._calculate_information_ratio(aligned_returns, aligned_benchmark)
                metrics['tracking_error'] = self._calculate_tracking_error(aligned_returns, aligned_benchmark)
                metrics['correlation'] = aligned_returns.corr(aligned_benchmark)
                metrics['r_squared'] = self._calculate_r_squared(aligned_returns, aligned_benchmark)
        
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
            break_even_trades = trades_df[trades_df['value'] == 0]
            
            metrics['winning_trades'] = len(winning_trades)
            metrics['losing_trades'] = len(losing_trades)
            metrics['break_even_trades'] = len(break_even_trades)
            metrics['win_rate'] = len(winning_trades) / len(trades_df) if len(trades_df) > 0 else 0
            
            # Calculate profit metrics
            if not winning_trades.empty:
                metrics['avg_win'] = winning_trades['value'].mean()
                metrics['max_win'] = winning_trades['value'].max()
            
            if not losing_trades.empty:
                metrics['avg_loss'] = losing_trades['value'].mean()
                metrics['max_loss'] = losing_trades['value'].min()  # Note: this is negative
            
            # Calculate profit factor and expectancy
            total_win = winning_trades['value'].sum() if len(winning_trades) > 0 else 0
            total_loss = abs(losing_trades['value'].sum()) if len(losing_trades) > 0 else 0
            metrics['profit_factor'] = total_win / total_loss if total_loss > 0 else float('inf')
            
            # Calculate profit per trade
            metrics['avg_profit_per_trade'] = trades_df['value'].mean() if len(trades_df) > 0 else 0
            metrics['net_profit'] = trades_df['value'].sum()
        
        # Calculate commission costs if available
        if 'commission' in trades_df.columns:
            metrics['total_commission'] = trades_df['commission'].sum()
            metrics['avg_commission_per_trade'] = trades_df['commission'].mean()
        
        return metrics
    
    def _calculate_total_return(self, returns: pd.Series) -> float:
        """
        Calculate the total return.
        
        Args:
            returns: Series of returns
            
        Returns:
            Total return as a fraction
        """
        return (1 + returns).prod() - 1
    
    def _calculate_annualized_return(self, returns: pd.Series) -> float:
        """
        Calculate annualized return.
        
        Args:
            returns: Series of returns
            
        Returns:
            Annualized return as a fraction
        """
        if returns.empty:
            return 0.0
            
        total_return = self._calculate_total_return(returns)
        n_periods = len(returns)
        years = n_periods / self.annualization_factor
        return (1 + total_return) ** (1 / max(years, 0.001)) - 1
    
    def _calculate_volatility(self, returns: pd.Series) -> float:
        """
        Calculate annualized volatility.
        
        Args:
            returns: Series of returns
            
        Returns:
            Annualized volatility
        """
        if returns.empty or len(returns) < 2:
            return 0.0
        return returns.std() * np.sqrt(self.annualization_factor)
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        """
        Calculate the Sharpe ratio.
        
        Args:
            returns: Series of returns
            risk_free_rate: Annual risk-free rate
            
        Returns:
            Sharpe ratio
        """
        if returns.empty or len(returns) < 2:
            return 0.0
            
        # Convert annual risk-free rate to per-period
        rf_per_period = (1 + risk_free_rate) ** (1 / self.annualization_factor) - 1
        excess_returns = returns - rf_per_period
        vol = returns.std()
        if vol == 0:
            return 0
        return (excess_returns.mean() / vol) * np.sqrt(self.annualization_factor)
    
    def _calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float = 0.0, target_return: float = 0.0) -> float:
        """
        Calculate the Sortino ratio.
        
        Args:
            returns: Series of returns
            risk_free_rate: Annual risk-free rate
            target_return: Target return threshold
            
        Returns:
            Sortino ratio
        """
        if returns.empty or len(returns) < 2:
            return 0.0
            
        # Convert annual risk-free rate to per-period
        rf_per_period = (1 + risk_free_rate) ** (1 / self.annualization_factor) - 1
        
        excess_returns = returns - rf_per_period
        downside_returns = returns[returns < target_return] - target_return
        downside_deviation = np.sqrt(np.mean(downside_returns ** 2)) if len(downside_returns) > 0 else 0
        
        if downside_deviation == 0:
            return 0
        
        return (excess_returns.mean() / downside_deviation) * np.sqrt(self.annualization_factor)
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """
        Calculate the maximum drawdown.
        
        Args:
            returns: Series of returns
            
        Returns:
            Maximum drawdown as a positive fraction
        """
        if returns.empty:
            return 0.0
            
        cumulative = (1 + returns).cumprod()
        peak = cumulative.expanding(min_periods=1).max()
        drawdown = (cumulative / peak) - 1
        return abs(drawdown.min())
    
    def _calculate_calmar_ratio(self, returns: pd.Series) -> float:
        """
        Calculate the Calmar ratio (annualized return / max drawdown).
        
        Args:
            returns: Series of returns
            
        Returns:
            Calmar ratio
        """
        ann_return = self._calculate_annualized_return(returns)
        max_dd = self._calculate_max_drawdown(returns)
        if max_dd == 0:
            return 0.0  # Avoid division by zero
        return ann_return / max_dd
    
    def _calculate_alpha(self, returns: pd.Series, benchmark_returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        """
        Calculate Jensen's alpha.
        
        Args:
            returns: Series of returns
            benchmark_returns: Benchmark returns
            risk_free_rate: Annual risk-free rate
            
        Returns:
            Alpha (annualized)
        """
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
    
    def _calculate_beta(self, returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """
        Calculate beta (market sensitivity).
        
        Args:
            returns: Series of returns
            benchmark_returns: Benchmark returns
            
        Returns:
            Beta
        """
        if returns.empty or benchmark_returns.empty or len(returns) < 2:
            return 0.0
            
        # Calculate covariance and variance
        cov = returns.cov(benchmark_returns)
        var = benchmark_returns.var()
        
        if var == 0:
            return 0.0  # Avoid division by zero
        
        return cov / var
    
    def _calculate_information_ratio(self, returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """
        Calculate information ratio.
        
        Args:
            returns: Series of returns
            benchmark_returns: Benchmark returns
            
        Returns:
            Information ratio
        """
        if returns.empty or benchmark_returns.empty or len(returns) < 2:
            return 0.0
            
        # Calculate active returns and tracking error
        active_returns = returns - benchmark_returns
        tracking_error = active_returns.std()
        
        if tracking_error == 0:
            return 0.0  # Avoid division by zero
        
        return (active_returns.mean() / tracking_error) * np.sqrt(self.annualization_factor)
    
    def _calculate_tracking_error(self, returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """
        Calculate tracking error.
        
        Args:
            returns: Series of returns
            benchmark_returns: Benchmark returns
            
        Returns:
            Tracking error (annualized)
        """
        if returns.empty or benchmark_returns.empty or len(returns) < 2:
            return 0.0
            
        # Calculate active returns and tracking error
        active_returns = returns - benchmark_returns
        tracking_error = active_returns.std() * np.sqrt(self.annualization_factor)
        
        return tracking_error
    
    def _calculate_r_squared(self, returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """
        Calculate r-squared (coefficient of determination).
        
        Args:
            returns: Series of returns
            benchmark_returns: Benchmark returns
            
        Returns:
            R-squared value
        """
        if returns.empty or benchmark_returns.empty or len(returns) < 3:
            return 0.0
            
        correlation = returns.corr(benchmark_returns)
        return correlation ** 2
    
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
            
        fig, axes = plt.subplots(3, 1, figsize=(14, 18), gridspec_kw={'height_ratios': [2, 1, 1]})
        
        # Calculate cumulative returns
        cumulative_returns = (1 + returns).cumprod()
        
        # Plot 1: Cumulative returns
        cumulative_returns.plot(ax=axes[0], color='blue', linewidth=2, label='Strategy')
        
        if benchmark_returns is not None and not benchmark_returns.empty:
            # Align benchmark returns with strategy returns
            benchmark_returns = benchmark_returns.reindex(returns.index, method='ffill')
            cumulative_benchmark = (1 + benchmark_returns.fillna(0)).cumprod()
            cumulative_benchmark.plot(ax=axes[0], color='gray', linewidth=1.5, label='Benchmark', alpha=0.7)
        
        axes[0].set_title('Cumulative Returns', fontsize=14)
        axes[0].set_ylabel('Cumulative Return')
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
        
        # Plot 2: Monthly returns heatmap
        try:
            # Only create heatmap if we have enough data
            if len(returns.index) >= 30:  # Arbitrary threshold for enough data
                monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
                monthly_returns = monthly_returns.to_frame('returns')
                monthly_returns['year'] = monthly_returns.index.year
                monthly_returns['month'] = monthly_returns.index.month
                
                # Pivot data for heatmap
                pivot_data = monthly_returns.pivot('year', 'month', 'returns')
                
                # Plot heatmap
                sns.heatmap(pivot_data, annot=True, fmt='.2%', cmap='RdYlGn', center=0, ax=axes[1],
                          cbar_kws={'label': 'Monthly Return'})
                axes[1].set_title('Monthly Returns', fontsize=14)
                axes[1].set_ylabel('Year')
                
                # Set month names
                month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                axes[1].set_xticklabels(month_names)
            else:
                axes[1].text(0.5, 0.5, 'Insufficient data for monthly returns heatmap',
                          ha='center', va='center', fontsize=12)
                axes[1].set_title('Monthly Returns', fontsize=14)
        except Exception as e:
            logger.warning(f"Could not create monthly returns heatmap: {e}")
            axes[1].text(0.5, 0.5, f'Error creating monthly returns heatmap: {str(e)}',
                      ha='center', va='center', fontsize=12)
            axes[1].set_title('Monthly Returns', fontsize=14)
        
        # Plot 3: Drawdown chart
        cumulative = (1 + returns).cumprod()
        peak = cumulative.expanding(min_periods=1).max()
        drawdown = (cumulative / peak) - 1
        
        drawdown.plot(ax=axes[2], color='red', alpha=0.5, linewidth=1, label='Drawdown')
        axes[2].fill_between(drawdown.index, drawdown.values, 0, color='red', alpha=0.3)
        axes[2].set_title('Drawdown', fontsize=14)
        axes[2].set_ylabel('Drawdown')
        axes[2].set_ylim(min(drawdown.min() * 1.1, -0.01), 0.01)  # Ensure y-axis includes 0
        axes[2].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close(fig)
        else:
            plt.show()
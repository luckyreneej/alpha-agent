import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from typing import Dict, List, Optional, Union, Any, Tuple
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FactorAnalyzer:
    """
    Analyzes alpha factors to evaluate their performance and predictive power.
    """
    
    def __init__(self, results_dir: str = 'evaluation/factor_results'):
        """
        Initialize the factor analyzer.
        
        Args:
            results_dir: Directory to save analysis results
        """
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        self.performance_data = {}
        self.factor_correlations = pd.DataFrame()
    
    def calculate_factor_metrics(self, 
                               data: pd.DataFrame, 
                               factors: List[str], 
                               forward_returns_periods: List[int] = [1, 5, 10, 20]) -> Dict[str, Dict]:
        """
        Calculate performance metrics for multiple factors.
        
        Args:
            data: DataFrame with price data and factor values
            factors: List of factor column names to evaluate
            forward_returns_periods: Periods for forward return calculation
            
        Returns:
            Dictionary of factor metrics
        """
        # Calculate forward returns if they don't exist
        for period in forward_returns_periods:
            col_name = f'forward_return_{period}d'
            if col_name not in data.columns:
                data[col_name] = data['close'].pct_change(period).shift(-period)
        
        metrics = {}
        
        for factor in factors:
            if factor not in data.columns:
                logger.warning(f"Factor {factor} not in data columns. Skipping.")
                continue
                
            factor_metrics = self._calculate_single_factor_metrics(data, factor, forward_returns_periods)
            metrics[factor] = factor_metrics
        
        # Store results for later use
        self.performance_data = metrics
        
        # Calculate factor correlations
        self._calculate_factor_correlations(data, factors)
        
        return metrics
    
    def _calculate_single_factor_metrics(self, 
                                       data: pd.DataFrame, 
                                       factor: str, 
                                       forward_returns_periods: List[int]) -> Dict[str, Any]:
        """
        Calculate metrics for a single factor.
        
        Args:
            data: DataFrame with price and factor data
            factor: Factor column name
            forward_returns_periods: Periods for forward return calculation
            
        Returns:
            Dictionary of factor metrics
        """
        metrics = {
            'name': factor,
            'coverage': (data[factor].notna().sum() / len(data)) * 100,  # % of non-NaN values
            'information_coefficients': {},
            'quantile_returns': {}
        }
        
        # Calculate information coefficients (correlation with forward returns)
        for period in forward_returns_periods:
            col = f'forward_return_{period}d'
            # Spearman rank correlation
            ic = data[[factor, col]].dropna().corr(method='spearman').iloc[0, 1]
            metrics['information_coefficients'][f'{period}d'] = ic
        
        # Create quintiles for factor
        data_copy = data.copy()
        data_copy[f'{factor}_quintile'] = pd.qcut(
            data_copy[factor].rank(method='first'), 5, labels=False, duplicates='drop')
        
        # Calculate returns by quintile
        for period in forward_returns_periods:
            col = f'forward_return_{period}d'
            quintile_returns = data_copy.groupby(f'{factor}_quintile')[col].mean()
            metrics['quantile_returns'][f'{period}d'] = quintile_returns.to_dict()
        
        # Calculate metric stability
        half_len = len(data) // 2
        first_half = data.iloc[:half_len]
        second_half = data.iloc[half_len:]
        
        # Calculate ICs for both halves
        ic_first_half = first_half[[factor, f'forward_return_1d']].dropna().corr(method='spearman').iloc[0, 1]
        ic_second_half = second_half[[factor, f'forward_return_1d']].dropna().corr(method='spearman').iloc[0, 1]
        
        metrics['stability'] = {
            'ic_first_half': ic_first_half,
            'ic_second_half': ic_second_half,
            'ic_consistency': ic_first_half * ic_second_half > 0  # True if sign is consistent
        }
        
        # Calculate turnover
        metrics['turnover'] = self._calculate_factor_turnover(data, factor)
        
        # Calculate factor decay
        metrics['decay'] = self._calculate_factor_decay(
            data, factor, forward_returns_periods)
        
        return metrics
    
    def _calculate_factor_turnover(self, data: pd.DataFrame, factor: str) -> float:
        """
        Calculate factor turnover (how often the factor values change rank).
        
        Args:
            data: DataFrame with factor values
            factor: Factor column name
            
        Returns:
            Turnover metric (0-1 scale)
        """
        # Calculate factor ranks each day
        factor_ranks = data[factor].rank(method='first')
        
        # Calculate rank changes day over day
        rank_changes = factor_ranks.diff().abs()
        
        # Turnover is the average magnitude of rank changes relative to total possible changes
        avg_change = rank_changes.mean()
        max_possible_change = len(data[factor].dropna().unique())
        
        # Normalize to 0-1 scale
        turnover = min(1.0, avg_change / max_possible_change) if max_possible_change > 0 else 0
        
        return turnover
    
    def _calculate_factor_decay(self, 
                               data: pd.DataFrame, 
                               factor: str, 
                               forward_returns_periods: List[int]) -> Dict[str, float]:
        """
        Calculate how quickly factor predictive power decays over time.
        
        Args:
            data: DataFrame with factor values
            factor: Factor column name
            forward_returns_periods: Periods for forward return calculation
            
        Returns:
            Dictionary of decay metrics
        """
        decay_metrics = {}
        
        # Calculate correlations for each period
        correlations = []
        for period in forward_returns_periods:
            col = f'forward_return_{period}d'
            if col in data.columns:
                corr = data[[factor, col]].dropna().corr(method='spearman').iloc[0, 1]
                correlations.append((period, corr))
        
        # Sort by period
        correlations.sort(key=lambda x: x[0])
        
        # Calculate decay rate if we have at least two correlation points
        if len(correlations) >= 2:
            periods = [x[0] for x in correlations]
            corrs = [x[1] for x in correlations]
            
            # Calculate simple decay rate (change in correlation per day)
            if correlations[0][1] != 0:
                decay_rate = (correlations[-1][1] - correlations[0][1]) / (periods[-1] - periods[0])
                decay_metrics['daily_decay_rate'] = decay_rate
                
                # Half-life approximation (days until correlation reduces by half)
                if decay_rate < 0:  # Only if correlation is decaying
                    half_life = -0.5 * correlations[0][1] / decay_rate
                    decay_metrics['half_life_days'] = half_life
        
        return decay_metrics
    
    def _calculate_factor_correlations(self, data: pd.DataFrame, factors: List[str]) -> None:
        """
        Calculate correlations between factors.
        
        Args:
            data: DataFrame with factor values
            factors: List of factor column names
        """
        # Filter out factors not in data
        valid_factors = [f for f in factors if f in data.columns]
        
        if len(valid_factors) < 2:
            logger.warning("Need at least 2 factors to calculate correlations.")
            return
        
        # Calculate correlation matrix
        self.factor_correlations = data[valid_factors].corr(method='spearman')
    
    def generate_factor_report(self, output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive report of factor performance metrics.
        
        Args:
            output_file: Path to save the report JSON (optional)
            
        Returns:
            Dictionary with factor performance report
        """
        if not self.performance_data:
            logger.error("No performance data available. Run calculate_factor_metrics first.")
            return {}
        
        report = {
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'factors_analyzed': len(self.performance_data),
            'factor_metrics': self.performance_data,
            'best_factors': self._identify_best_factors(),
            'factor_correlations': self.factor_correlations.to_dict() if not self.factor_correlations.empty else {}
        }
        
        # Save report to file if requested
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Factor report saved to {output_file}")
        
        return report
    
    def _identify_best_factors(self) -> Dict[str, List[str]]:
        """
        Identify the best performing factors based on various metrics.
        
        Returns:
            Dictionary with lists of best factors by category
        """
        if not self.performance_data:
            return {}
        
        best_factors = {
            'highest_ic_1d': [],
            'highest_ic_5d': [],
            'highest_ic_20d': [],
            'most_stable': [],
            'lowest_turnover': [],
            'longest_half_life': []
        }
        
        # Find factors with highest information coefficients
        ic_1d = [(factor, metrics['information_coefficients'].get('1d', 0)) 
                 for factor, metrics in self.performance_data.items()]
        ic_5d = [(factor, metrics['information_coefficients'].get('5d', 0)) 
                for factor, metrics in self.performance_data.items()]
        ic_20d = [(factor, metrics['information_coefficients'].get('20d', 0)) 
                 for factor, metrics in self.performance_data.items()]
        
        # Sort by absolute IC value (ignoring sign)
        ic_1d.sort(key=lambda x: abs(x[1]), reverse=True)
        ic_5d.sort(key=lambda x: abs(x[1]), reverse=True)
        ic_20d.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Get top 3 factors for each IC period
        best_factors['highest_ic_1d'] = [x[0] for x in ic_1d[:3]]
        best_factors['highest_ic_5d'] = [x[0] for x in ic_5d[:3]]
        best_factors['highest_ic_20d'] = [x[0] for x in ic_20d[:3]]
        
        # Find most stable factors
        stability = [(factor, metrics['stability']['ic_first_half'] * metrics['stability']['ic_second_half']) 
                    for factor, metrics in self.performance_data.items() if 'stability' in metrics]
        stability.sort(key=lambda x: x[1], reverse=True)
        best_factors['most_stable'] = [x[0] for x in stability[:3]]
        
        # Find factors with lowest turnover
        turnover = [(factor, metrics.get('turnover', 1.0)) for factor, metrics in self.performance_data.items()]
        turnover.sort(key=lambda x: x[1])  # Lower is better
        best_factors['lowest_turnover'] = [x[0] for x in turnover[:3]]
        
        # Find factors with longest half-life
        half_life = [(factor, metrics.get('decay', {}).get('half_life_days', 0)) 
                    for factor, metrics in self.performance_data.items()]
        half_life = [(f, hl) for f, hl in half_life if hl > 0]  # Only positive half-lives
        half_life.sort(key=lambda x: x[1], reverse=True)  # Higher is better
        best_factors['longest_half_life'] = [x[0] for x in half_life[:3]]
        
        return best_factors
    
    def plot_factor_returns(self, factor_name: str, output_dir: Optional[str] = None) -> None:
        """
        Plot quintile returns for a specific factor.
        
        Args:
            factor_name: Name of the factor to plot
            output_dir: Directory to save the plot (optional)
        """
        if not self.performance_data or factor_name not in self.performance_data:
            logger.error(f"No data for factor {factor_name}")
            return
        
        factor_data = self.performance_data[factor_name]
        quintile_returns = factor_data['quantile_returns']
        
        # Create plot
        plt.figure(figsize=(10, 6))
        
        # Plot quintile returns for different periods
        for period, returns in quintile_returns.items():
            quintiles = sorted(returns.keys())
            values = [returns[q] for q in quintiles]
            plt.plot(quintiles, values, marker='o', label=f'{period} Returns')
        
        plt.title(f'Quintile Returns for {factor_name}')
        plt.xlabel('Factor Quintile')
        plt.ylabel('Average Forward Return')
        plt.legend()
        plt.grid(True)
        
        # Save plot if directory is provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, f'{factor_name}_quintile_returns.png'))
            plt.close()
        else:
            plt.show()
    
    def plot_factor_correlation_heatmap(self, output_path: Optional[str] = None) -> None:
        """
        Plot a heatmap of factor correlations.
        
        Args:
            output_path: Path to save the plot (optional)
        """
        if self.factor_correlations.empty:
            logger.error("No factor correlations calculated")
            return
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(self.factor_correlations, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('Factor Correlations')
        plt.tight_layout()
        
        # Save plot if path is provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path)
            plt.close()
        else:
            plt.show()
    
    def plot_ic_decay(self, output_path: Optional[str] = None) -> None:
        """
        Plot the decay of information coefficients over time for all factors.
        
        Args:
            output_path: Path to save the plot (optional)
        """
        if not self.performance_data:
            logger.error("No performance data available")
            return
        
        # Collect IC values for different periods
        periods = [1, 5, 10, 20]  # Assuming these are the periods we're calculating
        
        plt.figure(figsize=(12, 8))
        
        for factor, metrics in self.performance_data.items():
            ic_values = []
            for period in periods:
                ic = metrics['information_coefficients'].get(f'{period}d', None)
                if ic is not None:
                    ic_values.append((period, ic))
            
            if ic_values:
                periods_plot, ics_plot = zip(*sorted(ic_values, key=lambda x: x[0]))
                plt.plot(periods_plot, ics_plot, marker='o', label=factor)
        
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
        plt.title('Information Coefficient Decay')
        plt.xlabel('Forward Return Period (days)')
        plt.ylabel('Information Coefficient')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        
        # Save plot if path is provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path)
            plt.close()
        else:
            plt.show()
    
    def get_optimal_factor_combination(self, 
                                     min_ic: float = 0.05, 
                                     max_correlation: float = 0.7,
                                     min_half_life: float = 5) -> List[str]:
        """
        Get an optimal combination of factors with high IC, low correlation, and good half-life.
        
        Args:
            min_ic: Minimum absolute IC value to consider
            max_correlation: Maximum allowed correlation between factors
            min_half_life: Minimum acceptable half-life (days)
            
        Returns:
            List of factor names in the optimal combination
        """
        if not self.performance_data or self.factor_correlations.empty:
            logger.error("Need factor metrics and correlations first")
            return []
        
        # Filter factors by IC threshold (using 1-day IC)
        qualifying_factors = []
        for factor, metrics in self.performance_data.items():
            if abs(metrics['information_coefficients'].get('1d', 0)) >= min_ic:
                # Check half life if available
                half_life = metrics.get('decay', {}).get('half_life_days', 0)
                if half_life >= min_half_life or half_life == 0:  # Include if no decay rate calculated
                    qualifying_factors.append(factor)
        
        if not qualifying_factors:
            logger.warning("No factors meet the minimum IC and half-life criteria")
            return []
        
        # Sort by absolute IC value (highest first)
        sorted_factors = sorted(
            qualifying_factors,
            key=lambda f: abs(self.performance_data[f]['information_coefficients'].get('1d', 0)),
            reverse=True
        )
        
        # Greedy selection to maintain low correlation
        selected_factors = [sorted_factors[0]]  # Start with the highest IC factor
        
        for factor in sorted_factors[1:]:
            # Check correlation with all already selected factors
            if all(abs(self.factor_correlations.loc[factor, sf]) < max_correlation 
                  for sf in selected_factors):
                selected_factors.append(factor)
        
        return selected_factors
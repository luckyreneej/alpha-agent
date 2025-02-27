import numpy as np
import pandas as pd
from scipy import stats
import talib
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
import json
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AlphaFactors:
    """
    Implementation of alpha selection factors based on the 101 Alphas paper.
    This class integrates alpha factors from the tradingbot implementation
    and adds factor evaluation capabilities.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.alpha_registry = self._register_alphas()
        self.factor_performance = {}
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading config: {e}")
        
        # Default configuration
        return {
            'default_window': 20,
            'min_data_points': 252,  # Approximately one year of trading days
            'use_alphas': ['alpha1', 'alpha12', 'alpha101'],  # Default alphas to use
            'evaluation': {
                'min_ic_threshold': 0.05,
                'max_correlation': 0.7,
                'min_half_life_days': 5
            }
        }
    
    def _register_alphas(self) -> Dict[str, callable]:
        """Register all alpha factor calculation functions."""
        return {
            'alpha1': self.alpha1,
            'alpha2': self.alpha2,
            'alpha3': self.alpha3,
            'alpha4': self.alpha4,
            'alpha5': self.alpha5,
            'alpha6': self.alpha6,
            'alpha7': self.alpha7,
            'alpha8': self.alpha8,
            'alpha9': self.alpha9,
            'alpha10': self.alpha10,
            'alpha11': self.alpha11,
            'alpha12': self.alpha12,
            'alpha13': self.alpha13,
            'alpha14': self.alpha14,
            'alpha15': self.alpha15,
            'alpha16': self.alpha16,
            'alpha101': self.alpha101,
            # Custom alphas
            'election_year_momentum': self.election_year_momentum,
            'sector_rotation': self.sector_rotation,
            'volatility_regime': self.volatility_regime
        }
    
    def calculate_alpha_factors(self, 
                               data: pd.DataFrame, 
                               selected_factors: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Calculate all or selected alpha factors for the given data.
        
        Args:
            data: DataFrame with at least OHLCV data
            selected_factors: List of factor names to calculate, or None for all
            
        Returns:
            DataFrame with alpha factors added as columns
        """
        # Make a copy to avoid modifying the original
        df_result = data.copy()
        
        # Validate required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df_result.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            raise ValueError(f"DataFrame must contain columns: {required_cols}")
        
        # Calculate returns if not present
        if 'returns' not in df_result.columns:
            df_result['returns'] = df_result['close'].pct_change()
        
        # Calculate volume moving average if not present
        for period in [5, 10, 20, 60, 120, 180]:
            col_name = f'adv{period}'
            if col_name not in df_result.columns:
                df_result[col_name] = df_result['volume'].rolling(window=period).mean()
        
        # Calculate VWAP if not present
        if 'vwap' not in df_result.columns:
            df_result['vwap'] = (df_result['volume'] * (df_result['high'] + df_result['low'] + df_result['close']) / 3).cumsum() / df_result['volume'].cumsum()
        
        # If market cap isn't provided, use close price as a proxy
        if 'cap' not in df_result.columns:
            df_result['cap'] = df_result['close']
        
        # Calculate selected factors or all factors
        factors_to_calculate = selected_factors or self.config['use_alphas']
        
        for factor_name in factors_to_calculate:
            if factor_name in self.alpha_registry:
                try:
                    # Call the factor calculation function
                    df_result[factor_name] = self.alpha_registry[factor_name](df_result)
                    logger.debug(f"Calculated {factor_name}")
                except Exception as e:
                    logger.error(f"Error calculating {factor_name}: {e}")
            else:
                logger.warning(f"Unknown factor: {factor_name}")
        
        return df_result
    
    def evaluate_factor_performance(self, 
                                   data: pd.DataFrame, 
                                   factor_names: List[str], 
                                   forward_returns_periods: List[int] = [1, 5, 10, 20]) -> Dict[str, Dict]:
        """
        Evaluate the performance of alpha factors based on correlation with future returns.
        
        Args:
            data: DataFrame with price data and calculated factors
            factor_names: List of factor names to evaluate
            forward_returns_periods: List of forward return periods (in days)
            
        Returns:
            Dictionary with performance metrics for each factor
        """
        # Calculate forward returns
        for period in forward_returns_periods:
            data[f'forward_return_{period}d'] = data['close'].pct_change(period).shift(-period)
        
        results = {}
        
        for factor in factor_names:
            if factor in data.columns:
                factor_results = {}
                
                # Calculate factor quintiles
                data[f'{factor}_quintile'] = pd.qcut(data[factor].rank(method='first'), 5, labels=False, duplicates='drop')
                
                # Calculate information coefficient (correlation with forward returns)
                for period in forward_returns_periods:
                    corr = data[[factor, f'forward_return_{period}d']].corr().iloc[0, 1]
                    factor_results[f'IC_{period}d'] = corr
                
                # Calculate mean return by quintile
                quintile_returns = {}
                for period in forward_returns_periods:
                    quintile_means = data.groupby(f'{factor}_quintile')[f'forward_return_{period}d'].mean()
                    quintile_returns[f'{period}d'] = quintile_means.to_dict()
                
                factor_results['quintile_returns'] = quintile_returns
                
                # Calculate factor turnover
                factor_results['turnover'] = self._calculate_factor_turnover(data, factor)
                
                # Calculate factor decay
                factor_results['decay'] = self._calculate_factor_decay(data, factor, forward_returns_periods)
                
                # Store results
                results[factor] = factor_results
            else:
                logger.warning(f"Factor {factor} not found in data")
        
        # Update internal factor performance metrics
        self.factor_performance.update(results)
        
        return results
    
    def _calculate_factor_turnover(self, data: pd.DataFrame, factor: str) -> float:
        """
        Calculate factor turnover (how often the factor values change rank).
        
        Args:
            data: DataFrame with calculated factor
            factor: Factor name
            
        Returns:
            Turnover metric (0-1 scale)
        """
        # Calculate factor ranks each day
        factor_ranks = data[factor].rank(method='first')
        
        # Calculate rank changes day over day
        rank_changes = factor_ranks.diff().abs()
        
        # Turnover is the average magnitude of rank changes relative to total possible changes
        avg_change = rank_changes.mean()
        max_possible_change = len(data[factor].unique())
        
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
            data: DataFrame with calculated factor
            factor: Factor name
            forward_returns_periods: List of forward periods
            
        Returns:
            Dictionary of decay metrics
        """
        decay_metrics = {}
        
        # Calculate correlations for each period
        correlations = []
        for period in forward_returns_periods:
            col = f'forward_return_{period}d'
            if col in data.columns:
                corr = data[[factor, col]].corr().iloc[0, 1]
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
    
    def select_optimal_factors(self, 
                             min_ic_threshold: float = 0.05, 
                             max_correlation: float = 0.7,
                             min_half_life_days: float = 5) -> List[str]:
        """
        Select optimal combination of alpha factors based on performance metrics.
        
        Args:
            min_ic_threshold: Minimum information coefficient threshold
            max_correlation: Maximum allowed correlation between factors
            min_half_life_days: Minimum factor half-life (days)
            
        Returns:
            List of optimal factor names
        """
        if not self.factor_performance:
            logger.error("No factor performance data available. Run evaluate_factor_performance first.")
            return []
        
        # Filter factors by IC threshold
        viable_factors = []
        for factor, metrics in self.factor_performance.items():
            # Check if factor has decent predictive power for any period
            has_predictive_power = any(abs(v) >= min_ic_threshold for k, v in metrics.items() if k.startswith('IC_'))
            
            # Check if factor has acceptable decay rate
            acceptable_decay = True
            if 'decay' in metrics and 'half_life_days' in metrics['decay']:
                acceptable_decay = metrics['decay']['half_life_days'] >= min_half_life_days
            
            if has_predictive_power and acceptable_decay:
                viable_factors.append(factor)
        
        # If no viable factors, return empty list
        if not viable_factors:
            return []
        
        # Greedily select factors with low correlation to already selected factors
        selected_factors = [viable_factors[0]]  # Start with the best factor
        
        for candidate in viable_factors[1:]:
            # Check correlation with already selected factors
            is_correlated = False
            for selected in selected_factors:
                if self._get_factor_correlation(candidate, selected) > max_correlation:
                    is_correlated = True
                    break
            
            if not is_correlated:
                selected_factors.append(candidate)
        
        return selected_factors
    
    def _get_factor_correlation(self, factor1: str, factor2: str) -> float:
        """
        Get correlation between two factors using stored performance data.
        
        Args:
            factor1: First factor name
            factor2: Second factor name
            
        Returns:
            Correlation coefficient or 0 if not available
        """
        # In a real implementation, you would have a correlation matrix
        # This is just a placeholder that should be replaced with actual correlation calculation
        return 0.5 if factor1 != factor2 else 1.0
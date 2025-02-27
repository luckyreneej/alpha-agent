import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Union, Tuple, Any
import scipy.optimize as sco

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PortfolioOptimizer:
    """
    Portfolio optimization tools for asset allocation and risk management.
    Provides various optimization methods like Mean-Variance, Max Sharpe, Min Variance, etc.
    """
    
    def __init__(self, risk_free_rate: float = 0.0):
        """
        Initialize the portfolio optimizer.
        
        Args:
            risk_free_rate: Annual risk-free rate
        """
        self.risk_free_rate = risk_free_rate
    
    def optimize(self, returns: pd.DataFrame, method: str = 'max_sharpe', 
                 constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Optimize portfolio weights using the specified method.
        
        Args:
            returns: DataFrame with asset returns (each column is an asset)
            method: Optimization method ('max_sharpe', 'min_variance', 'equal_weight', 'risk_parity')
            constraints: Additional constraints for optimization
            
        Returns:
            Dictionary with optimized weights and metrics
        """
        if returns.empty:
            logger.warning("Empty returns DataFrame provided")
            return {'weights': {}, 'metrics': {}}
            
        if method == 'max_sharpe':
            return self._optimize_max_sharpe(returns, constraints)
        elif method == 'min_variance':
            return self._optimize_min_variance(returns, constraints)
        elif method == 'equal_weight':
            return self._optimize_equal_weight(returns)
        elif method == 'risk_parity':
            return self._optimize_risk_parity(returns, constraints)
        else:
            logger.warning(f"Unknown optimization method: {method}, using equal weight")
            return self._optimize_equal_weight(returns)
    
    def _optimize_equal_weight(self, returns: pd.DataFrame) -> Dict[str, Any]:
        """
        Equal weight portfolio allocation.
        
        Args:
            returns: DataFrame with asset returns
            
        Returns:
            Dictionary with optimized weights and metrics
        """
        n_assets = len(returns.columns)
        weights = np.ones(n_assets) / n_assets
        
        # Calculate portfolio metrics
        portfolio_return, portfolio_volatility, sharpe_ratio = self._calculate_portfolio_metrics(
            weights, returns
        )
        
        return {
            'weights': dict(zip(returns.columns, weights)),
            'metrics': {
                'return': portfolio_return,
                'volatility': portfolio_volatility,
                'sharpe_ratio': sharpe_ratio
            }
        }
    
    def _optimize_max_sharpe(self, returns: pd.DataFrame, 
                           constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Maximize Sharpe ratio portfolio optimization.
        
        Args:
            returns: DataFrame with asset returns
            constraints: Additional constraints (e.g., {'min_weight': 0.0, 'max_weight': 1.0})
            
        Returns:
            Dictionary with optimized weights and metrics
        """
        n_assets = len(returns.columns)
        
        # Parse constraints
        min_weight = constraints.get('min_weight', 0.0) if constraints else 0.0
        max_weight = constraints.get('max_weight', 1.0) if constraints else 1.0
        
        # Initial weights guess
        weights_guess = np.ones(n_assets) / n_assets
        
        # Define optimization constraints
        bounds = tuple((min_weight, max_weight) for _ in range(n_assets))
        constraints_list = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1.0})
        
        # Define objective function to minimize (negative Sharpe ratio)
        def objective(weights):
            portfolio_return, portfolio_volatility, sharpe_ratio = self._calculate_portfolio_metrics(
                weights, returns
            )
            return -sharpe_ratio
        
        # Run optimization
        result = sco.minimize(
            objective,
            weights_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list
        )
        
        if not result.success:
            logger.warning(f"Optimization failed: {result.message}")
            # Fallback to equal weight
            return self._optimize_equal_weight(returns)
        
        optimal_weights = result.x
        
        # Calculate portfolio metrics with optimal weights
        portfolio_return, portfolio_volatility, sharpe_ratio = self._calculate_portfolio_metrics(
            optimal_weights, returns
        )
        
        return {
            'weights': dict(zip(returns.columns, optimal_weights)),
            'metrics': {
                'return': portfolio_return,
                'volatility': portfolio_volatility,
                'sharpe_ratio': sharpe_ratio
            }
        }
    
    def _optimize_min_variance(self, returns: pd.DataFrame, 
                            constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Minimize portfolio variance optimization.
        
        Args:
            returns: DataFrame with asset returns
            constraints: Additional constraints (e.g., {'min_weight': 0.0, 'max_weight': 1.0})
            
        Returns:
            Dictionary with optimized weights and metrics
        """
        n_assets = len(returns.columns)
        
        # Parse constraints
        min_weight = constraints.get('min_weight', 0.0) if constraints else 0.0
        max_weight = constraints.get('max_weight', 1.0) if constraints else 1.0
        min_return = constraints.get('min_return', None)
        
        # Initial weights guess
        weights_guess = np.ones(n_assets) / n_assets
        
        # Calculate covariance matrix
        cov_matrix = returns.cov()
        
        # Define optimization constraints
        bounds = tuple((min_weight, max_weight) for _ in range(n_assets))
        constraints_list = [{'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1.0}]
        
        # Add minimum return constraint if specified
        if min_return is not None:
            mean_returns = returns.mean()
            constraints_list.append({
                'type': 'ineq',
                'fun': lambda weights: np.dot(weights, mean_returns) - min_return
            })
        
        # Define objective function to minimize (portfolio variance)
        def objective(weights):
            return np.dot(weights.T, np.dot(cov_matrix, weights))
        
        # Run optimization
        result = sco.minimize(
            objective,
            weights_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list
        )
        
        if not result.success:
            logger.warning(f"Optimization failed: {result.message}")
            # Fallback to equal weight
            return self._optimize_equal_weight(returns)
        
        optimal_weights = result.x
        
        # Calculate portfolio metrics with optimal weights
        portfolio_return, portfolio_volatility, sharpe_ratio = self._calculate_portfolio_metrics(
            optimal_weights, returns
        )
        
        return {
            'weights': dict(zip(returns.columns, optimal_weights)),
            'metrics': {
                'return': portfolio_return,
                'volatility': portfolio_volatility,
                'sharpe_ratio': sharpe_ratio
            }
        }
    
    def _optimize_risk_parity(self, returns: pd.DataFrame, 
                           constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Risk parity portfolio optimization.
        
        Args:
            returns: DataFrame with asset returns
            constraints: Additional constraints
            
        Returns:
            Dictionary with optimized weights and metrics
        """
        n_assets = len(returns.columns)
        
        # Calculate covariance matrix
        cov_matrix = returns.cov().values
        
        # Define the risk budget (equal risk contribution for each asset)
        risk_budget = np.ones(n_assets) / n_assets
        
        # Initial weights guess
        weights_guess = np.ones(n_assets) / n_assets
        
        # Define optimization constraints
        constraints_list = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1.0})
        bounds = tuple((0.0, 1.0) for _ in range(n_assets))
        
        # Define objective function for risk parity
        def objective(weights):
            # Calculate portfolio variance
            portfolio_var = np.dot(weights.T, np.dot(cov_matrix, weights))
            portfolio_vol = np.sqrt(portfolio_var)
            
            # Calculate risk contribution of each asset
            marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol
            risk_contrib = weights * marginal_contrib
            total_risk_contrib = np.sum(risk_contrib)
            
            # Normalize risk contributions
            risk_contrib_normalized = risk_contrib / total_risk_contrib
            
            # Calculate sum of squared error between risk contributions and risk budget
            return np.sum((risk_contrib_normalized - risk_budget) ** 2)
        
        # Run optimization
        result = sco.minimize(
            objective,
            weights_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list,
            options={'maxiter': 1000}
        )
        
        if not result.success:
            logger.warning(f"Risk parity optimization failed: {result.message}")
            # Fallback to equal weight
            return self._optimize_equal_weight(returns)
        
        optimal_weights = result.x
        
        # Calculate portfolio metrics with optimal weights
        portfolio_return, portfolio_volatility, sharpe_ratio = self._calculate_portfolio_metrics(
            optimal_weights, returns
        )
        
        return {
            'weights': dict(zip(returns.columns, optimal_weights)),
            'metrics': {
                'return': portfolio_return,
                'volatility': portfolio_volatility,
                'sharpe_ratio': sharpe_ratio
            }
        }
    
    def _calculate_portfolio_metrics(self, weights: np.ndarray, returns: pd.DataFrame) -> Tuple[float, float, float]:
        """
        Calculate portfolio return, volatility, and Sharpe ratio.
        
        Args:
            weights: Array of asset weights
            returns: DataFrame with asset returns
            
        Returns:
            Tuple of (portfolio_return, portfolio_volatility, sharpe_ratio)
        """
        # Annual portfolio return
        mean_returns = returns.mean().values
        portfolio_return = np.sum(mean_returns * weights) * 252  # Annualize
        
        # Annual portfolio volatility
        cov_matrix = returns.cov().values
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)  # Annualize
        
        # Sharpe ratio
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
        
        return portfolio_return, portfolio_volatility, sharpe_ratio
    
    def efficient_frontier(self, returns: pd.DataFrame, points: int = 50) -> pd.DataFrame:
        """
        Calculate the efficient frontier.
        
        Args:
            returns: DataFrame with asset returns
            points: Number of points to calculate
            
        Returns:
            DataFrame with efficient frontier points (return, volatility, sharpe)
        """
        # Calculate mean returns and covariance matrix
        mean_returns = returns.mean() * 252  # Annualize
        cov_matrix = returns.cov() * 252  # Annualize
        
        # Find minimum volatility and maximum return portfolios
        min_vol_result = self._optimize_min_variance(returns)
        min_vol_return = min_vol_result['metrics']['return']
        min_vol_vol = min_vol_result['metrics']['volatility']
        
        # Find max return (single asset with highest return)
        max_return_asset = mean_returns.idxmax()
        max_return = mean_returns.max()
        
        # Define return targets between min vol return and max return
        target_returns = np.linspace(min_vol_return, max_return, points)
        efficient_frontier_data = []
        
        for target_return in target_returns:
            # Optimize for minimum volatility at this target return
            result = self._optimize_min_variance(
                returns, 
                constraints={'min_weight': 0.0, 'max_weight': 1.0, 'min_return': target_return / 252}  # De-annualize
            )
            
            # Extract metrics
            portfolio_return = result['metrics']['return']
            portfolio_volatility = result['metrics']['volatility']
            sharpe_ratio = result['metrics']['sharpe_ratio']
            
            efficient_frontier_data.append({
                'return': portfolio_return,
                'volatility': portfolio_volatility,
                'sharpe_ratio': sharpe_ratio
            })
        
        return pd.DataFrame(efficient_frontier_data)
    
    def rolling_optimization(self, returns: pd.DataFrame, window: int = 252, 
                           method: str = 'max_sharpe', step: int = 20) -> Dict[str, pd.DataFrame]:
        """
        Perform rolling window portfolio optimization.
        
        Args:
            returns: DataFrame with asset returns
            window: Window size for optimization
            method: Optimization method
            step: Step size for rolling window
            
        Returns:
            Dictionary with weights and metrics DataFrames
        """
        if len(returns) < window:
            logger.warning(f"Not enough data for rolling optimization. Need at least {window} periods.")
            return {'weights': pd.DataFrame(), 'metrics': pd.DataFrame()}
        
        # Convert to float index for shifting
        returns = returns.reset_index()
        dates = returns['date'] if 'date' in returns.columns else returns.index
        returns = returns.drop('date', axis=1) if 'date' in returns.columns else returns
        
        # Prepare containers for results
        all_weights = []
        all_metrics = []
        
        # Loop through returns with specified window and step
        for i in range(0, len(returns) - window + 1, step):
            # Define window end
            window_end = i + window
            if window_end > len(returns):
                window_end = len(returns)
                
            # Get window data
            window_returns = returns.iloc[i:window_end]
            window_date = dates.iloc[window_end - 1] if hasattr(dates, 'iloc') else dates[window_end - 1]
            
            # Optimize for this window
            result = self.optimize(window_returns, method=method)
            
            # Store weights and metrics with date
            weights_dict = result['weights']
            weights_dict['date'] = window_date
            all_weights.append(weights_dict)
            
            metrics_dict = result['metrics']
            metrics_dict['date'] = window_date
            all_metrics.append(metrics_dict)
        
        # Convert to DataFrames
        weights_df = pd.DataFrame(all_weights).set_index('date')
        metrics_df = pd.DataFrame(all_metrics).set_index('date')
        
        return {'weights': weights_df, 'metrics': metrics_df}
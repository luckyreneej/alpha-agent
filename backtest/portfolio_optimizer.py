import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, Tuple, Any
import scipy.optimize as sco

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PortfolioOptimizer:
    """
    Portfolio optimization tools for asset allocation and risk management.
    Provides various optimization methods like Max Sharpe, Min Variance, etc.
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

        # Select optimization method
        optimizers = {
            'max_sharpe': self._optimize_max_sharpe,
            'min_variance': self._optimize_min_variance,
            'equal_weight': self._optimize_equal_weight,
            'risk_parity': self._optimize_risk_parity
        }

        if method not in optimizers:
            logger.warning(f"Unknown optimization method: {method}, using equal weight")
            method = 'equal_weight'

        # Call appropriate optimizer
        if method == 'equal_weight':
            # Equal weight doesn't use constraints
            return optimizers[method](returns)
        else:
            return optimizers[method](returns, constraints)

    def _optimize_equal_weight(self, returns: pd.DataFrame) -> Dict[str, Any]:
        """Equal weight portfolio allocation."""
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
        """Maximize Sharpe ratio portfolio optimization."""
        n_assets = len(returns.columns)

        # Parse constraints
        min_weight = constraints.get('min_weight', 0.0) if constraints else 0.0
        max_weight = constraints.get('max_weight', 1.0) if constraints else 1.0

        # Initial weights guess (equal weight)
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
        try:
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
        except Exception as e:
            logger.error(f"Error in max_sharpe optimization: {e}")
            return self._optimize_equal_weight(returns)

    def _optimize_min_variance(self, returns: pd.DataFrame,
                               constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Minimize portfolio variance optimization."""
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
        try:
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
        except Exception as e:
            logger.error(f"Error in min_variance optimization: {e}")
            return self._optimize_equal_weight(returns)

    def _optimize_risk_parity(self, returns: pd.DataFrame,
                              constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Risk parity portfolio optimization."""
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
            risk_contrib_normalized = risk_contrib / np.sum(risk_contrib)

            # Calculate sum of squared error between risk contributions and risk budget
            return np.sum((risk_contrib_normalized - risk_budget) ** 2)

        # Run optimization
        try:
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
        except Exception as e:
            logger.error(f"Error in risk_parity optimization: {e}")
            return self._optimize_equal_weight(returns)

    def _calculate_portfolio_metrics(self, weights: np.ndarray, returns: pd.DataFrame) -> Tuple[float, float, float]:
        """Calculate portfolio return, volatility, and Sharpe ratio."""
        # Annual portfolio return
        mean_returns = returns.mean().values
        portfolio_return = np.sum(mean_returns * weights) * 252  # Annualize

        # Annual portfolio volatility
        cov_matrix = returns.cov().values
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)  # Annualize

        # Sharpe ratio
        sharpe_ratio = (
                                   portfolio_return - self.risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0

        return portfolio_return, portfolio_volatility, sharpe_ratio

    def efficient_frontier(self, returns: pd.DataFrame, points: int = 20) -> pd.DataFrame:
        """
        Calculate the efficient frontier.

        Args:
            returns: DataFrame with asset returns
            points: Number of points to calculate

        Returns:
            DataFrame with efficient frontier points (return, volatility, sharpe)
        """
        # Find minimum volatility portfolio
        min_vol_result = self._optimize_min_variance(returns)
        min_vol_return = min_vol_result['metrics']['return']

        # Find maximum return (single asset with highest return)
        mean_returns = returns.mean() * 252  # Annualize
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

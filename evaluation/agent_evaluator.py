import pandas as pd
import numpy as np
import json
import os
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

# Import utility modules
from metrics_utils import (
    calculate_error_metrics,
    calculate_directional_accuracy,
    sample_time_series
)
from data_cache import DataCache
from visualization import create_performance_dashboard

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AgentEvaluator:
    """
    Evaluates the performance of individual agents and the overall multi-agent system.
    Tracks metrics over time and provides visualization tools.

    Attributes:
        evaluation_dir (str): Directory for storing evaluation results
        data_cache (DataCache): Cache for storing and retrieving metrics data
        agent_metrics (dict): Performance metrics for each agent
        system_metrics (dict): System-level metrics
    """

    def __init__(self, evaluation_dir: str = 'evaluation/results'):
        """
        Initialize the agent evaluator.

        Args:
            evaluation_dir: Directory for storing evaluation results
        """
        self.evaluation_dir = evaluation_dir
        os.makedirs(evaluation_dir, exist_ok=True)

        # Initialize data cache
        self.data_cache = DataCache(evaluation_dir)

        # Performance metrics for each agent
        self.agent_metrics = {}

        # System-level metrics
        self.system_metrics = {
            'prediction_accuracy': [],
            'trading_performance': [],
            'communication_efficiency': [],
            'runtime_performance': [],
            'overall_score': []
        }

        # For tracking last auto-save time
        self._last_save_time = time.time()
        self._auto_save_interval = 300  # 5 minutes

    def evaluate_prediction_agent(self,
                                  predictions: Dict[str, Any],
                                  actual_values: Dict[str, Any],
                                  window: int = 20) -> Dict[str, Any]:
        """
        Evaluate the performance of the prediction agent.

        Args:
            predictions: Dictionary of predicted values
            actual_values: Dictionary of actual values
            window: Window size for rolling metrics

        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {}

        # Check that we have data for the same tickers
        common_tickers = set(predictions.keys()) & set(actual_values.keys())

        if not common_tickers:
            logger.warning("No common tickers between predictions and actual values")
            return {'error': 'No common tickers'}

        # Calculate metrics for each ticker
        ticker_metrics = {}
        for ticker in common_tickers:
            pred = np.array(predictions[ticker]) if isinstance(predictions[ticker], list) else np.array(
                [predictions[ticker]])
            actual = np.array(actual_values[ticker]) if isinstance(actual_values[ticker], list) else np.array(
                [actual_values[ticker]])

            # Calculate basic error metrics
            error_metrics = calculate_error_metrics(pred, actual)
            ticker_metrics[ticker] = error_metrics

            # Add metrics to result with ticker prefix
            for key, value in error_metrics.items():
                metrics[f'{ticker}_{key}'] = value

            # Calculate directional accuracy if enough data
            if len(pred) > 1 and len(actual) > 1:
                dir_acc = calculate_directional_accuracy(pred, actual)
                metrics[f'{ticker}_directional_accuracy'] = dir_acc
                ticker_metrics[ticker]['directional_accuracy'] = dir_acc

            # Calculate rolling metrics if enough data
            if len(pred) >= window and len(actual) >= window:
                # Use pandas rolling window functions for efficiency
                pred_series = pd.Series(pred)
                actual_series = pd.Series(actual)

                # Calculate rolling MSE
                error_squared = (pred_series - actual_series) ** 2
                rolling_mse = error_squared.rolling(window=window).mean().dropna().tolist()
                metrics[f'{ticker}_rolling_mse'] = rolling_mse

                # Calculate rolling directional accuracy if enough data
                if len(pred) > window + 1:
                    pred_dir = np.sign(pred_series.diff()).rolling(window=window)
                    actual_dir = np.sign(actual_series.diff()).rolling(window=window)

                    # Calculate accuracy for each window
                    rolling_dir_acc = []
                    for i in range(window, len(pred)):
                        window_pred_dir = np.sign(np.diff(pred[i - window:i]))
                        window_actual_dir = np.sign(np.diff(actual[i - window:i]))
                        dir_acc = np.mean(window_pred_dir == window_actual_dir)
                        rolling_dir_acc.append(dir_acc)

                    metrics[f'{ticker}_rolling_dir_acc'] = rolling_dir_acc

        # Calculate aggregate metrics across all tickers
        metrics['avg_mse'] = np.mean([m.get('mse', 0) for m in ticker_metrics.values()])
        metrics['avg_rmse'] = np.mean([m.get('rmse', 0) for m in ticker_metrics.values()])
        metrics['avg_mae'] = np.mean([m.get('mae', 0) for m in ticker_metrics.values()])

        dir_acc_values = [m.get('directional_accuracy', 0) for m in ticker_metrics.values() if
                          'directional_accuracy' in m]
        metrics['avg_directional_accuracy'] = np.mean(dir_acc_values) if dir_acc_values else 0

        # Calculate overall score (0-100) using simplified approach
        normalized_dir_acc = metrics['avg_directional_accuracy'] * 100  # 0-100 scale

        # Normalize error metrics (lower is better) with more intuitive linear scaling
        normalized_error = max(0, 100 - (metrics['avg_rmse'] * 20))  # Simple linear scaling

        # Combine into overall score
        metrics['overall_score'] = 0.7 * normalized_dir_acc + 0.3 * normalized_error

        # Add timestamp
        metrics['timestamp'] = datetime.now().isoformat()

        # Save to agent metrics
        if 'prediction_agent' not in self.agent_metrics:
            self.agent_metrics['prediction_agent'] = []
        self.agent_metrics['prediction_agent'].append(metrics)

        # Update system metrics
        self.system_metrics['prediction_accuracy'].append({
            'timestamp': metrics['timestamp'],
            'score': metrics['overall_score']
        })

        # Check for auto-save
        self._check_auto_save()

        return metrics

    def evaluate_trading_agent(self, trades: List[Dict], portfolio_history: List[Dict]) -> Dict[str, Any]:
        """
        Evaluate the performance of the trading agent.

        Args:
            trades: List of executed trades
            portfolio_history: Historical portfolio values

        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {}

        if not trades or not portfolio_history:
            logger.warning("No trades or portfolio history to evaluate")
            return {'error': 'Insufficient data'}

        # Convert to DataFrames for easier analysis
        trades_df = pd.DataFrame(trades)
        portfolio_df = pd.DataFrame(portfolio_history)

        # Basic trading metrics
        metrics['total_trades'] = len(trades_df)

        if 'pnl' in trades_df.columns and len(trades_df) > 0:
            # Profitable vs. losing trades
            metrics['profitable_trades'] = int(trades_df[trades_df['pnl'] > 0].shape[0])
            metrics['losing_trades'] = int(trades_df[trades_df['pnl'] < 0].shape[0])
            metrics['breakeven_trades'] = int(trades_df[trades_df['pnl'] == 0].shape[0])

            # Calculate win rate and profit metrics
            metrics['win_rate'] = metrics['profitable_trades'] / metrics['total_trades']

            if metrics['profitable_trades'] > 0:
                metrics['avg_profit'] = float(trades_df[trades_df['pnl'] > 0]['pnl'].mean())
            else:
                metrics['avg_profit'] = 0.0

            if metrics['losing_trades'] > 0:
                metrics['avg_loss'] = float(trades_df[trades_df['pnl'] < 0]['pnl'].mean())
            else:
                metrics['avg_loss'] = 0.0

            # Profit factor (ratio of gross profit to gross loss)
            total_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
            total_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())

            if total_loss != 0:
                metrics['profit_factor'] = float(total_profit / total_loss)
            else:
                metrics['profit_factor'] = float('inf') if total_profit > 0 else 0.0

            metrics['net_profit'] = float(trades_df['pnl'].sum())
        else:
            # Default values if PnL data is missing
            metrics['profitable_trades'] = 0
            metrics['losing_trades'] = 0
            metrics['win_rate'] = 0.0
            metrics['avg_profit'] = 0.0
            metrics['avg_loss'] = 0.0
            metrics['profit_factor'] = 0.0
            metrics['net_profit'] = 0.0

        # Portfolio metrics
        if 'portfolio_value' in portfolio_df.columns and len(portfolio_df) > 1:
            initial_value = portfolio_df['portfolio_value'].iloc[0]
            final_value = portfolio_df['portfolio_value'].iloc[-1]

            metrics['total_return'] = float((final_value - initial_value) / initial_value)
            metrics['total_return_pct'] = float(metrics['total_return'] * 100)

            # Daily returns and risk metrics
            if 'date' in portfolio_df.columns:
                # Ensure date column is datetime
                if not pd.api.types.is_datetime64_dtype(portfolio_df['date']):
                    portfolio_df['date'] = pd.to_datetime(portfolio_df['date'])

                portfolio_df = portfolio_df.sort_values('date')
                portfolio_df['daily_return'] = portfolio_df['portfolio_value'].pct_change()

                # Calculate annualized metrics if we have enough data
                days = (portfolio_df['date'].iloc[-1] - portfolio_df['date'].iloc[0]).days
                if days > 0:
                    years = days / 365.25
                    metrics['annualized_return'] = float((1 + metrics['total_return']) ** (1 / years) - 1)
                    metrics['annualized_return_pct'] = float(metrics['annualized_return'] * 100)

                    # Volatility (annualized standard deviation of returns)
                    daily_returns = portfolio_df['daily_return'].dropna()
                    if len(daily_returns) > 0:
                        metrics['volatility'] = float(daily_returns.std() * np.sqrt(252))

                        # Sharpe ratio (assuming 0% risk-free rate for simplicity)
                        if metrics['volatility'] > 0:
                            metrics['sharpe_ratio'] = float(metrics['annualized_return'] / metrics['volatility'])
                        else:
                            metrics['sharpe_ratio'] = 0.0

                        # Drawdown analysis
                        portfolio_df['cumulative_return'] = (1 + portfolio_df['daily_return']).cumprod()
                        portfolio_df['running_max'] = portfolio_df['cumulative_return'].cummax()
                        portfolio_df['drawdown'] = (portfolio_df['cumulative_return'] / portfolio_df['running_max']) - 1

                        metrics['max_drawdown'] = float(portfolio_df['drawdown'].min())
                        metrics['max_drawdown_pct'] = float(metrics['max_drawdown'] * 100)

                        # Calmar ratio (return / max drawdown)
                        if abs(metrics['max_drawdown']) > 0:
                            metrics['calmar_ratio'] = float(metrics['annualized_return'] / abs(metrics['max_drawdown']))
                        else:
                            metrics['calmar_ratio'] = float('inf') if metrics['annualized_return'] > 0 else 0.0

        # Calculate overall trading score (0-100)
        score_components = []

        # Win rate component (0-40 points)
        if 'win_rate' in metrics:
            win_rate_score = min(40.0, metrics['win_rate'] * 40)
            score_components.append(win_rate_score)

        # Profit factor component (0-30 points)
        if 'profit_factor' in metrics:
            # Cap at profit factor of 3 for scoring purposes
            capped_profit_factor = min(3.0, metrics['profit_factor'])
            profit_factor_score = capped_profit_factor * 10
            score_components.append(profit_factor_score)

        # Return vs volatility component (0-30 points)
        if 'sharpe_ratio' in metrics:
            # Cap at Sharpe ratio of 3 for scoring purposes
            capped_sharpe = min(3.0, metrics['sharpe_ratio'])
            sharpe_score = capped_sharpe * 10
            score_components.append(sharpe_score)

        # Calculate overall score
        if score_components:
            metrics['overall_score'] = sum(score_components)
        else:
            metrics['overall_score'] = 0

        # Add timestamp
        metrics['timestamp'] = datetime.now().isoformat()

        # Save to agent metrics
        if 'trading_agent' not in self.agent_metrics:
            self.agent_metrics['trading_agent'] = []
        self.agent_metrics['trading_agent'].append(metrics)

        # Update system metrics
        self.system_metrics['trading_performance'].append({
            'timestamp': metrics['timestamp'],
            'score': metrics['overall_score']
        })

        # Check for auto-save
        self._check_auto_save()

        return metrics

    def evaluate_communication(self, message_logs: List[Dict], time_period: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate communication efficiency between agents.

        Args:
            message_logs: List of message log entries
            time_period: Optional time period to filter logs ('day', 'week', etc.)

        Returns:
            Dictionary of communication metrics
        """
        metrics = {}

        if not message_logs:
            logger.warning("No message logs to evaluate")
            return {'error': 'No message logs'}

        # Convert to DataFrame
        logs_df = pd.DataFrame(message_logs)

        # Filter by time period if specified
        if time_period and 'timestamp' in logs_df.columns:
            # Convert to datetime if needed
            if not pd.api.types.is_datetime64_dtype(logs_df['timestamp']):
                logs_df['timestamp'] = pd.to_datetime(logs_df['timestamp'])

            # Apply time filter
            if time_period == 'day':
                start_time = datetime.now() - timedelta(days=1)
            elif time_period == 'week':
                start_time = datetime.now() - timedelta(days=7)
            elif time_period == 'month':
                start_time = datetime.now() - timedelta(days=30)
            else:
                start_time = datetime.now() - timedelta(days=1)  # Default to one day

            logs_df = logs_df[logs_df['timestamp'] >= start_time]

        # Basic message metrics
        metrics['total_messages'] = len(logs_df)

        # Message metrics by type
        if 'message_type' in logs_df.columns:
            type_counts = logs_df['message_type'].value_counts().to_dict()
            for msg_type, count in type_counts.items():
                metrics[f'{msg_type}_count'] = count

        # Agent communication metrics
        if 'sender_id' in logs_df.columns and 'receiver_id' in logs_df.columns:
            # Message count by agent
            sender_counts = logs_df['sender_id'].value_counts().to_dict()
            receiver_counts = logs_df['receiver_id'].value_counts().to_dict()

            metrics['messages_by_sender'] = sender_counts
            metrics['messages_by_receiver'] = receiver_counts

            # Create communication matrix
            comm_matrix = pd.crosstab(logs_df['sender_id'], logs_df['receiver_id'])
            metrics['communication_matrix'] = comm_matrix.to_dict()

        # Response time metrics (if available)
        if 'response_time' in logs_df.columns:
            response_times = logs_df['response_time'].dropna()
            if len(response_times) > 0:
                metrics['avg_response_time'] = float(response_times.mean())
                metrics['min_response_time'] = float(response_times.min())
                metrics['max_response_time'] = float(response_times.max())
                metrics['median_response_time'] = float(response_times.median())
                metrics['p95_response_time'] = float(np.percentile(response_times, 95))

        # Calculate efficiency score (0-100)
        # This is a simplified scoring approach
        score = 80  # Default good score

        # Penalize for response time issues
        if 'avg_response_time' in metrics:
            # Example: Penalize if average response time > 1 second
            if metrics['avg_response_time'] > 1000:  # Convert to ms
                score -= min(30.0, (metrics['avg_response_time'] - 1000) / 100)

        # Penalize for communication imbalance
        if 'communication_matrix' in metrics:
            # Identify communication imbalance
            num_agents = len(metrics['messages_by_sender'])
            if num_agents > 1:
                message_counts = list(metrics['messages_by_sender'].values())
                max_count = max(message_counts)
                min_count = min(message_counts)
                if max_count > min_count * 5:  # Arbitrary threshold
                    score -= 10

        metrics['efficiency_score'] = max(0, min(100, score))

        # Add timestamp
        metrics['timestamp'] = datetime.now().isoformat()

        # Save communication metrics
        if 'communication' not in self.agent_metrics:
            self.agent_metrics['communication'] = []
        self.agent_metrics['communication'].append(metrics)

        # Update system metrics
        self.system_metrics['communication_efficiency'].append({
            'timestamp': metrics['timestamp'],
            'score': metrics['efficiency_score']
        })

        # Check for auto-save
        self._check_auto_save()

        return metrics

    def evaluate_system_performance(self, runtime_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate system-wide performance metrics.

        Args:
            runtime_metrics: Dictionary of runtime performance metrics

        Returns:
            Dictionary of system evaluation metrics
        """
        metrics = {}

        # Add all runtime metrics
        metrics.update(runtime_metrics)

        # Get latest component scores
        component_scores = []

        # Add prediction accuracy if available
        if self.system_metrics['prediction_accuracy']:
            component_scores.append(self.system_metrics['prediction_accuracy'][-1]['score'])

        # Add trading performance if available
        if self.system_metrics['trading_performance']:
            component_scores.append(self.system_metrics['trading_performance'][-1]['score'])

        # Add communication efficiency if available
        if self.system_metrics['communication_efficiency']:
            component_scores.append(self.system_metrics['communication_efficiency'][-1]['score'])

        # Calculate runtime score (0-100)
        runtime_score = 80  # Default good score

        # Penalize for high memory usage
        if 'memory_usage_mb' in runtime_metrics:
            # Example: Penalize if memory usage > 1GB
            memory_usage = runtime_metrics['memory_usage_mb']
            if memory_usage > 1000:
                penalty = min(20, (memory_usage - 1000) / 200)
                runtime_score -= penalty

        # Penalize for high CPU usage
        if 'cpu_usage_percent' in runtime_metrics:
            # Example: Penalize if CPU usage > 70%
            cpu_usage = runtime_metrics['cpu_usage_percent']
            if cpu_usage > 70:
                penalty = min(20, (cpu_usage - 70) / 1.5)
                runtime_score -= penalty

        # Add runtime score to metrics and component scores
        metrics['runtime_score'] = max(0, min(100, runtime_score))
        component_scores.append(metrics['runtime_score'])

        # Calculate overall system score as average of component scores
        if component_scores:
            metrics['system_score'] = float(np.mean(component_scores))
        else:
            metrics['system_score'] = 0.0

        # Add timestamp
        metrics['timestamp'] = datetime.now().isoformat()

        # Update system metrics
        self.system_metrics['runtime_performance'].append({
            'timestamp': metrics['timestamp'],
            'score': metrics['runtime_score']
        })

        self.system_metrics['overall_score'].append({
            'timestamp': metrics['timestamp'],
            'score': metrics['system_score']
        })

        # Check for auto-save
        self._check_auto_save()

        return metrics

    def save_metrics(self, save_plots: bool = True) -> str:
        """
        Save current metrics to disk.

        Args:
            save_plots: Whether to also generate and save plots

        Returns:
            Path to the saved directory
        """
        # Create timestamped directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_dir = os.path.join(self.evaluation_dir, f'evaluation_{timestamp}')
        os.makedirs(save_dir, exist_ok=True)

        # Save agent metrics
        for agent_name, metrics_list in self.agent_metrics.items():
            agent_file = os.path.join(save_dir, f'{agent_name}_metrics.json')

            # Sample data if too large
            if len(metrics_list) > 1000:
                sampled_metrics = sample_time_series(metrics_list, max_points=1000)
                logger.info(f"Sampled {agent_name} metrics from {len(metrics_list)} to {len(sampled_metrics)} points")
                metrics_list = sampled_metrics

            with open(agent_file, 'w') as f:
                json.dump(metrics_list, f, indent=2)

        # Save system metrics
        system_file = os.path.join(save_dir, 'system_metrics.json')

        # Sample system metrics if too large
        for metric_type, metrics_list in self.system_metrics.items():
            if len(metrics_list) > 1000:
                self.system_metrics[metric_type] = sample_time_series(metrics_list, max_points=1000)

        with open(system_file, 'w') as f:
            json.dump(self.system_metrics, f, indent=2)

        # Generate and save plots if requested
        if save_plots:
            self._generate_plots(save_dir)

        logger.info(f"Evaluation metrics saved to {save_dir}")

        # Reset last save time
        self._last_save_time = time.time()

        return save_dir

    def _generate_plots(self, save_dir: str) -> None:
        """
        Generate evaluation plots and save them to disk.

        Args:
            save_dir: Directory to save plots
        """
        # Create plots directory
        plots_dir = os.path.join(save_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)

        # Create performance dashboard
        metrics_data = {
            'system_metrics': self._prepare_system_metrics_for_plotting(),
            'agent_contributions': self._calculate_agent_contributions(),
        }

        # Add agent-specific metrics if available
        if 'prediction_agent' in self.agent_metrics and self.agent_metrics['prediction_agent']:
            metrics_data['prediction_metrics'] = pd.DataFrame(self.agent_metrics['prediction_agent'])

        if 'trading_agent' in self.agent_metrics and self.agent_metrics['trading_agent']:
            metrics_data['trading_metrics'] = pd.DataFrame(self.agent_metrics['trading_agent'])

        # Create dashboard
        create_performance_dashboard(metrics_data, plots_dir, prefix="system")

    def _prepare_system_metrics_for_plotting(self) -> pd.DataFrame:
        """
        Prepare system metrics for plotting.

        Returns:
            DataFrame with system metrics
        """
        # Combine all metrics into a single DataFrame
        combined_data = []

        # Extract metrics from each category
        for metric_type, metrics_list in self.system_metrics.items():
            for metric in metrics_list:
                data_point = {
                    'timestamp': metric.get('timestamp'),
                    'metric_type': metric_type,
                    'score': metric.get('score', 0)
                }
                combined_data.append(data_point)

        if not combined_data:
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(combined_data)

        # Convert timestamp to datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Pivot to get metrics as columns
        df_pivoted = df.pivot_table(
            index='timestamp',
            columns='metric_type',
            values='score',
            aggfunc='last'  # Use last value if multiple values for same timestamp
        ).reset_index()

        return df_pivoted

    def _calculate_agent_contributions(self) -> Dict[str, float]:
        """
        Calculate the contribution of each agent to the overall system.

        Returns:
            Dictionary mapping agent names to contribution percentages
        """
        # Simple approach: use latest scores from each agent type
        agent_scores = {}

        # Extract latest scores for each agent type
        for agent_name, metrics_list in self.agent_metrics.items():
            if metrics_list:
                latest_metrics = metrics_list[-1]
                if 'overall_score' in latest_metrics:
                    agent_scores[agent_name] = latest_metrics['overall_score']

        # Normalize to percentages
        total_score = sum(agent_scores.values())
        if total_score > 0:
            contributions = {name: (score / total_score) * 100 for name, score in agent_scores.items()}
        else:
            # Equal contribution if no scores
            count = len(agent_scores)
            if count > 0:
                equal_share = 100 / count
                contributions = {name: equal_share for name in agent_scores.keys()}
            else:
                contributions = {}

        return contributions

    def _check_auto_save(self) -> None:
        """Check if it's time to auto-save."""
        current_time = time.time()
        if current_time - self._last_save_time > self._auto_save_interval:
            try:
                self.save_metrics(save_plots=False)  # Skip plots for auto-save to improve performance
            except Exception as e:
                logger.error(f"Error during auto-save: {e}")
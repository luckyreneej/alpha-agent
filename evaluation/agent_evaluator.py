import pandas as pd
import numpy as np
import json
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AgentEvaluator:
    """
    Evaluates the performance of individual agents and the overall multi-agent system.
    Tracks metrics over time and provides visualization tools.
    """
    
    def __init__(self, evaluation_dir: str = 'evaluation/results'):
        """
        Initialize the agent evaluator.
        
        Args:
            evaluation_dir: Directory for storing evaluation results
        """
        self.evaluation_dir = evaluation_dir
        os.makedirs(evaluation_dir, exist_ok=True)
        
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
        for ticker in common_tickers:
            pred = predictions[ticker]
            actual = actual_values[ticker]
            
            # Ensure we have numpy arrays
            if isinstance(pred, list):
                pred = np.array(pred)
            if isinstance(actual, list):
                actual = np.array(actual)
            
            # Calculate basic error metrics
            error = pred - actual
            squared_error = error ** 2
            abs_error = np.abs(error)
            
            # Mean metrics
            metrics[f'{ticker}_mse'] = float(np.mean(squared_error))
            metrics[f'{ticker}_rmse'] = float(np.sqrt(metrics[f'{ticker}_mse']))
            metrics[f'{ticker}_mae'] = float(np.mean(abs_error))
            metrics[f'{ticker}_mape'] = float(np.mean(np.abs(error / actual) * 100))
            
            # Direction accuracy
            if len(pred) > 1 and len(actual) > 1:
                pred_direction = np.sign(np.diff(pred))
                actual_direction = np.sign(np.diff(actual))
                directional_accuracy = np.mean(pred_direction == actual_direction)
                metrics[f'{ticker}_directional_accuracy'] = float(directional_accuracy)
            
            # Calculate rolling metrics if enough data
            if len(pred) >= window and len(actual) >= window:
                rolling_mse = []
                rolling_dir_acc = []
                
                for i in range(window, len(pred)):
                    window_pred = pred[i-window:i]
                    window_actual = actual[i-window:i]
                    
                    # MSE
                    window_mse = np.mean((window_pred - window_actual) ** 2)
                    rolling_mse.append(window_mse)
                    
                    # Directional accuracy
                    window_pred_dir = np.sign(np.diff(window_pred))
                    window_actual_dir = np.sign(np.diff(window_actual))
                    dir_acc = np.mean(window_pred_dir == window_actual_dir)
                    rolling_dir_acc.append(dir_acc)
                
                metrics[f'{ticker}_rolling_mse'] = rolling_mse
                metrics[f'{ticker}_rolling_dir_acc'] = rolling_dir_acc
        
        # Calculate aggregate metrics across all tickers
        mse_values = [metrics[f'{ticker}_mse'] for ticker in common_tickers]
        rmse_values = [metrics[f'{ticker}_rmse'] for ticker in common_tickers]
        mae_values = [metrics[f'{ticker}_mae'] for ticker in common_tickers]
        dir_acc_values = [metrics.get(f'{ticker}_directional_accuracy', 0) for ticker in common_tickers]
        
        metrics['avg_mse'] = float(np.mean(mse_values))
        metrics['avg_rmse'] = float(np.mean(rmse_values))
        metrics['avg_mae'] = float(np.mean(mae_values))
        metrics['avg_directional_accuracy'] = float(np.mean(
            [v for v in dir_acc_values if v > 0]))  # Only consider valid values
        
        # Calculate overall score (0-100)
        # Higher directional accuracy is better, lower error is better
        normalized_dir_acc = metrics['avg_directional_accuracy'] * 100  # 0-100 scale
        
        # Normalize error metrics (lower is better)
        # We use a simple scaling approach - this should be customized based on typical error ranges
        max_reasonable_error = np.max(rmse_values) * 2  # Adjust based on domain knowledge
        normalized_error = 100 * (1 - np.min([1, metrics['avg_rmse'] / max_reasonable_error]))
        
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
        metrics['profitable_trades'] = len(trades_df[trades_df['pnl'] > 0]) if 'pnl' in trades_df.columns else 0
        metrics['losing_trades'] = len(trades_df[trades_df['pnl'] < 0]) if 'pnl' in trades_df.columns else 0
        
        if 'pnl' in trades_df.columns and len(trades_df) > 0:
            metrics['win_rate'] = metrics['profitable_trades'] / metrics['total_trades']
            metrics['avg_profit'] = float(trades_df[trades_df['pnl'] > 0]['pnl'].mean()) if metrics['profitable_trades'] > 0 else 0
            metrics['avg_loss'] = float(trades_df[trades_df['pnl'] < 0]['pnl'].mean()) if metrics['losing_trades'] > 0 else 0
            metrics['profit_factor'] = abs(metrics['avg_profit'] / metrics['avg_loss']) if metrics['avg_loss'] != 0 else float('inf')
            metrics['net_profit'] = float(trades_df['pnl'].sum())
        
        # Portfolio metrics
        if 'portfolio_value' in portfolio_df.columns and len(portfolio_df) > 1:
            initial_value = portfolio_df['portfolio_value'].iloc[0]
            final_value = portfolio_df['portfolio_value'].iloc[-1]
            
            metrics['total_return'] = float((final_value - initial_value) / initial_value)
            metrics['total_return_pct'] = float(metrics['total_return'] * 100)
            
            # Calculate daily returns
            if 'date' in portfolio_df.columns:
                portfolio_df['date'] = pd.to_datetime(portfolio_df['date'])
                portfolio_df.set_index('date', inplace=True)
                portfolio_df.sort_index(inplace=True)
                portfolio_df['daily_return'] = portfolio_df['portfolio_value'].pct_change()
                
                # Annualized metrics
                days = (portfolio_df.index[-1] - portfolio_df.index[0]).days
                years = days / 365.25
                
                if years > 0:
                    metrics['annualized_return'] = float((1 + metrics['total_return']) ** (1 / years) - 1)
                    metrics['annualized_return_pct'] = float(metrics['annualized_return'] * 100)
                    
                    # Risk metrics
                    daily_returns = portfolio_df['daily_return'].dropna().values
                    metrics['volatility'] = float(np.std(daily_returns) * np.sqrt(252))
                    metrics['sharpe_ratio'] = float(metrics['annualized_return'] / metrics['volatility']) if metrics['volatility'] > 0 else 0
                    
                    # Drawdown
                    portfolio_df['cumulative_return'] = (1 + portfolio_df['daily_return']).cumprod()
                    portfolio_df['running_max'] = portfolio_df['cumulative_return'].cummax()
                    portfolio_df['drawdown'] = (portfolio_df['cumulative_return'] / portfolio_df['running_max']) - 1
                    metrics['max_drawdown'] = float(portfolio_df['drawdown'].min())
                    metrics['max_drawdown_pct'] = float(metrics['max_drawdown'] * 100)
                    
                    # Calmar ratio
                    if metrics['max_drawdown'] != 0:
                        metrics['calmar_ratio'] = float(metrics['annualized_return'] / abs(metrics['max_drawdown']))
        
        # Calculate overall trading score (0-100)
        score_components = []
        
        # Win rate component (0-40 points)
        if 'win_rate' in metrics:
            win_rate_score = min(40, metrics['win_rate'] * 40)
            score_components.append(win_rate_score)
        
        # Profit factor component (0-30 points)
        if 'profit_factor' in metrics:
            profit_factor_score = min(30, metrics['profit_factor'] * 10)  # Cap at 30
            score_components.append(profit_factor_score)
        
        # Return vs volatility component (0-30 points)
        if 'sharpe_ratio' in metrics:
            sharpe_score = min(30, metrics['sharpe_ratio'] * 10)  # Cap at 30
            score_components.append(sharpe_score)
        
        # Calculate overall score
        if score_components:
            metrics['overall_score'] = float(sum(score_components))
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
            logs_df['timestamp'] = pd.to_datetime(logs_df['timestamp'])
            
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
            metrics['avg_response_time'] = float(logs_df['response_time'].mean())
            metrics['min_response_time'] = float(logs_df['response_time'].min())
            metrics['max_response_time'] = float(logs_df['response_time'].max())
        
        # Calculate efficiency score (0-100)
        # This is a simplified score - customize based on your specific needs
        score = 80  # Default good score
        
        # Penalize for response time issues
        if 'avg_response_time' in metrics:
            # Example: Penalize if average response time > 1 second
            if metrics['avg_response_time'] > 1.0:
                score -= min(30, (metrics['avg_response_time'] - 1.0) * 10)
        
        # Penalize for communication imbalance
        if 'communication_matrix' in metrics:
            # Example: Slight penalty if communication is very imbalanced
            # (some agents doing much more work than others)
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
        
        # Calculate system score based on component scores
        component_scores = []
        
        # Get latest component scores
        if self.system_metrics['prediction_accuracy']:
            component_scores.append(self.system_metrics['prediction_accuracy'][-1]['score'])
        
        if self.system_metrics['trading_performance']:
            component_scores.append(self.system_metrics['trading_performance'][-1]['score'])
        
        if self.system_metrics['communication_efficiency']:
            component_scores.append(self.system_metrics['communication_efficiency'][-1]['score'])
        
        # Add runtime score (0-100)
        runtime_score = 80  # Default good score
        
        # Penalize for memory usage
        if 'memory_usage_mb' in runtime_metrics:
            # Example: Penalize if memory usage > 1GB
            if runtime_metrics['memory_usage_mb'] > 1000:
                runtime_score -= min(20, (runtime_metrics['memory_usage_mb'] - 1000) / 100)
        
        # Penalize for high CPU usage
        if 'cpu_usage_percent' in runtime_metrics:
            # Example: Penalize if CPU usage > 70%
            if runtime_metrics['cpu_usage_percent'] > 70:
                runtime_score -= min(20, (runtime_metrics['cpu_usage_percent'] - 70) / 2)
        
        # Add runtime score to components
        component_scores.append(runtime_score)
        
        # Calculate overall system score
        if component_scores:
            metrics['system_score'] = float(np.mean(component_scores))
        else:
            metrics['system_score'] = 0
        
        # Add timestamp
        metrics['timestamp'] = datetime.now().isoformat()
        
        # Update system metrics
        self.system_metrics['runtime_performance'].append({
            'timestamp': metrics['timestamp'],
            'score': runtime_score
        })
        
        self.system_metrics['overall_score'].append({
            'timestamp': metrics['timestamp'],
            'score': metrics['system_score']
        })
        
        return metrics
    
    def save_metrics(self, save_plots: bool = True):
        """
        Save current metrics to disk.
        
        Args:
            save_plots: Whether to also generate and save plots
        """
        # Create timestamped directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_dir = os.path.join(self.evaluation_dir, f'evaluation_{timestamp}')
        os.makedirs(save_dir, exist_ok=True)
        
        # Save agent metrics
        for agent_name, metrics_list in self.agent_metrics.items():
            agent_file = os.path.join(save_dir, f'{agent_name}_metrics.json')
            with open(agent_file, 'w') as f:
                json.dump(metrics_list, f, indent=2)
        
        # Save system metrics
        system_file = os.path.join(save_dir, 'system_metrics.json')
        with open(system_file, 'w') as f:
            json.dump(self.system_metrics, f, indent=2)
        
        # Generate and save plots if requested
        if save_plots:
            self._generate_plots(save_dir)
        
        logger.info(f"Evaluation metrics saved to {save_dir}")
        return save_dir
    
    def _generate_plots(self, save_dir: str):
        """
        Generate evaluation plots and save them to disk.
        
        Args:
            save_dir: Directory to save plots
        """
        # Create plots directory
        plots_dir = os.path.join(save_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Plot system scores over time
        self._plot_system_scores(os.path.join(plots_dir, 'system_scores.png'))
        
        # Plot prediction metrics if available
        if 'prediction_agent' in self.agent_metrics and self.agent_metrics['prediction_agent']:
            self._plot_prediction_metrics(os.path.join(plots_dir, 'prediction_metrics.png'))
        
        # Plot trading metrics if available
        if 'trading_agent' in self.agent_metrics and self.agent_metrics['trading_agent']:
            self._plot_trading_metrics(os.path.join(plots_dir, 'trading_metrics.png'))
    
    def _plot_system_scores(self, save_path: str):
        """
        Plot system scores over time.
        
        Args:
            save_path: Path to save the plot
        """
        plt.figure(figsize=(12, 8))
        
        # Convert timestamp strings to datetime for x-axis
        for metric_name in ['prediction_accuracy', 'trading_performance', 'communication_efficiency', 'overall_score']:
            if not self.system_metrics[metric_name]:
                continue
                
            data = pd.DataFrame(self.system_metrics[metric_name])
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            data.sort_values('timestamp', inplace=True)
            
            plt.plot(data['timestamp'], data['score'], marker='o', label=metric_name.replace('_', ' ').title())
        
        plt.title('System Performance Metrics Over Time', fontsize=14)
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Score (0-100)', fontsize=12)
        plt.ylim(0, 105)  # Slight margin above 100
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Format x-axis dates
        plt.gcf().autofmt_xdate()
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    
    def _plot_prediction_metrics(self, save_path: str):
        """
        Plot prediction metrics over time.
        
        Args:
            save_path: Path to save the plot
        """
        data = pd.DataFrame(self.agent_metrics['prediction_agent'])
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data.sort_values('timestamp', inplace=True)
        
        plt.figure(figsize=(12, 12))
        
        # Create a 2x2 grid of subplots
        plt.subplot(2, 2, 1)
        plt.plot(data['timestamp'], data['avg_directional_accuracy'], marker='o', color='green')
        plt.title('Directional Accuracy', fontsize=12)
        plt.xlabel('Time', fontsize=10)
        plt.ylabel('Accuracy (0-1)', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.gcf().autofmt_xdate()
        
        plt.subplot(2, 2, 2)
        plt.plot(data['timestamp'], data['avg_rmse'], marker='o', color='red')
        plt.title('Root Mean Squared Error', fontsize=12)
        plt.xlabel('Time', fontsize=10)
        plt.ylabel('RMSE', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.gcf().autofmt_xdate()
        
        plt.subplot(2, 2, 3)
        plt.plot(data['timestamp'], data['avg_mae'], marker='o', color='orange')
        plt.title('Mean Absolute Error', fontsize=12)
        plt.xlabel('Time', fontsize=10)
        plt.ylabel('MAE', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.gcf().autofmt_xdate()
        
        plt.subplot(2, 2, 4)
        plt.plot(data['timestamp'], data['overall_score'], marker='o', color='blue')
        plt.title('Overall Prediction Score', fontsize=12)
        plt.xlabel('Time', fontsize=10)
        plt.ylabel('Score (0-100)', fontsize=10)
        plt.ylim(0, 105)  # Slight margin above 100
        plt.grid(True, alpha=0.3)
        plt.gcf().autofmt_xdate()
        
        plt.suptitle('Prediction Agent Metrics Over Time', fontsize=14)
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)  # Make space for suptitle
        
        plt.savefig(save_path)
        plt.close()
    
    def _plot_trading_metrics(self, save_path: str):
        """
        Plot trading metrics over time.
        
        Args:
            save_path: Path to save the plot
        """
        data = pd.DataFrame(self.agent_metrics['trading_agent'])
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data.sort_values('timestamp', inplace=True)
        
        # Check which metrics are available
        metrics_to_plot = []
        for metric in ['win_rate', 'profit_factor', 'sharpe_ratio', 'total_return_pct']:
            if metric in data.columns:
                metrics_to_plot.append(metric)
        
        if not metrics_to_plot:
            # No meaningful metrics to plot
            return
        
        # Determine grid size based on number of metrics
        n_metrics = len(metrics_to_plot)
        if n_metrics <= 4:
            n_rows, n_cols = 2, 2
        else:
            n_rows = (n_metrics + 2) // 3  # Ceiling division
            n_cols = 3
        
        plt.figure(figsize=(12, 4 * n_rows))
        
        for i, metric in enumerate(metrics_to_plot, 1):
            plt.subplot(n_rows, n_cols, i)
            plt.plot(data['timestamp'], data[metric], marker='o')
            plt.title(metric.replace('_', ' ').title(), fontsize=12)
            plt.xlabel('Time', fontsize=10)
            plt.ylabel('Value', fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.gcf().autofmt_xdate()
        
        # Plot overall score in the last position
        if 'overall_score' in data.columns:
            plt.subplot(n_rows, n_cols, n_rows * n_cols)
            plt.plot(data['timestamp'], data['overall_score'], marker='o', color='blue')
            plt.title('Overall Trading Score', fontsize=12)
            plt.xlabel('Time', fontsize=10)
            plt.ylabel('Score (0-100)', fontsize=10)
            plt.ylim(0, 105)  # Slight margin above 100
            plt.grid(True, alpha=0.3)
            plt.gcf().autofmt_xdate()
        
        plt.suptitle('Trading Agent Metrics Over Time', fontsize=14)
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)  # Make space for suptitle
        
        plt.savefig(save_path)
        plt.close()
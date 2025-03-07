#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Optimized Contribution Analyzer Module

This module analyzes the contribution of individual agents to the overall system performance,
tracking their impact and effectiveness over time with improved performance and memory usage.
"""

import os
import json
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging

# Import utility modules
from metrics_utils import is_metric_better, calculate_time_window
from data_cache import DataCache
from visualization import plot_contribution_breakdown, plot_stacked_area_chart

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ContributionAnalyzer:
    """
    Analyzes the contribution of individual agents to overall system performance.
    Optimized for better performance and memory efficiency.
    """

    def __init__(self, data_dir: str = None):
        """
        Initialize the contribution analyzer.

        Args:
            data_dir: Directory containing metrics data files
        """
        self.data_dir = data_dir or os.path.join('data', 'metrics')
        os.makedirs(self.data_dir, exist_ok=True)

        # Initialize data cache
        self.data_cache = DataCache(self.data_dir)

        # Cache for contribution history
        self._contribution_history = {}
        self._last_update_time = datetime.now()

    def load_agent_data(self, agent_id: str = None, start_date: Optional[datetime] = None,
                        end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Load agent performance data efficiently using the data cache.

        Args:
            agent_id: Optional agent ID to filter by
            start_date: Optional start date
            end_date: Optional end date

        Returns:
            DataFrame containing agent performance data
        """
        # Define file pattern based on agent_id
        if agent_id:
            file_pattern = f"{agent_id}_*.json"
        else:
            file_pattern = "*.json"

        # Get list of files
        files = self.data_cache.list_files(file_pattern)

        # Filter out system report files
        files = [f for f in files if "system_report" not in os.path.basename(f)]

        if not files:
            logger.warning(f"No data files found for {'agent ' + agent_id if agent_id else 'any agent'}")
            return pd.DataFrame()

        # Parse files and collect data
        records = []

        for file_path in files:
            try:
                # Get data from cache
                data = self.data_cache.get_dataframe(file_path)

                if data.empty:
                    # Try loading directly if cache failed
                    with open(file_path, 'r') as f:
                        data = pd.json_normalize(json.load(f))

                if 'agent_id' not in data.columns or 'timestamp' not in data.columns:
                    continue

                # Extract timestamp
                if not pd.api.types.is_datetime64_dtype(data['timestamp']):
                    data['timestamp'] = pd.to_datetime(data['timestamp'])

                # Filter by date if specified
                if start_date and any(data['timestamp'] < start_date):
                    data = data[data['timestamp'] >= start_date]

                if end_date and any(data['timestamp'] > end_date):
                    data = data[data['timestamp'] <= end_date]

                if not data.empty:
                    records.append(data)

            except Exception as e:
                logger.error(f"Error parsing file {file_path}: {e}")
                continue

        if not records:
            return pd.DataFrame()

        try:
            # Combine all data
            df = pd.concat(records, ignore_index=True)

            # Sort by timestamp
            df = df.sort_values('timestamp')

            return df

        except Exception as e:
            logger.error(f"Error combining agent data: {e}")
            return pd.DataFrame()

    def calculate_agent_contributions(self, data: Optional[pd.DataFrame] = None,
                                      metric_weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Calculate the contribution of each agent to overall system performance.

        Args:
            data: DataFrame with agent performance data, or None to load from files
            metric_weights: Dictionary mapping metrics to their weights in the calculation

        Returns:
            Dictionary mapping agent IDs to contribution scores (0-100)
        """
        # Load data if not provided
        if data is None:
            data = self.load_agent_data()

        if data.empty:
            logger.warning("No data available for contribution analysis")
            return {}

        # Default metric weights if not provided
        if metric_weights is None:
            metric_weights = {
                'metrics.reliability.success_rate': 0.25,
                'metrics.reliability.error_rate': -0.20,  # Negative weight for errors
                'metrics.latency.mean_latency': -0.15,  # Negative weight for latency
                'metrics.throughput.throughput_ops_per_sec': 0.20,
                'metrics.accuracy.accuracy': 0.20
            }

        # Find available metrics in the data
        valid_metrics = [m for m in metric_weights.keys() if m in data.columns]

        if not valid_metrics:
            # Try alternative column formats (different JSON structure)
            alt_metrics = {}
            for metric in metric_weights:
                # Check variations of the metric name
                base_name = metric.split('.')[-1]
                for col in data.columns:
                    if base_name in col.lower():
                        alt_metrics[col] = metric_weights[metric]
                        break

            if alt_metrics:
                valid_metrics = list(alt_metrics.keys())
                metric_weights = alt_metrics
            else:
                logger.warning("None of the specified metrics found in data")
                return {}

        # Normalize weights to sum to 1.0
        total_weight = sum(abs(metric_weights[m]) for m in valid_metrics)
        if total_weight == 0:
            logger.warning("Total weight is zero, unable to calculate contributions")
            return {}

        normalized_weights = {m: metric_weights[m] / total_weight for m in valid_metrics}

        # Group by agent and calculate metrics
        agent_scores = {}

        # Process each agent
        for agent_id, group in data.groupby('agent_id'):
            metrics_data = {}

            for metric in valid_metrics:
                if metric in group.columns:
                    # Calculate average for this metric
                    values = group[metric].dropna()
                    if len(values) > 0:
                        metrics_data[metric] = values.mean()

            if metrics_data:
                agent_scores[agent_id] = metrics_data

        # Normalize metrics across agents
        normalized_scores = {}
        metric_min_max = {}

        # First pass: find min/max for each metric
        for metric in valid_metrics:
            values = [agent_data.get(metric) for agent_data in agent_scores.values()
                      if metric in agent_data and not pd.isna(agent_data[metric])]

            if values:
                metric_min_max[metric] = (min(values), max(values))

        # Second pass: normalize each metric
        for agent_id, agent_data in agent_scores.items():
            normalized_scores[agent_id] = {}

            for metric in valid_metrics:
                if metric in agent_data and metric in metric_min_max:
                    min_val, max_val = metric_min_max[metric]

                    # Skip if all values are the same
                    if min_val == max_val:
                        continue

                    # Determine if higher is better (positive weight) or lower is better (negative weight)
                    higher_is_better = metric_weights[metric] > 0

                    # Normalize the value
                    value = agent_data[metric]
                    if higher_is_better:
                        norm_value = (value - min_val) / (max_val - min_val)
                    else:
                        norm_value = 1 - ((value - min_val) / (max_val - min_val))

                    normalized_scores[agent_id][metric] = norm_value

        # Calculate final contribution scores
        contribution_scores = {}

        for agent_id, scores in normalized_scores.items():
            # Calculate weighted average
            weighted_sum = sum(scores.get(metric, 0) * normalized_weights.get(metric, 0)
                               for metric in valid_metrics)

            # Count number of valid metrics for this agent
            valid_count = sum(1 for metric in valid_metrics if metric in scores)

            # Adjust score based on metric coverage
            coverage_factor = valid_count / len(valid_metrics) if valid_metrics else 0

            # Final contribution score, scaled to 0-100
            contribution_scores[agent_id] = weighted_sum * coverage_factor * 100

        # Normalize to sum to 100%
        total_contribution = sum(contribution_scores.values())
        if total_contribution > 0:
            contribution_scores = {agent_id: (score / total_contribution) * 100
                                   for agent_id, score in contribution_scores.items()}

        # Update cache
        self._contribution_history[datetime.now().strftime('%Y-%m-%d')] = contribution_scores
        self._last_update_time = datetime.now()

        return contribution_scores

    def calculate_contribution_history(self, window_size: int = 7,
                                       step_size: int = 1) -> pd.DataFrame:
        """
        Calculate agent contributions over time using a rolling window.

        Args:
            window_size: Size of rolling window in days
            step_size: Step size for window in days

        Returns:
            DataFrame with contribution scores over time
        """
        # Load all agent data
        all_data = self.load_agent_data()

        if all_data.empty:
            logger.warning("No data available for contribution history")
            return pd.DataFrame()

        # Determine date range
        min_date = all_data['timestamp'].min().date()
        max_date = all_data['timestamp'].max().date()

        # Create windows - optimize for fewer windows
        windows = []
        current_date = min_date

        while current_date <= max_date:
            window_end = current_date + timedelta(days=window_size)
            windows.append((current_date, window_end))
            current_date += timedelta(days=step_size)

        # Calculate contributions for each window
        contributions_history = []

        for start_date, end_date in windows:
            # Convert dates to datetime for filtering
            start_datetime = datetime.combine(start_date, datetime.min.time())
            end_datetime = datetime.combine(end_date, datetime.max.time())

            # Filter data for this window
            window_data = all_data[(all_data['timestamp'] >= start_datetime) &
                                   (all_data['timestamp'] <= end_datetime)]

            if window_data.empty:
                continue

            # Calculate contributions for this window
            window_contributions = self.calculate_agent_contributions(window_data)

            # Add to history with window midpoint as timestamp
            window_mid = start_date + (end_date - start_date) / 2

            for agent_id, contribution in window_contributions.items():
                contributions_history.append({
                    'window_start': start_date,
                    'window_end': end_date,
                    'window_mid': window_mid,
                    'agent_id': agent_id,
                    'contribution': contribution
                })

        # Convert to DataFrame
        if not contributions_history:
            return pd.DataFrame()

        df = pd.DataFrame(contributions_history)

        # Add agent type if available
        agent_types = {}
        for _, row in all_data.iterrows():
            agent_id = row['agent_id']
            if 'agent_type' in row and agent_id not in agent_types:
                agent_types[agent_id] = row['agent_type']

        if agent_types:
            df['agent_type'] = df['agent_id'].map(agent_types)

        return df

    def calculate_adaptive_weights(self, lookback_days: int = 30) -> Dict[str, float]:
        """
        Calculate adaptive weights for agents based on their historical performance.

        Args:
            lookback_days: Number of days to look back

        Returns:
            Dictionary mapping agent IDs to weights (summing to 1.0)
        """
        # Define end date as now, start date as lookback days ago
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)

        # Load data within time range
        data = self.load_agent_data(start_date=start_date, end_date=end_date)

        if data.empty:
            logger.warning("No data available for adaptive weight calculation")
            return {}

        # Default performance metrics to consider
        performance_metrics = [
            'metrics.reliability.success_rate',
            'metrics.latency.mean_latency',
            'metrics.throughput.throughput_ops_per_sec',
            'metrics.accuracy.accuracy'
        ]

        # Find available metrics in the data
        available_metrics = []
        for base_metric in performance_metrics:
            # Check for exact match
            if base_metric in data.columns:
                available_metrics.append(base_metric)
                continue

            # Check for similar metrics
            metric_name = base_metric.split('.')[-1]
            for col in data.columns:
                if metric_name in col.lower():
                    available_metrics.append(col)
                    break

        if not available_metrics:
            logger.warning("No suitable metrics found for adaptive weighting")
            return {}

        # Group by agent and calculate scores
        agent_scores = {}

        for agent_id, group in data.groupby('agent_id'):
            # Calculate score for each available metric
            metric_scores = {}

            for metric in available_metrics:
                if metric in group.columns:
                    values = group[metric].dropna()
                    if len(values) > 0:
                        # Determine if higher or lower values are better
                        is_higher_better = is_metric_better(metric)

                        avg_value = values.mean()
                        metric_scores[metric] = avg_value if is_higher_better else (
                            1.0 / avg_value if avg_value != 0 else float('inf'))

            # Calculate overall score as average of metric scores
            if metric_scores:
                agent_scores[agent_id] = sum(metric_scores.values()) / len(metric_scores)

        # Convert scores to weights that sum to 1.0
        if not agent_scores:
            return {}

        total_score = sum(agent_scores.values())
        if total_score > 0:
            weights = {agent_id: score / total_score for agent_id, score in agent_scores.items()}
        else:
            # Equal weights if total score is zero
            count = len(agent_scores)
            weights = {agent_id: 1.0 / count for agent_id in agent_scores.keys()}

        return weights

    def plot_contribution_breakdown(self, data: Optional[pd.DataFrame] = None,
                                    output_file: Optional[str] = None) -> str:
        """
        Generate a visualization of agent contributions.

        Args:
            data: DataFrame containing agent data or None to load from files
            output_file: Path to save the plot or None for default path

        Returns:
            Path to the saved visualization file
        """
        # Calculate contributions if data not provided
        contributions = self.calculate_agent_contributions(data)

        if not contributions:
            logger.warning("No contribution data available for visualization")
            return ""

        # Use visualization module to generate the plot
        return plot_contribution_breakdown(
            contributions,
            title="Agent Contribution Analysis",
            output_path=output_file or os.path.join(self.data_dir,
                                                    f"contribution_breakdown_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        )

    def plot_contribution_over_time(self, window_size: int = 7, step_size: int = 1,
                                    output_file: Optional[str] = None) -> str:
        """
        Plot agent contributions over time.

        Args:
            window_size: Size of rolling window in days
            step_size: Step size for window in days
            output_file: Path to save the plot or None for default path

        Returns:
            Path to the saved visualization file
        """
        # Calculate contribution history
        history_df = self.calculate_contribution_history(window_size, step_size)

        if history_df.empty:
            logger.warning("No contribution history data available")
            return ""

        # Create default output file path if not provided
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.data_dir, f"contribution_time_{timestamp}.png")

        # Create stacked area chart using the visualization module
        # First pivot the data for plotting
        pivot_df = history_df.pivot(index='window_mid', columns='agent_id', values='contribution')

        # Sort columns by average contribution (largest first)
        col_means = pivot_df.mean()
        sorted_cols = col_means.sort_values(ascending=False).index

        return plot_stacked_area_chart(
            df=pivot_df.reset_index(),
            x_column='window_mid',
            y_columns=sorted_cols,
            title=f"Agent Contribution Over Time (Window: {window_size} days)",
            xlabel="Date",
            ylabel="Contribution (%)",
            output_path=output_file
        )

    def identify_key_contributors(self, threshold_pct: float = 10.0) -> List[Dict[str, Any]]:
        """
        Identify key contributors to system performance.

        Args:
            threshold_pct: Minimum contribution percentage to be considered key

        Returns:
            List of key contributors with their contribution metrics
        """
        # Calculate current contributions
        contributions = self.calculate_agent_contributions()

        if not contributions:
            return []

        # Find agents above threshold
        key_contributors = []
        for agent_id, contribution in contributions.items():
            if contribution >= threshold_pct:
                key_contributors.append({
                    'agent_id': agent_id,
                    'contribution_pct': contribution,
                    'is_critical': contribution >= 25.0  # Consider agents with >25% contribution critical
                })

        # Sort by contribution (highest first)
        key_contributors.sort(key=lambda x: x['contribution_pct'], reverse=True)

        return key_contributors

    def get_contribution_summary(self, time_range: Optional[str] = None) -> Dict[str, Any]:
        """
        Get a summary of agent contributions.

        Args:
            time_range: Optional time range like '7d', '30d', etc.

        Returns:
            Dictionary with contribution summary data
        """
        # Convert time range to start date
        start_date = None
        if time_range:
            delta = calculate_time_window(time_range)
            if delta:
                start_date = datetime.now() - delta

        # Calculate current contributions
        current_contributions = self.calculate_agent_contributions()

        # Get historical data if time range specified
        historical_data = None
        contribution_change = {}

        if start_date and current_contributions:
            # Load historical data
            historical_df = self.load_agent_data(start_date=start_date)

            if not historical_df.empty:
                # Get earliest data point after start date
                earliest_date = historical_df['timestamp'].min()

                # Filter data close to start date
                early_data = historical_df[historical_df['timestamp'] <= (earliest_date + timedelta(days=1))]

                if not early_data.empty:
                    # Calculate historical contributions
                    historical_contributions = self.calculate_agent_contributions(early_data)

                    # Calculate change for each agent
                    for agent_id, current_contrib in current_contributions.items():
                        if agent_id in historical_contributions:
                            historical_contrib = historical_contributions[agent_id]
                            contribution_change[agent_id] = current_contrib - historical_contrib

        # Identify key contributors
        key_contributors = self.identify_key_contributors()

        # Create summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'time_range': time_range,
            'agent_count': len(current_contributions),
            'contributions': current_contributions,
            'key_contributors': key_contributors,
            'contribution_change': contribution_change
        }

        return summary

    def get_agent_performance_impact(self, agent_id: str) -> Dict[str, Any]:
        """
        Analyze the performance impact of a specific agent.

        Args:
            agent_id: ID of the agent to analyze

        Returns:
            Dictionary with impact analysis data
        """
        # Load agent data
        agent_data = self.load_agent_data(agent_id=agent_id)

        if agent_data.empty:
            logger.warning(f"No data available for agent {agent_id}")
            return {'error': f"No data found for agent {agent_id}"}

        # Get current contribution
        all_contributions = self.calculate_agent_contributions()
        current_contribution = all_contributions.get(agent_id, 0)

        # Calculate contribution history for this agent
        history_df = self.calculate_contribution_history()

        agent_history = history_df[history_df['agent_id'] == agent_id] if not history_df.empty else pd.DataFrame()

        contribution_trend = []
        if not agent_history.empty:
            # Get last 10 data points
            recent_history = agent_history.sort_values('window_mid').tail(10)

            for _, row in recent_history.iterrows():
                contribution_trend.append({
                    'date': row['window_mid'].strftime('%Y-%m-%d'),
                    'contribution': row['contribution']
                })

        # Get agent metrics
        metrics = {}
        for col in agent_data.columns:
            if col.startswith('metrics.'):
                metric_parts = col.split('.')
                if len(metric_parts) >= 3:
                    category = metric_parts[1]
                    name = metric_parts[2]

                    if category not in metrics:
                        metrics[category] = {}

                    # Get latest value
                    latest_value = agent_data.sort_values('timestamp')[col].iloc[-1]
                    metrics[category][name] = latest_value

        # Create impact analysis
        impact_analysis = {
            'agent_id': agent_id,
            'agent_type': agent_data['agent_type'].iloc[0] if 'agent_type' in agent_data.columns else None,
            'current_contribution': current_contribution,
            'contribution_trend': contribution_trend,
            'metrics': metrics,
            'system_rank': list(all_contributions.keys()).index(
                agent_id) + 1 if agent_id in all_contributions else None,
            'total_agents': len(all_contributions)
        }

        return impact_analysis

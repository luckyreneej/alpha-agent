#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Optimized Performance Tracker Module

This module provides functionality for tracking and analyzing agent performance over time,
including historical performance logging, time-based comparisons, and trend analysis.
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import logging
from collections import defaultdict

# Import utility modules
from metrics_utils import sample_time_series, calculate_time_window
from data_cache import DataCache

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PerformanceTracker:
    """
    Tracks and analyzes agent performance over time with optimized
    data loading and memory usage.
    """

    def __init__(self, data_dir: str = None, max_cache_items: int = 50):
        """
        Initialize the performance tracker.

        Args:
            data_dir: Directory containing performance data files
            max_cache_items: Maximum number of dataframes to keep in memory
        """
        self.data_dir = data_dir or os.path.join('data', 'metrics')
        os.makedirs(self.data_dir, exist_ok=True)

        # Initialize data cache
        self.data_cache = DataCache(self.data_dir, max_size=max_cache_items)

        # Keep performance trend data
        self.performance_trends = {}

    def load_agent_history(self, agent_id: str, metric_types: List[str] = None,
                           start_date: Optional[datetime] = None,
                           end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Load historical performance data for an agent.

        Args:
            agent_id: Agent identifier
            metric_types: Optional list of metric types to include
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            DataFrame with historical performance metrics
        """
        # Define date filters
        start_date = start_date or datetime.min
        end_date = end_date or datetime.max

        # Find all metric files for this agent
        file_pattern = f"{agent_id}_*.json"
        metric_files = self.data_cache.list_files(file_pattern)

        if not metric_files:
            logger.warning(f"No metric files found for agent {agent_id}")
            return pd.DataFrame()

        # Load and combine data
        data_list = []
        for file_path in metric_files:
            try:
                # Load data from cache
                df = self.data_cache.get_dataframe(file_path)

                if df.empty:
                    # Try loading directly if not in cache
                    with open(file_path, 'r') as f:
                        data = json.load(f)

                    # Check if it's a single record or list
                    if isinstance(data, dict):
                        # Convert to DataFrame and flatten metrics
                        flattened = {
                            'timestamp': data.get('timestamp'),
                            'agent_id': data.get('agent_id'),
                            'agent_type': data.get('agent_type')
                        }

                        # Extract metrics with flattened keys
                        metrics = data.get('metrics', {})
                        for category, category_metrics in metrics.items():
                            for metric_name, metric_value in category_metrics.items():
                                flattened[f"metrics.{category}.{metric_name}"] = metric_value

                        df = pd.DataFrame([flattened])
                    elif isinstance(data, list):
                        # Multiple records
                        flattened_list = []
                        for record in data:
                            flattened = {
                                'timestamp': record.get('timestamp'),
                                'agent_id': record.get('agent_id', agent_id)
                            }

                            # Extract metrics with flattened keys
                            metrics = record.get('metrics', {})
                            for category, category_metrics in metrics.items():
                                for metric_name, metric_value in category_metrics.items():
                                    flattened[f"metrics.{category}.{metric_name}"] = metric_value

                            flattened_list.append(flattened)

                        df = pd.DataFrame(flattened_list)

                # Parse timestamp
                if 'timestamp' in df.columns and not pd.api.types.is_datetime64_dtype(df['timestamp']):
                    df['timestamp'] = pd.to_datetime(df['timestamp'])

                # Skip if outside date range
                if any(df['timestamp'] < start_date) or any(df['timestamp'] > end_date):
                    df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]

                    if df.empty:
                        continue

                # Filter metric types if specified
                if metric_types:
                    # Keep only columns that match the specified metric types
                    keep_cols = ['timestamp', 'agent_id', 'agent_type']
                    for col in df.columns:
                        if any(col.startswith(f"metrics.{metric_type}") for metric_type in metric_types):
                            keep_cols.append(col)

                    df = df[keep_cols]

                data_list.append(df)

            except Exception as e:
                logger.error(f"Error loading metrics from {file_path}: {e}")

        # Create DataFrame
        if not data_list:
            logger.warning(f"No data found for agent {agent_id} in specified date range")
            return pd.DataFrame()

        df = pd.concat(data_list, ignore_index=True)

        # Sort by timestamp
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp')

        return df

    def load_system_history(self, metric_names: List[str] = None,
                            start_date: Optional[datetime] = None,
                            end_date: Optional[datetime] = None) -> Dict[str, pd.DataFrame]:
        """
        Load historical performance data for all agents in the system.

        Args:
            metric_names: Optional list of metric names to include
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            Dictionary mapping agent IDs to performance DataFrames
        """
        # Find all unique agent IDs from metric files
        file_pattern = "*.json"
        metric_files = self.data_cache.list_files(file_pattern)

        # Extract unique agent IDs
        agent_ids = set()
        for file_path in metric_files:
            file_name = os.path.basename(file_path)

            # Skip system reports
            if file_name.startswith("system_report"):
                continue

            parts = file_name.split('_')
            if len(parts) >= 1:
                agent_id = parts[0]  # Assuming filename format is agent_id_date.json
                agent_ids.add(agent_id)

        # Load data for each agent
        system_data = {}
        for agent_id in agent_ids:
            df = self.load_agent_history(
                agent_id,
                start_date=start_date,
                end_date=end_date
            )

            if not df.empty:
                # Filter to requested metric names if specified
                if metric_names:
                    # Keep all columns that contain any of the metric names
                    keep_cols = ['timestamp', 'agent_id', 'agent_type']
                    for col in df.columns:
                        if any(metric_name in col for metric_name in metric_names):
                            keep_cols.append(col)

                    df = df[keep_cols]

                system_data[agent_id] = df

        return system_data

    def calculate_performance_trends(self, agent_id: str,
                                     metric_names: List[str],
                                     window_days: int = 7) -> Dict[str, Any]:
        """
        Calculate performance trends for specific metrics.

        Args:
            agent_id: Agent identifier
            metric_names: List of metrics to analyze
            window_days: Window size for trend calculation

        Returns:
            Dictionary with trend analysis
        """
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=window_days * 2)  # Get twice the window for trend

        # Load data
        df = self.load_agent_history(agent_id, start_date=start_date, end_date=end_date)

        if df.empty:
            logger.warning(f"No data available for trend analysis of agent {agent_id}")
            return {}

        trends = {}

        for metric_name in metric_names:
            # Find matching columns
            matching_cols = [col for col in df.columns if metric_name in col]

            for col in matching_cols:
                if col in df.columns:
                    # Ensure column has data
                    values = df[col].dropna()

                    if len(values) < 2:
                        # Need at least 2 points for trend
                        continue

                    # Split into two windows
                    midpoint = len(values) // 2
                    first_half = values.iloc[:midpoint]
                    second_half = values.iloc[midpoint:]

                    # Calculate averages
                    first_avg = first_half.mean()
                    second_avg = second_half.mean()

                    # Calculate trend percentage
                    if first_avg != 0:
                        trend_pct = ((second_avg - first_avg) / abs(first_avg)) * 100
                    else:
                        trend_pct = 0 if second_avg == 0 else float('inf')

                    # Determine if higher values are better for this metric
                    higher_is_better = not any(keyword in col.lower()
                                               for keyword in ['error', 'latency', 'time', 'failure'])

                    # Determine if trend is positive or negative
                    is_positive = (trend_pct > 0 and higher_is_better) or (trend_pct < 0 and not higher_is_better)

                    trends[col] = {
                        'first_period_avg': float(first_avg),
                        'second_period_avg': float(second_avg),
                        'trend_pct': float(trend_pct),
                        'is_positive': is_positive,
                        'samples': len(values)
                    }

        # Save to cache
        self.performance_trends[agent_id] = {
            'timestamp': datetime.now().isoformat(),
            'window_days': window_days,
            'trends': trends
        }

        return trends

    def compare_agents(self, agent_ids: List[str],
                       metric_names: List[str],
                       time_range: str = '7d') -> Dict[str, Any]:
        """
        Compare performance of multiple agents.

        Args:
            agent_ids: List of agent IDs to compare
            metric_names: List of metric names to compare
            time_range: Time range for comparison ('1d', '7d', '30d', etc.)

        Returns:
            Dictionary with comparison results
        """
        # Calculate date range
        delta = calculate_time_window(time_range)
        start_date = None

        if delta:
            start_date = datetime.now() - delta

        # Load data for each agent
        agent_data = {}
        for agent_id in agent_ids:
            df = self.load_agent_history(agent_id, start_date=start_date)
            if not df.empty:
                agent_data[agent_id] = df

        if not agent_data:
            logger.warning("No data available for comparison")
            return {}

        # Prepare comparison results
        comparison = {
            'time_range': time_range,
            'metrics': {},
            'agent_ranks': defaultdict(dict)
        }

        # Compare each metric
        for metric_name in metric_names:
            metric_results = {}

            # Find all matching metric columns across agents
            all_matching_cols = set()
            for df in agent_data.values():
                matching_cols = [col for col in df.columns if metric_name in col]
                all_matching_cols.update(matching_cols)

            # Process each matching column
            for col in all_matching_cols:
                col_results = {}
                values_by_agent = {}

                # Extract values for each agent
                for agent_id, df in agent_data.items():
                    if col in df.columns:
                        values = df[col].dropna()
                        if len(values) > 0:
                            avg_value = float(values.mean())
                            values_by_agent[agent_id] = avg_value

                if not values_by_agent:
                    continue

                # Determine if higher values are better for this metric
                higher_is_better = not any(keyword in col.lower()
                                           for keyword in ['error', 'latency', 'time', 'failure'])

                # Sort agents by value
                sorted_agents = sorted(
                    values_by_agent.items(),
                    key=lambda x: x[1],
                    reverse=higher_is_better
                )

                # Store results
                for rank, (agent_id, value) in enumerate(sorted_agents, 1):
                    col_results[agent_id] = {
                        'value': value,
                        'rank': rank
                    }

                    # Update agent ranks
                    comparison['agent_ranks'][agent_id][col] = rank

                # Add to metrics
                comparison['metrics'][col] = {
                    'results': col_results,
                    'higher_is_better': higher_is_better,
                    'best_agent': sorted_agents[0][0] if sorted_agents else None,
                    'worst_agent': sorted_agents[-1][0] if sorted_agents else None
                }

        # Calculate overall rankings
        overall_scores = defaultdict(int)
        metric_count = len(comparison['metrics'])

        if metric_count > 0:
            for agent_id, ranks in comparison['agent_ranks'].items():
                # Average rank across all metrics
                overall_scores[agent_id] = sum(ranks.values()) / len(ranks)

            # Sort by overall score (lower is better for ranks)
            sorted_overall = sorted(overall_scores.items(), key=lambda x: x[1])

            comparison['overall_ranking'] = {
                agent_id: {'rank': rank, 'avg_rank': score}
                for rank, (agent_id, score) in enumerate(sorted_overall, 1)
            }

        return comparison

    def detect_performance_anomalies(self, agent_id: str,
                                     time_range: str = '7d',
                                     threshold: float = 2.0) -> List[Dict[str, Any]]:
        """
        Detect performance anomalies for an agent.

        Args:
            agent_id: Agent identifier
            time_range: Time range to analyze
            threshold: Standard deviation threshold for anomaly detection

        Returns:
            List of detected anomalies
        """
        # Calculate date range
        delta = calculate_time_window(time_range)
        start_date = None

        if delta:
            start_date = datetime.now() - delta

        # Load data
        df = self.load_agent_history(agent_id, start_date=start_date)

        if df.empty:
            logger.warning(f"No data available for anomaly detection of agent {agent_id}")
            return []

        anomalies = []

        # Process each metric column
        for col in df.columns:
            # Skip non-metric columns
            if not col.startswith('metrics.'):
                continue

            # Get values
            values = df[col].dropna()

            if len(values) < 5:  # Need enough data points
                continue

            # Calculate statistics
            mean_value = values.mean()
            std_dev = values.std()

            if std_dev == 0:  # Skip if no variation
                continue

            # Find anomalies (values beyond threshold standard deviations)
            for idx, value in values.items():
                z_score = abs(value - mean_value) / std_dev

                if z_score > threshold:
                    # Get timestamp
                    timestamp = df.loc[idx, 'timestamp'] if 'timestamp' in df.columns else None

                    anomalies.append({
                        'metric': col,
                        'timestamp': timestamp.isoformat() if timestamp else None,
                        'value': float(value),
                        'mean': float(mean_value),
                        'std_dev': float(std_dev),
                        'z_score': float(z_score),
                        'is_high': value > mean_value
                    })

        # Sort by z-score (highest first)
        anomalies.sort(key=lambda x: x['z_score'], reverse=True)

        return anomalies

    def get_performance_summary(self, agent_id: str = None,
                                time_range: str = '7d') -> Dict[str, Any]:
        """
        Get a performance summary for an agent or the whole system.

        Args:
            agent_id: Optional agent to summarize (None for system summary)
            time_range: Time range for the summary

        Returns:
            Dictionary with performance summary
        """
        # Calculate date range
        delta = calculate_time_window(time_range)
        start_date = None

        if delta:
            start_date = datetime.now() - delta

        # Load data
        if agent_id:
            # Single agent summary
            data = {agent_id: self.load_agent_history(agent_id, start_date=start_date)}
        else:
            # System summary
            data = self.load_system_history(start_date=start_date)

        if not data:
            logger.warning(f"No data available for {'agent ' + agent_id if agent_id else 'system'} summary")
            return {'error': 'No data available for the specified parameters'}

        summary = {
            'timestamp': datetime.now().isoformat(),
            'time_range': time_range,
            'agent_count': len(data),
            'agents': {}
        }

        # System-wide metrics
        if not agent_id:
            total_messages = 0
            total_operations = 0
            avg_response_times = []
            avg_latencies = []

            for agent_id, df in data.items():
                # Count messages
                msg_sent_col = next((col for col in df.columns if 'messages_sent' in col.lower()), None)
                msg_received_col = next((col for col in df.columns if 'messages_received' in col.lower()), None)

                if msg_sent_col and msg_sent_col in df.columns:
                    total_messages += df[msg_sent_col].sum()

                if msg_received_col and msg_received_col in df.columns:
                    total_messages += df[msg_received_col].sum()

                # Count operations
                success_col = next((col for col in df.columns if 'success' in col.lower() and 'count' in col.lower()),
                                   None)
                error_col = next((col for col in df.columns if 'error' in col.lower() and 'count' in col.lower()), None)

                if success_col and success_col in df.columns:
                    total_operations += df[success_col].sum()

                if error_col and error_col in df.columns:
                    total_operations += df[error_col].sum()

                # Collect response times
                resp_time_col = next(
                    (col for col in df.columns if 'response_time' in col.lower() and 'mean' in col.lower()), None)
                if resp_time_col and resp_time_col in df.columns:
                    avg_val = df[resp_time_col].mean()
                    if not pd.isna(avg_val):
                        avg_response_times.append(avg_val)

                # Collect latencies
                latency_col = next((col for col in df.columns if 'latency' in col.lower() and 'mean' in col.lower()),
                                   None)
                if latency_col and latency_col in df.columns:
                    avg_val = df[latency_col].mean()
                    if not pd.isna(avg_val):
                        avg_latencies.append(avg_val)

            # Add system metrics
            summary['system_metrics'] = {
                'total_messages': int(total_messages),
                'total_operations': int(total_operations),
                'avg_response_time': float(np.mean(avg_response_times)) if avg_response_times else None,
                'avg_latency': float(np.mean(avg_latencies)) if avg_latencies else None,
            }

        # Process each agent
        for agent_id, df in data.items():
            if df.empty:
                continue

            agent_summary = {
                'agent_type': df['agent_type'].iloc[0] if 'agent_type' in df.columns else None,
                'metrics': {}
            }

            # Extract key metrics by category
            metric_categories = set()
            for col in df.columns:
                if col.startswith('metrics.'):
                    parts = col.split('.')
                    if len(parts) >= 2:
                        metric_categories.add(parts[1])

            for category in metric_categories:
                category_metrics = {}

                # Find all metrics for this category
                for col in df.columns:
                    if col.startswith(f'metrics.{category}.'):
                        metric_name = col.split('.')[-1]
                        values = df[col].dropna()

                        if not values.empty:
                            category_metrics[metric_name] = {
                                'avg': float(values.mean()),
                                'min': float(values.min()),
                                'max': float(values.max()),
                                'latest': float(values.iloc[-1])
                            }

                if category_metrics:
                    agent_summary['metrics'][category] = category_metrics

            # Add to summary
            summary['agents'][agent_id] = agent_summary

            # Add anomalies for individual agent summaries
            if agent_id == agent_id:  # This is just for the single agent case
                anomalies = self.detect_performance_anomalies(agent_id, time_range)
                if anomalies:
                    agent_summary['anomalies'] = anomalies

        return summary

    def create_performance_baseline(self, agent_id: str,
                                    baseline_days: int = 30) -> Dict[str, Any]:
        """
        Create a performance baseline for future comparisons.

        Args:
            agent_id: Agent identifier
            baseline_days: Number of days of data to use for baseline

        Returns:
            Dictionary with baseline metrics
        """
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=baseline_days)

        # Load data
        df = self.load_agent_history(agent_id, start_date=start_date, end_date=end_date)

        if df.empty:
            logger.warning(f"No data available for baseline creation of agent {agent_id}")
            return {}

        baseline = {
            'agent_id': agent_id,
            'created_at': datetime.now().isoformat(),
            'baseline_range': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat(),
                'days': baseline_days
            },
            'metrics': {}
        }

        # Process metric columns
        for col in df.columns:
            if not col.startswith('metrics.'):
                continue

            values = df[col].dropna()
            if values.empty:
                continue

            # Calculate baseline statistics
            baseline['metrics'][col] = {
                'mean': float(values.mean()),
                'median': float(values.median()),
                'std_dev': float(values.std()),
                'min': float(values.min()),
                'max': float(values.max()),
                'p25': float(np.percentile(values, 25)),
                'p75': float(np.percentile(values, 75)),
                'p95': float(np.percentile(values, 95))
            }

        # Save baseline
        baseline_file = os.path.join(self.data_dir, f"{agent_id}_baseline.json")

        with open(baseline_file, 'w') as f:
            json.dump(baseline, f, indent=2)

        logger.info(f"Created performance baseline for agent {agent_id}")

        return baseline

    def compare_to_baseline(self, agent_id: str, time_range: str = '1d') -> Dict[str, Any]:
        """
        Compare current performance to the established baseline.

        Args:
            agent_id: Agent identifier
            time_range: Time range for current performance

        Returns:
            Dictionary with comparison results
        """
        # Load baseline
        baseline_file = os.path.join(self.data_dir, f"{agent_id}_baseline.json")

        if not os.path.exists(baseline_file):
            logger.warning(f"No baseline found for agent {agent_id}")
            return {'error': f"No baseline found for agent {agent_id}"}

        try:
            with open(baseline_file, 'r') as f:
                baseline = json.load(f)
        except Exception as e:
            logger.error(f"Error loading baseline for agent {agent_id}: {e}")
            return {'error': f"Error loading baseline: {str(e)}"}

        # Calculate date range for current performance
        delta = calculate_time_window(time_range)
        start_date = None

        if delta:
            start_date = datetime.now() - delta

        # Load current data
        df = self.load_agent_history(agent_id, start_date=start_date)

        if df.empty:
            logger.warning(f"No current data available for agent {agent_id}")
            return {'error': 'No current data available'}

        # Compare metrics
        comparison = {
            'agent_id': agent_id,
            'timestamp': datetime.now().isoformat(),
            'time_range': time_range,
            'baseline_date': baseline.get('created_at'),
            'metrics': {},
            'summary': {'improved': 0, 'degraded': 0, 'unchanged': 0}
        }

        for col, baseline_stats in baseline.get('metrics', {}).items():
            if col not in df.columns:
                continue

            values = df[col].dropna()
            if values.empty:
                continue

            # Calculate current statistics
            current_mean = float(values.mean())

            # Compare to baseline
            baseline_mean = baseline_stats.get('mean')
            if baseline_mean is None:
                continue

            # Calculate percent change
            percent_change = ((current_mean - baseline_mean) / abs(baseline_mean)) * 100 if baseline_mean != 0 else 0

            # Determine if higher is better
            higher_is_better = not any(keyword in col.lower()
                                       for keyword in ['error', 'latency', 'time', 'failure'])

            # Determine if performance improved or degraded
            status = 'unchanged'
            if abs(percent_change) > 5:  # 5% threshold for meaningful change
                if (percent_change > 0 and higher_is_better) or (percent_change < 0 and not higher_is_better):
                    status = 'improved'
                    comparison['summary']['improved'] += 1
                else:
                    status = 'degraded'
                    comparison['summary']['degraded'] += 1
            else:
                comparison['summary']['unchanged'] += 1

            # Add to comparison
            comparison['metrics'][col] = {
                'baseline': baseline_mean,
                'current': current_mean,
                'percent_change': percent_change,
                'status': status,
                'higher_is_better': higher_is_better
            }

        # Calculate overall status
        if comparison['summary']['degraded'] > comparison['summary']['improved']:
            comparison['overall_status'] = 'degraded'
        elif comparison['summary']['improved'] > comparison['summary']['degraded']:
            comparison['overall_status'] = 'improved'
        else:
            comparison['overall_status'] = 'unchanged'

        return comparison

    def detect_performance_shifts(self, agent_id: str, window_days: int = 30,
                                  min_samples: int = 10) -> Dict[str, Any]:
        """
        Detect gradual performance shifts over time.

        Args:
            agent_id: Agent identifier
            window_days: Number of days to analyze
            min_samples: Minimum number of samples required for analysis

        Returns:
            Dictionary with detected shifts
        """
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=window_days)

        # Load data
        df = self.load_agent_history(agent_id, start_date=start_date, end_date=end_date)

        if df.empty:
            logger.warning(f"No data available for shift detection of agent {agent_id}")
            return {}

        # Ensure timestamp is datetime
        if 'timestamp' in df.columns and not pd.api.types.is_datetime64_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Analyze metric columns
        shifts = {}

        for col in df.columns:
            if not col.startswith('metrics.'):
                continue

            values = df[col].dropna()
            if len(values) < min_samples:
                continue

            # Skip if no timestamp column
            if 'timestamp' not in df.columns:
                continue

            # Create time series
            ts_data = pd.Series(values.values, index=df.loc[values.index, 'timestamp'])

            # Sort by timestamp
            ts_data = ts_data.sort_index()

            # Split into three periods
            period_len = len(ts_data) // 3
            if period_len < 3:  # Need at least 3 points per period
                continue

            early = ts_data.iloc[:period_len]
            middle = ts_data.iloc[period_len:2 * period_len]
            recent = ts_data.iloc[2 * period_len:]

            # Calculate averages
            early_avg = early.mean()
            middle_avg = middle.mean()
            recent_avg = recent.mean()

            # Calculate changes
            early_to_middle = ((middle_avg - early_avg) / abs(early_avg)) * 100 if early_avg != 0 else 0
            middle_to_recent = ((recent_avg - middle_avg) / abs(middle_avg)) * 100 if middle_avg != 0 else 0
            overall_change = ((recent_avg - early_avg) / abs(early_avg)) * 100 if early_avg != 0 else 0

            # Determine if shift is consistent
            consistent_direction = (early_to_middle > 0 and middle_to_recent > 0) or \
                                   (early_to_middle < 0 and middle_to_recent < 0)

            # Only consider significant changes
            if abs(overall_change) > 10 and consistent_direction:
                # Determine if higher is better
                higher_is_better = not any(keyword in col.lower()
                                           for keyword in ['error', 'latency', 'time', 'failure'])

                # Determine status
                status = 'improving' if (overall_change > 0 and higher_is_better) or \
                                        (overall_change < 0 and not higher_is_better) else 'degrading'

                shifts[col] = {
                    'early_avg': float(early_avg),
                    'middle_avg': float(middle_avg),
                    'recent_avg': float(recent_avg),
                    'overall_change': float(overall_change),
                    'early_to_middle': float(early_to_middle),
                    'middle_to_recent': float(middle_to_recent),
                    'status': status,
                    'higher_is_better': higher_is_better
                }

        # Return shifts ordered by magnitude of change
        return {
            'agent_id': agent_id,
            'window_days': window_days,
            'timestamp': datetime.now().isoformat(),
            'shifts': dict(sorted(shifts.items(), key=lambda x: abs(x[1]['overall_change']), reverse=True))
        }

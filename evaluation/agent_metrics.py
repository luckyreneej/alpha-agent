#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Optimized Agent Metrics Module

This module provides performance metrics and tracking functionality for individual agents
in the Alpha-Agent system. It includes accuracy, latency, throughput, and reliability metrics
with optimized memory usage and data processing.
"""

import time
import numpy as np
import json
import os
import uuid
import logging
from datetime import datetime
from typing import Dict, Any, Optional, Union
from collections import deque

# Import utility modules
from metrics_utils import sample_time_series
from data_cache import DataCache

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AgentMetrics:
    """
    Tracks and calculates performance metrics for individual agents.
    Optimized for memory efficiency and faster calculations.
    """

    def __init__(self, agent_id: str, agent_type: str, metrics_dir: str = None,
                 max_history: int = 10000):
        """
        Initialize agent metrics tracker.

        Args:
            agent_id: Unique identifier for the agent
            agent_type: Type of agent (Market Analyst, Predictive Model, etc.)
            metrics_dir: Directory to store metrics data
            max_history: Maximum number of metrics to keep in memory
        """
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.metrics_dir = metrics_dir or os.path.join('data', 'metrics')
        self.max_history = max_history

        # Ensure metrics directory exists
        os.makedirs(self.metrics_dir, exist_ok=True)

        # Initialize data cache
        self.data_cache = DataCache(self.metrics_dir)

        # Initialize metrics storage using deques for memory efficiency
        self.execution_times = deque(maxlen=max_history)
        self.error_counts = 0
        self.success_counts = 0
        self.prediction_actual_values = deque(maxlen=max_history)
        self.task_metrics = {}
        self.messages_sent = 0
        self.messages_received = 0

        # Track request/response metrics
        self.request_timestamps = {}
        self.response_times = deque(maxlen=max_history)

        # Daily metrics storage
        self.daily_metrics = {}

        # Last save timestamp
        self.last_save_time = time.time()
        self.auto_save_interval = 300  # 5 minutes

        logger.info(f"Initialized metrics tracking for agent {agent_id} of type {agent_type}")

    def record_execution_time(self, task_id: str, execution_time_ms: float) -> None:
        """
        Record the execution time for a task.

        Args:
            task_id: Identifier for the task
            execution_time_ms: Execution time in milliseconds
        """
        self.execution_times.append(execution_time_ms)

        # Store by task ID with memory-efficient approach
        if task_id not in self.task_metrics:
            self.task_metrics[task_id] = {
                'execution_times': deque(maxlen=self.max_history),
                'errors': 0,
                'successes': 0
            }

        self.task_metrics[task_id]['execution_times'].append(execution_time_ms)
        self._check_auto_save()

    def record_error(self, task_id: str, error_type: str, details: Optional[Dict] = None) -> None:
        """
        Record an error occurrence.

        Args:
            task_id: Identifier for the task
            error_type: Type of error
            details: Additional error details
        """
        self.error_counts += 1

        if task_id not in self.task_metrics:
            self.task_metrics[task_id] = {
                'execution_times': deque(maxlen=self.max_history),
                'errors': 0,
                'successes': 0
            }

        self.task_metrics[task_id]['errors'] += 1

        # Log detailed error information
        error_info = {
            'timestamp': datetime.now().isoformat(),
            'agent_id': self.agent_id,
            'task_id': task_id,
            'error_type': error_type
        }

        if details:
            error_info.update(details)

        # Append to error log file
        error_log_path = os.path.join(self.metrics_dir, f"{self.agent_id}_errors.jsonl")
        with open(error_log_path, 'a') as f:
            f.write(json.dumps(error_info) + '\n')

        self._check_auto_save()

    def record_success(self, task_id: str) -> None:
        """
        Record a successful task execution.

        Args:
            task_id: Identifier for the task
        """
        self.success_counts += 1

        if task_id not in self.task_metrics:
            self.task_metrics[task_id] = {
                'execution_times': deque(maxlen=self.max_history),
                'errors': 0,
                'successes': 0
            }

        self.task_metrics[task_id]['successes'] += 1
        self._check_auto_save()

    def record_prediction(self, predicted_value: Union[float, list], actual_value: Union[float, list],
                          timestamp: Optional[datetime] = None, metadata: Optional[Dict] = None) -> None:
        """
        Record a prediction and its actual outcome for accuracy tracking.

        Args:
            predicted_value: The predicted value
            actual_value: The actual value (ground truth)
            timestamp: Optional timestamp of prediction
            metadata: Additional metadata about the prediction
        """
        timestamp = timestamp or datetime.now()

        prediction_data = {
            'timestamp': timestamp,
            'predicted': predicted_value,
            'actual': actual_value
        }

        if metadata:
            prediction_data.update(metadata)

        self.prediction_actual_values.append(prediction_data)
        self._check_auto_save()

    def record_request_start(self, request_id: str) -> None:
        """
        Record the start time of a request.

        Args:
            request_id: Identifier for the request
        """
        self.request_timestamps[request_id] = time.time()

    def record_request_end(self, request_id: str) -> None:
        """
        Record the end time of a request and calculate response time.

        Args:
            request_id: Identifier for the request
        """
        if request_id in self.request_timestamps:
            start_time = self.request_timestamps[request_id]
            response_time = time.time() - start_time
            self.response_times.append(response_time * 1000)  # Convert to milliseconds
            del self.request_timestamps[request_id]
        else:
            logger.warning(f"No start time recorded for request {request_id}")

    def record_message_sent(self, message_size: Optional[int] = None) -> None:
        """
        Record a sent message.

        Args:
            message_size: Optional size of message in bytes
        """
        self.messages_sent += 1
        self._check_auto_save()

    def record_message_received(self, message_size: Optional[int] = None) -> None:
        """
        Record a received message.

        Args:
            message_size: Optional size of message in bytes
        """
        self.messages_received += 1
        self._check_auto_save()

    def calculate_accuracy(self) -> Dict[str, float]:
        """
        Calculate prediction accuracy metrics with improved algorithm.

        Returns:
            Dictionary of accuracy metrics
        """
        if not self.prediction_actual_values:
            return {'accuracy': 0.0, 'mae': 0.0, 'rmse': 0.0}

        # Extract values for vectorized operations
        predictions = []
        actuals = []

        for record in self.prediction_actual_values:
            pred = record['predicted']
            actual = record['actual']

            # Handle different types of predictions
            if isinstance(pred, (list, tuple)) and isinstance(actual, (list, tuple)):
                if len(pred) == len(actual) and all(isinstance(p, (int, float)) and isinstance(a, (int, float))
                                                    for p, a in zip(pred, actual)):
                    # For numeric arrays, add element-wise pairs
                    for p, a in zip(pred, actual):
                        predictions.append(p)
                        actuals.append(a)

            elif isinstance(pred, (int, float)) and isinstance(actual, (int, float)):
                # For numeric predictions
                predictions.append(pred)
                actuals.append(actual)

        # If no valid numeric predictions, return default values
        if not predictions:
            return {'accuracy': 0.0, 'mae': 0.0, 'rmse': 0.0}

        # Convert to numpy arrays for efficient calculation
        predictions = np.array(predictions)
        actuals = np.array(actuals)

        # Calculate error metrics
        error = predictions - actuals
        abs_error = np.abs(error)
        squared_error = error ** 2

        # Calculate directional accuracy for continuous values
        if len(predictions) > 1:
            pred_direction = np.sign(np.diff(predictions))
            actual_direction = np.sign(np.diff(actuals))
            dir_matches = (pred_direction == actual_direction)
            dir_accuracy = np.mean(dir_matches)
        else:
            dir_accuracy = 0.0

        # Calculate binary accuracy by thresholding
        binary_accuracy = np.mean(np.abs(predictions - actuals) < 0.5)

        # Combined accuracy metric - blend of directional and binary accuracy
        accuracy = (dir_accuracy * 0.7 + binary_accuracy * 0.3) if len(predictions) > 1 else binary_accuracy

        return {
            'accuracy': float(accuracy),
            'mae': float(np.mean(abs_error)),
            'rmse': float(np.sqrt(np.mean(squared_error))),
            'directional_accuracy': float(dir_accuracy) if len(predictions) > 1 else 0.0
        }

    def calculate_latency_metrics(self) -> Dict[str, float]:
        """
        Calculate latency metrics.

        Returns:
            Dictionary of latency metrics in milliseconds
        """
        if not self.execution_times:
            return {'mean_latency': 0.0, 'median_latency': 0.0, 'p95_latency': 0.0, 'p99_latency': 0.0}

        # Convert to numpy array for efficient calculation
        times = np.array(list(self.execution_times))

        return {
            'mean_latency': float(np.mean(times)),
            'median_latency': float(np.median(times)),
            'p95_latency': float(np.percentile(times, 95)),
            'p99_latency': float(np.percentile(times, 99))
        }

    def calculate_throughput_metrics(self, time_window: float = 3600) -> Dict[str, float]:
        """
        Calculate throughput metrics.

        Args:
            time_window: Time window in seconds for throughput calculation

        Returns:
            Dictionary of throughput metrics
        """
        current_time = time.time()
        total_operations = self.success_counts + self.error_counts

        # Calculate operations per second
        time_diff = current_time - self.last_save_time
        if time_diff <= 0:
            time_diff = 1.0  # Avoid division by zero

        throughput = total_operations / time_diff

        return {
            'throughput_ops_per_sec': float(throughput),
            'total_operations': total_operations,
            'messages_sent_per_sec': float(self.messages_sent / time_diff),
            'messages_received_per_sec': float(self.messages_received / time_diff)
        }

    def calculate_reliability_metrics(self) -> Dict[str, float]:
        """
        Calculate reliability metrics with better efficiency.

        Returns:
            Dictionary of reliability metrics
        """
        total_operations = self.success_counts + self.error_counts

        if total_operations == 0:
            return {
                'error_rate': 0.0,
                'success_rate': 0.0,
                'total_errors': 0,
                'total_successes': 0
            }

        error_rate = (self.error_counts / total_operations) * 100  # Percentage
        success_rate = (self.success_counts / total_operations) * 100  # Percentage

        return {
            'error_rate': float(error_rate),
            'success_rate': float(success_rate),
            'total_errors': self.error_counts,
            'total_successes': self.success_counts
        }

    def calculate_response_time_metrics(self) -> Dict[str, float]:
        """
        Calculate response time metrics with memory optimization.

        Returns:
            Dictionary of response time metrics in milliseconds
        """
        if not self.response_times:
            return {'mean_response_time': 0.0, 'median_response_time': 0.0,
                    'p95_response_time': 0.0, 'p99_response_time': 0.0}

        # Convert to numpy array for efficient calculation
        times = np.array(list(self.response_times))

        return {
            'mean_response_time': float(np.mean(times)),
            'median_response_time': float(np.median(times)),
            'p95_response_time': float(np.percentile(times, 95)),
            'p99_response_time': float(np.percentile(times, 99))
        }

    def get_all_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Get all metrics for the agent.

        Returns:
            Dictionary containing all calculated metrics
        """
        return {
            'accuracy': self.calculate_accuracy(),
            'latency': self.calculate_latency_metrics(),
            'throughput': self.calculate_throughput_metrics(),
            'reliability': self.calculate_reliability_metrics(),
            'response_time': self.calculate_response_time_metrics()
        }

    def save_metrics(self, file_path: Optional[str] = None) -> str:
        """
        Save current metrics to a file with optimized data storage.

        Args:
            file_path: Optional file path; if None, a default path will be used

        Returns:
            Path to the saved file
        """
        if file_path is None:
            current_date = datetime.now().strftime('%Y%m%d')
            file_path = os.path.join(self.metrics_dir, f"{self.agent_id}_{current_date}.json")

        # Get all metrics
        metrics = self.get_all_metrics()

        # Create output dictionary
        output = {
            'agent_id': self.agent_id,
            'agent_type': self.agent_type,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics
        }

        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'w') as f:
            json.dump(output, f, indent=2)

        self.last_save_time = time.time()
        logger.info(f"Metrics for agent {self.agent_id} saved to {file_path}")

        return file_path

    def reset_metrics(self) -> None:
        """
        Reset all metrics for a new tracking period.
        """
        # Save current metrics first
        self.save_metrics()

        # Reset all metrics
        self.execution_times.clear()
        self.error_counts = 0
        self.success_counts = 0
        self.prediction_actual_values.clear()
        self.task_metrics = {}
        self.response_times.clear()
        self.messages_sent = 0
        self.messages_received = 0
        self.request_timestamps = {}

        logger.info(f"Metrics for agent {self.agent_id} have been reset")

    def _check_auto_save(self) -> None:
        """
        Check if it's time to auto-save metrics based on the interval.
        """
        current_time = time.time()
        if current_time - self.last_save_time > self.auto_save_interval:
            self.save_metrics()


class AgentMetricsTracker:
    """
    Central tracker for metrics from multiple agents.
    Optimized for better memory usage and reporting.
    """

    def __init__(self, metrics_dir: str = None):
        """
        Initialize the multi-agent metrics tracker.

        Args:
            metrics_dir: Directory to store metrics data
        """
        self.metrics_dir = metrics_dir or os.path.join('data', 'metrics')
        os.makedirs(self.metrics_dir, exist_ok=True)

        # Initialize data cache
        self.data_cache = DataCache(self.metrics_dir)

        # Dictionary to hold agent metrics instances
        self.agent_metrics: Dict[str, AgentMetrics] = {}

        # Track when the last system report was generated
        self.last_report_time = time.time()

    def register_agent(self, agent_id: str, agent_type: str, max_history: int = 10000) -> AgentMetrics:
        """
        Register a new agent for metrics tracking.

        Args:
            agent_id: Unique identifier for the agent
            agent_type: Type of agent
            max_history: Maximum history items to keep in memory

        Returns:
            AgentMetrics instance for the agent
        """
        if agent_id in self.agent_metrics:
            logger.warning(f"Agent {agent_id} already registered, returning existing metrics tracker")
            return self.agent_metrics[agent_id]

        metrics = AgentMetrics(agent_id, agent_type, self.metrics_dir, max_history)
        self.agent_metrics[agent_id] = metrics
        return metrics

    def get_agent_metrics(self, agent_id: str) -> Optional[AgentMetrics]:
        """
        Get metrics tracker for a specific agent.

        Args:
            agent_id: Unique identifier for the agent

        Returns:
            AgentMetrics instance or None if not found
        """
        return self.agent_metrics.get(agent_id)

    def get_all_agents_metrics(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Get current metrics for all agents with better performance.

        Returns:
            Dictionary of all agent metrics
        """
        all_metrics = {}

        # Batch calculation for better performance
        for agent_id, metrics in self.agent_metrics.items():
            try:
                all_metrics[agent_id] = metrics.get_all_metrics()
            except Exception as e:
                logger.error(f"Error getting metrics for agent {agent_id}: {e}")
                all_metrics[agent_id] = {'error': str(e)}

        return all_metrics

    def save_all_metrics(self) -> Dict[str, str]:
        """
        Save metrics for all agents efficiently.

        Returns:
            Dictionary mapping agent IDs to saved file paths
        """
        saved_paths = {}

        for agent_id, metrics in self.agent_metrics.items():
            try:
                saved_paths[agent_id] = metrics.save_metrics()
            except Exception as e:
                logger.error(f"Error saving metrics for agent {agent_id}: {e}")

        return saved_paths

    def reset_all_metrics(self) -> None:
        """
        Reset metrics for all agents with a clear confirmation.
        """
        # Save current metrics before resetting
        self.save_all_metrics()

        # Reset all agents' metrics
        for metrics in self.agent_metrics.values():
            metrics.reset_metrics()

        logger.info(f"Reset metrics for all {len(self.agent_metrics)} agents")

    def generate_system_report(self, output_format: str = 'json',
                               include_raw_data: bool = False) -> Union[str, Dict]:
        """
        Generate a comprehensive system-wide performance report with optimizations.

        Args:
            output_format: Format for the report ('json' or 'html')
            include_raw_data: Whether to include raw metrics data

        Returns:
            Report in the specified format
        """
        report = {
            'report_id': str(uuid.uuid4()),
            'timestamp': datetime.now().isoformat(),
            'agent_count': len(self.agent_metrics),
            'agents': {},
            'system_summary': {
                'avg_error_rate': 0.0,
                'avg_response_time': 0.0,
                'total_messages': 0,
                'total_operations': 0
            }
        }

        # Initialize aggregates
        total_error_rate = 0.0
        total_response_time = 0.0
        total_messages = 0
        total_operations = 0
        agent_count = 0

        # Collect agent data efficiently
        for agent_id, metrics in self.agent_metrics.items():
            try:
                # Get agent metrics
                agent_data = {
                    'agent_type': metrics.agent_type,
                }

                # Get metrics (summary or full)
                if include_raw_data:
                    agent_data['metrics'] = metrics.get_all_metrics()
                else:
                    # Just include summaries for each category
                    agent_data['metrics'] = {
                        'accuracy': {'accuracy': metrics.calculate_accuracy().get('accuracy', 0)},
                        'reliability': {'success_rate': metrics.calculate_reliability_metrics().get('success_rate', 0)},
                        'latency': {'mean_latency': metrics.calculate_latency_metrics().get('mean_latency', 0)},
                        'response_time': {
                            'mean_response_time': metrics.calculate_response_time_metrics().get('mean_response_time',
                                                                                                0)}
                    }

                report['agents'][agent_id] = agent_data

                # Update system-wide metrics
                reliability = metrics.calculate_reliability_metrics()
                response_time = metrics.calculate_response_time_metrics()

                total_error_rate += reliability.get('error_rate', 0)
                total_response_time += response_time.get('mean_response_time', 0)
                total_messages += (metrics.messages_sent + metrics.messages_received)
                total_operations += reliability.get('total_errors', 0) + reliability.get('total_successes', 0)
                agent_count += 1

            except Exception as e:
                logger.error(f"Error processing agent {agent_id} for system report: {e}")
                report['agents'][agent_id] = {
                    'agent_type': metrics.agent_type,
                    'error': str(e)
                }

        # Calculate system summary
        if agent_count > 0:
            report['system_summary']['avg_error_rate'] = total_error_rate / agent_count
            report['system_summary']['avg_response_time'] = total_response_time / agent_count

        report['system_summary']['total_messages'] = total_messages
        report['system_summary']['total_operations'] = total_operations

        # Add system health score (0-100)
        if agent_count > 0:
            # Calculate health score based on error rate and response time
            error_score = max(0.0, 100 - report['system_summary']['avg_error_rate'])

            # Convert response time to score (lower is better)
            # Assuming >2000ms is bad (0 points), <100ms is great (100 points)
            response_time = report['system_summary']['avg_response_time']
            response_score = max(0.0, 100 - (response_time / 20))

            # Overall score is weighted average
            health_score = (error_score * 0.7) + (response_score * 0.3)
            report['system_summary']['health_score'] = min(100, health_score)
        else:
            report['system_summary']['health_score'] = 0

        # Save the report
        current_date = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = os.path.join(self.metrics_dir, f"system_report_{current_date}.json")

        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        self.last_report_time = time.time()
        logger.info(f"System report generated and saved to {report_file}")

        if output_format == 'json':
            return report
        elif output_format == 'html':
            return self._format_report_as_html(report)
        else:
            return report

    def _format_report_as_html(self, report: Dict) -> str:
        """
        Format a report as HTML with improved styling and visualizations.

        Args:
            report: Report dictionary

        Returns:
            HTML string
        """
        # Create a more modern and readable HTML report
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Alpha-Agent System Report</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        :root {{
            --primary: #2563eb;
            --secondary: #64748b;
            --success: #22c55e;
            --warning: #eab308;
            --danger: #ef4444;
            --gray-50: #f9fafb;
            --gray-100: #f3f4f6;
            --gray-200: #e5e7eb;
            --gray-800: #1f2937;
        }}

        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, Cantarell, "Open Sans", "Helvetica Neue", sans-serif;
            line-height: 1.5;
            color: var(--gray-800);
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: var(--gray-50);
        }}

        h1, h2, h3, h4 {{ 
            margin-top: 1.5em;
            margin-bottom: 0.5em;
            font-weight: 600;
        }}

        .card {{
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 24px;
            padding: 24px;
        }}

        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}

        .summary-item {{
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
            padding: 16px;
            text-align: center;
        }}

        .summary-item .value {{
            font-size: 32px;
            font-weight: 600;
            margin: 8px 0;
        }}

        .summary-item .label {{
            color: var(--secondary);
            font-size: 14px;
        }}

        .agent-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
            gap: 24px;
        }}

        .agent-card {{
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            overflow: hidden;
        }}

        .agent-header {{
            padding: 16px;
            background-color: var(--primary);
            color: white;
        }}

        .agent-content {{
            padding: 16px;
        }}

        .metric-group {{
            margin-bottom: 16px;
        }}

        .metric-group h4 {{
            margin-top: 0;
            margin-bottom: 8px;
            color: var(--secondary);
            border-bottom: 1px solid var(--gray-200);
            padding-bottom: 4px;
        }}

        .metric-row {{
            display: flex;
            justify-content: space-between;
            padding: 4px 0;
        }}

        .metric-label {{
            font-weight: 500;
        }}

        .metric-value {{
            font-family: monospace;
            font-weight: 600;
        }}

        .health-indicator {{
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }}

        .health-good {{ background-color: var(--success); }}
        .health-warning {{ background-color: var(--warning); }}
        .health-poor {{ background-color: var(--danger); }}

        footer {{
            margin-top: 48px;
            text-align: center;
            color: var(--secondary);
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <h1>Alpha-Agent System Performance Report</h1>
    <p>Generated on: {report['timestamp']}</p>

    <div class="card">
        <h2>System Summary</h2>
        <div class="summary">
            <div class="summary-item">
                <div class="label">Health Score</div>
                <div class="value">
                    <span class="health-indicator {self._get_health_class(report['system_summary'].get('health_score', 0))}"></span>
                    {report['system_summary'].get('health_score', 0):.1f}
                </div>
            </div>
            <div class="summary-item">
                <div class="label">Total Agents</div>
                <div class="value">{report['agent_count']}</div>
            </div>
            <div class="summary-item">
                <div class="label">Error Rate</div>
                <div class="value">{report['system_summary'].get('avg_error_rate', 0):.2f}%</div>
            </div>
            <div class="summary-item">
                <div class="label">Avg Response Time</div>
                <div class="value">{report['system_summary'].get('avg_response_time', 0):.2f} ms</div>
            </div>
            <div class="summary-item">
                <div class="label">Total Messages</div>
                <div class="value">{report['system_summary'].get('total_messages', 0):,}</div>
            </div>
            <div class="summary-item">
                <div class="label">Total Operations</div>
                <div class="value">{report['system_summary'].get('total_operations', 0):,}</div>
            </div>
        </div>
    </div>

    <h2>Agent Performance</h2>
    <div class="agent-grid">
"""

        # Add agent sections
        for agent_id, agent_data in report['agents'].items():
            # Skip if agent has error
            if 'error' in agent_data:
                html += f"""
        <div class="agent-card">
            <div class="agent-header">
                <h3>{agent_id}</h3>
                <div>Type: {agent_data.get('agent_type', 'Unknown')}</div>
            </div>
            <div class="agent-content">
                <div class="metric-group">
                    <h4>Error</h4>
                    <div class="metric-row">
                        <span class="metric-label">Error Message</span>
                        <span class="metric-value">{agent_data['error']}</span>
                    </div>
                </div>
            </div>
        </div>"""
                continue

            # Get metrics
            metrics = agent_data.get('metrics', {})

            # Calculate health score for agent
            health_score = 0
            reliability = metrics.get('reliability', {})
            accuracy = metrics.get('accuracy', {})

            if 'success_rate' in reliability:
                health_score += reliability['success_rate'] * 0.6

            if 'accuracy' in accuracy:
                health_score += accuracy['accuracy'] * 100 * 0.4

            html += f"""
        <div class="agent-card">
            <div class="agent-header">
                <h3>{agent_id}</h3>
                <div>Type: {agent_data.get('agent_type', 'Unknown')}</div>
            </div>
            <div class="agent-content">
                <div class="metric-group">
                    <h4>Health</h4>
                    <div class="metric-row">
                        <span class="metric-label">Score</span>
                        <span class="metric-value">
                            <span class="health-indicator {self._get_health_class(health_score)}"></span>
                            {health_score:.1f}
                        </span>
                    </div>
                </div>
"""

            # Add reliability metrics
            if 'reliability' in metrics:
                html += """
                <div class="metric-group">
                    <h4>Reliability</h4>
"""
                for name, value in metrics['reliability'].items():
                    html += f"""
                    <div class="metric-row">
                        <span class="metric-label">{name.replace('_', ' ').title()}</span>
                        <span class="metric-value">{value:.2f}{'%' if 'rate' in name.lower() else ''}</span>
                    </div>"""
                html += """
                </div>
"""

            # Add other metrics based on availability
            for category in ['accuracy', 'latency', 'response_time']:
                if category in metrics and metrics[category]:
                    html += f"""
                <div class="metric-group">
                    <h4>{category.replace('_', ' ').title()}</h4>
"""
                    for name, value in metrics[category].items():
                        if isinstance(value, (int, float)):
                            html += f"""
                    <div class="metric-row">
                        <span class="metric-label">{name.replace('_', ' ').title()}</span>
                        <span class="metric-value">{value:.2f}</span>
                    </div>"""
                    html += """
                </div>
"""

            html += """
            </div>
        </div>"""

        # Close HTML
        html += """
    </div>

    <footer>
        <p>Generated by Alpha-Agent Metrics System</p>
    </footer>
</body>
</html>"""

        return html

    def _get_health_class(self, score: float) -> str:
        """Get health indicator CSS class based on score."""
        if score >= 80:
            return "health-good"
        elif score >= 50:
            return "health-warning"
        else:
            return "health-poor"

    def get_agent_comparison(self, metric_type: str, metric_name: str) -> Dict[str, Any]:
        """
        Compare all agents on a specific metric.

        Args:
            metric_type: Type of metric to compare (e.g., 'accuracy', 'latency')
            metric_name: Specific metric name (e.g., 'accuracy', 'mean_latency')

        Returns:
            Comparison data with rankings
        """
        # Get values for all agents
        comparison = {
            'metric_type': metric_type,
            'metric_name': metric_name,
            'values': {},
            'rankings': [],
            'best_agent': None,
            'worst_agent': None
        }

        # Determine if higher is better
        higher_is_better = not any(keyword in metric_name.lower()
                                   for keyword in ['error', 'latency', 'time', 'failure'])

        for agent_id, metrics in self.agent_metrics.items():
            # Get the metrics for the specified type
            metric_dict = getattr(metrics, f"calculate_{metric_type}_metrics")()

            if metric_name in metric_dict:
                comparison['values'][agent_id] = metric_dict[metric_name]

        # If we have values, calculate rankings
        if comparison['values']:
            # Sort agents by value
            sorted_agents = sorted(
                comparison['values'].items(),
                key=lambda x: x[1],
                reverse=higher_is_better
            )

            # Set rankings
            for rank, (agent_id, value) in enumerate(sorted_agents, 1):
                comparison['rankings'].append({
                    'rank': rank,
                    'agent_id': agent_id,
                    'value': value
                })

            # Set best and worst agents
            comparison['best_agent'] = sorted_agents[0][0]
            comparison['worst_agent'] = sorted_agents[-1][0]

        return comparison

    def sample_agent_metrics(self, sample_size: int = 1000) -> None:
        """
        Sample agent metrics to reduce memory usage.

        Args:
            sample_size: Target number of data points to keep per metric
        """
        for agent_id, metrics in self.agent_metrics.items():
            # Sample execution times
            if len(metrics.execution_times) > sample_size:
                sampled = sample_time_series(list(metrics.execution_times), sample_size)
                metrics.execution_times = deque(sampled, maxlen=metrics.max_history)

            # Sample response times
            if len(metrics.response_times) > sample_size:
                sampled = sample_time_series(list(metrics.response_times), sample_size)
                metrics.response_times = deque(sampled, maxlen=metrics.max_history)

            # Sample prediction values
            if len(metrics.prediction_actual_values) > sample_size:
                sampled = sample_time_series(list(metrics.prediction_actual_values), sample_size)
                metrics.prediction_actual_values = deque(sampled, maxlen=metrics.max_history)

            # Log sampling
            logger.info(f"Sampled metrics for agent {agent_id} to reduce memory usage")

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Agent Metrics Module

This module provides performance metrics and tracking functionality for individual agents
in the Alpha-Agent system. It includes accuracy, latency, throughput, and reliability metrics.
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime, timedelta
import logging
import json
import os
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AgentMetrics:
    """
    Tracks and calculates performance metrics for individual agents.
    """
    
    def __init__(self, agent_id: str, agent_type: str, metrics_dir: str = None):
        """
        Initialize agent metrics tracker.
        
        Args:
            agent_id: Unique identifier for the agent
            agent_type: Type of agent (Market Analyst, Predictive Model, etc.)
            metrics_dir: Directory to store metrics data
        """
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.metrics_dir = metrics_dir or os.path.join('data', 'metrics')
        
        # Ensure metrics directory exists
        os.makedirs(self.metrics_dir, exist_ok=True)
        
        # Initialize metrics storage
        self.execution_times = []
        self.error_counts = 0
        self.success_counts = 0
        self.prediction_actual_values = []
        self.task_metrics = {}
        self.messages_sent = 0
        self.messages_received = 0
        
        # Track request/response metrics
        self.request_timestamps = {}
        self.response_times = []
        
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
        
        # Also store by task ID
        if task_id not in self.task_metrics:
            self.task_metrics[task_id] = {
                'execution_times': [],
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
                'execution_times': [],
                'errors': 0,
                'successes': 0
            }
        
        self.task_metrics[task_id]['errors'] += 1
        
        # Log detailed error information
        error_info = {
            'timestamp': datetime.now().isoformat(),
            'task_id': task_id,
            'error_type': error_type
        }
        
        if details:
            error_info.update(details)
        
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
                'execution_times': [],
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
        Calculate prediction accuracy metrics.
        
        Returns:
            Dictionary of accuracy metrics
        """
        if not self.prediction_actual_values:
            return {'accuracy': 0.0, 'mae': 0.0, 'rmse': 0.0}
        
        errors = []
        squared_errors = []
        correct_predictions = 0
        total_predictions = len(self.prediction_actual_values)
        
        for record in self.prediction_actual_values:
            predicted = record['predicted']
            actual = record['actual']
            
            # Handle different types of predictions
            if isinstance(predicted, (list, tuple)) and isinstance(actual, (list, tuple)):
                # For multi-class predictions, check exact match
                if predicted == actual:
                    correct_predictions += 1
                
                if len(predicted) == len(actual) and all(isinstance(p, (int, float)) and isinstance(a, (int, float)) 
                                                        for p, a in zip(predicted, actual)):
                    # If they're numeric arrays, calculate element-wise error
                    for p, a in zip(predicted, actual):
                        error = abs(p - a)
                        errors.append(error)
                        squared_errors.append(error ** 2)
            
            elif isinstance(predicted, (int, float)) and isinstance(actual, (int, float)):
                # For numeric predictions
                error = abs(predicted - actual)
                errors.append(error)
                squared_errors.append(error ** 2)
                
                # For binary/classification tasks with threshold
                if abs(predicted - actual) < 0.5:  # Assuming 0.5 as threshold
                    correct_predictions += 1
            
            else:
                # For categorical/string predictions
                if predicted == actual:
                    correct_predictions += 1
        
        # Calculate metrics
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        mae = np.mean(errors) if errors else 0.0
        rmse = np.sqrt(np.mean(squared_errors)) if squared_errors else 0.0
        
        return {
            'accuracy': accuracy,
            'mae': mae,
            'rmse': rmse
        }
    
    def calculate_latency_metrics(self) -> Dict[str, float]:
        """
        Calculate latency metrics.
        
        Returns:
            Dictionary of latency metrics in milliseconds
        """
        if not self.execution_times:
            return {'mean_latency': 0.0, 'median_latency': 0.0, 'p95_latency': 0.0, 'p99_latency': 0.0}
        
        return {
            'mean_latency': np.mean(self.execution_times),
            'median_latency': np.median(self.execution_times),
            'p95_latency': np.percentile(self.execution_times, 95),
            'p99_latency': np.percentile(self.execution_times, 99)
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
        throughput = total_operations / max(time_diff, 1.0)  # Avoid division by zero
        
        return {
            'throughput_ops_per_sec': throughput,
            'total_operations': total_operations,
            'messages_sent_per_sec': self.messages_sent / max(time_diff, 1.0),
            'messages_received_per_sec': self.messages_received / max(time_diff, 1.0)
        }
    
    def calculate_reliability_metrics(self) -> Dict[str, float]:
        """
        Calculate reliability metrics.
        
        Returns:
            Dictionary of reliability metrics
        """
        total_operations = self.success_counts + self.error_counts
        error_rate = self.error_counts / max(total_operations, 1) * 100  # Percentage
        success_rate = self.success_counts / max(total_operations, 1) * 100  # Percentage
        
        return {
            'error_rate': error_rate,
            'success_rate': success_rate,
            'total_errors': self.error_counts,
            'total_successes': self.success_counts
        }
    
    def calculate_response_time_metrics(self) -> Dict[str, float]:
        """
        Calculate response time metrics.
        
        Returns:
            Dictionary of response time metrics in milliseconds
        """
        if not self.response_times:
            return {'mean_response_time': 0.0, 'median_response_time': 0.0, 
                    'p95_response_time': 0.0, 'p99_response_time': 0.0}
        
        return {
            'mean_response_time': np.mean(self.response_times),
            'median_response_time': np.median(self.response_times),
            'p95_response_time': np.percentile(self.response_times, 95),
            'p99_response_time': np.percentile(self.response_times, 99)
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
        Save current metrics to a file.
        
        Args:
            file_path: Optional file path; if None, a default path will be used
            
        Returns:
            Path to the saved file
        """
        if file_path is None:
            current_date = datetime.now().strftime('%Y%m%d')
            file_path = os.path.join(self.metrics_dir, f"{self.agent_id}_{current_date}.json")
        
        metrics = {
            'agent_id': self.agent_id,
            'agent_type': self.agent_type,
            'timestamp': datetime.now().isoformat(),
            'metrics': self.get_all_metrics()
        }
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
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
        self.execution_times = []
        self.error_counts = 0
        self.success_counts = 0
        self.prediction_actual_values = []
        self.task_metrics = {}
        self.response_times = []
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
    """
    
    def __init__(self, metrics_dir: str = None):
        """
        Initialize the multi-agent metrics tracker.
        
        Args:
            metrics_dir: Directory to store metrics data
        """
        self.metrics_dir = metrics_dir or os.path.join('data', 'metrics')
        os.makedirs(self.metrics_dir, exist_ok=True)
        
        self.agent_metrics: Dict[str, AgentMetrics] = {}
    
    def register_agent(self, agent_id: str, agent_type: str) -> AgentMetrics:
        """
        Register a new agent for metrics tracking.
        
        Args:
            agent_id: Unique identifier for the agent
            agent_type: Type of agent
            
        Returns:
            AgentMetrics instance for the agent
        """
        if agent_id in self.agent_metrics:
            logger.warning(f"Agent {agent_id} already registered, returning existing metrics tracker")
            return self.agent_metrics[agent_id]
        
        metrics = AgentMetrics(agent_id, agent_type, self.metrics_dir)
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
        Get current metrics for all agents.
        
        Returns:
            Dictionary of all agent metrics
        """
        all_metrics = {}
        for agent_id, metrics in self.agent_metrics.items():
            all_metrics[agent_id] = metrics.get_all_metrics()
        
        return all_metrics
    
    def save_all_metrics(self) -> Dict[str, str]:
        """
        Save metrics for all agents.
        
        Returns:
            Dictionary mapping agent IDs to saved file paths
        """
        saved_paths = {}
        for agent_id, metrics in self.agent_metrics.items():
            saved_paths[agent_id] = metrics.save_metrics()
        
        return saved_paths
    
    def reset_all_metrics(self) -> None:
        """
        Reset metrics for all agents.
        """
        for metrics in self.agent_metrics.values():
            metrics.reset_metrics()
    
    def generate_system_report(self, output_format: str = 'json') -> Union[str, Dict]:
        """
        Generate a comprehensive system-wide performance report.
        
        Args:
            output_format: Format for the report ('json' or 'html')
            
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
        
        total_error_rate = 0.0
        total_response_time = 0.0
        total_messages = 0
        total_operations = 0
        agent_count = 0
        
        for agent_id, metrics in self.agent_metrics.items():
            agent_data = {
                'agent_type': metrics.agent_type,
                'metrics': metrics.get_all_metrics()
            }
            
            report['agents'][agent_id] = agent_data
            
            # Calculate system-wide metrics
            reliability = agent_data['metrics']['reliability']
            response_time = agent_data['metrics']['response_time']
            
            total_error_rate += reliability.get('error_rate', 0)
            total_response_time += response_time.get('mean_response_time', 0)
            total_messages += (metrics.messages_sent + metrics.messages_received)
            total_operations += reliability.get('total_errors', 0) + reliability.get('total_successes', 0)
            agent_count += 1
        
        if agent_count > 0:
            report['system_summary']['avg_error_rate'] = total_error_rate / agent_count
            report['system_summary']['avg_response_time'] = total_response_time / agent_count
        
        report['system_summary']['total_messages'] = total_messages
        report['system_summary']['total_operations'] = total_operations
        
        # Save the report
        current_date = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = os.path.join(self.metrics_dir, f"system_report_{current_date}.json")
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"System report generated and saved to {report_file}")
        
        if output_format == 'json':
            return report
        elif output_format == 'html':
            return self._format_report_as_html(report)
        else:
            return report
    
    def _format_report_as_html(self, report: Dict) -> str:
        """
        Format a report as HTML.
        
        Args:
            report: Report dictionary
            
        Returns:
            HTML string
        """
        # Simple HTML formatting for the report
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Alpha-Agent System Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1, h2, h3 {{ color: #333; }}
        .summary {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
        .agent {{ background-color: #fff; border: 1px solid #ddd; padding: 15px; margin-bottom: 15px; border-radius: 5px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <h1>Alpha-Agent System Performance Report</h1>
    <p>Generated on: {report['timestamp']}</p>
    
    <div class="summary">
        <h2>System Summary</h2>
        <p>Total Agents: {report['agent_count']}</p>
        <p>Average Error Rate: {report['system_summary']['avg_error_rate']:.2f}%</p>
        <p>Average Response Time: {report['system_summary']['avg_response_time']:.2f} ms</p>
        <p>Total Messages: {report['system_summary']['total_messages']}</p>
        <p>Total Operations: {report['system_summary']['total_operations']}</p>
    </div>
    
    <h2>Agent Performance</h2>
"""
        
        # Add agent sections
        for agent_id, agent_data in report['agents'].items():
            metrics = agent_data['metrics']
            html += f"""    
    <div class="agent">
        <h3>Agent: {agent_id} (Type: {agent_data['agent_type']})</h3>
        
        <h4>Reliability</h4>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Error Rate</td>
                <td>{metrics['reliability']['error_rate']:.2f}%</td>
            </tr>
            <tr>
                <td>Success Rate</td>
                <td>{metrics['reliability']['success_rate']:.2f}%</td>
            </tr>
            <tr>
                <td>Total Errors</td>
                <td>{metrics['reliability']['total_errors']}</td>
            </tr>
            <tr>
                <td>Total Successes</td>
                <td>{metrics['reliability']['total_successes']}</td>
            </tr>
        </table>
        
        <h4>Performance</h4>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Mean Latency</td>
                <td>{metrics['latency']['mean_latency']:.2f} ms</td>
            </tr>
            <tr>
                <td>95th Percentile Latency</td>
                <td>{metrics['latency']['p95_latency']:.2f} ms</td>
            </tr>
            <tr>
                <td>Mean Response Time</td>
                <td>{metrics['response_time']['mean_response_time']:.2f} ms</td>
            </tr>
            <tr>
                <td>Throughput (ops/sec)</td>
                <td>{metrics['throughput']['throughput_ops_per_sec']:.2f}</td>
            </tr>
        </table>
"""
            
            # Add accuracy metrics if available
            if 'accuracy' in metrics and metrics['accuracy'].get('accuracy', 0) > 0:
                html += f"""        
        <h4>Accuracy</h4>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Accuracy</td>
                <td>{metrics['accuracy']['accuracy']:.4f}</td>
            </tr>
            <tr>
                <td>MAE</td>
                <td>{metrics['accuracy']['mae']:.4f}</td>
            </tr>
            <tr>
                <td>RMSE</td>
                <td>{metrics['accuracy']['rmse']:.4f}</td>
            </tr>
        </table>
"""
            
            html += "    </div>\n"
        
        # Close HTML
        html += "</body>\n</html>"
        
        return html
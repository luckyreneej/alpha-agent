#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Contribution Analyzer Module

This module analyzes the contribution of individual agents to the overall system performance,
tracking their impact and effectiveness over time.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ContributionAnalyzer:
    """
    Analyzes the contribution of individual agents to overall system performance.
    """
    
    def __init__(self, data_dir: str = None):
        """
        Initialize the contribution analyzer.
        
        Args:
            data_dir: Directory containing metrics data files
        """
        self.data_dir = data_dir or os.path.join('data', 'metrics')
        os.makedirs(self.data_dir, exist_ok=True)
    
    def load_agent_data(self, agent_id: str = None, start_date: Optional[datetime] = None,
                      end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Load agent performance data.
        
        Args:
            agent_id: Optional agent ID to filter by
            start_date: Optional start date
            end_date: Optional end date
            
        Returns:
            DataFrame containing agent performance data
        """
        # Define file pattern based on agent_id
        if agent_id:
            file_pattern = os.path.join(self.data_dir, f"{agent_id}_*.json")
        else:
            file_pattern = os.path.join(self.data_dir, "*.json")
        
        import glob
        files = glob.glob(file_pattern)
        
        # Filter out system report files
        files = [f for f in files if "system_report" not in os.path.basename(f)]
        
        if not files:
            logger.warning(f"No data files found for {'agent ' + agent_id if agent_id else 'any agent'}")
            return pd.DataFrame()
        
        # Parse files and collect data
        records = []
        
        for file_path in files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Extract basic info
                if 'agent_id' not in data or 'timestamp' not in data:
                    continue
                    
                agent_id = data['agent_id']
                timestamp = datetime.fromisoformat(data['timestamp'])
                
                # Filter by date if specified
                if start_date and timestamp < start_date:
                    continue
                if end_date and timestamp > end_date:
                    continue
                
                # Extract agent type
                agent_type = data.get('agent_type', 'unknown')
                
                # Extract performance metrics
                metrics = data.get('metrics', {})
                
                # Create record for this data point
                record = {
                    'agent_id': agent_id,
                    'agent_type': agent_type,
                    'timestamp': timestamp
                }
                
                # Add flattened metrics
                for category, category_metrics in metrics.items():
                    for metric_name, value in category_metrics.items():
                        record[f"{category}_{metric_name}"] = value
                
                records.append(record)
                
            except Exception as e:
                logger.error(f"Error parsing file {file_path}: {e}")
                continue
        
        if not records:
            return pd.DataFrame()
            
        df = pd.DataFrame(records)
        df = df.sort_values('timestamp')
        
        return df
    
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
                'reliability_success_rate': 0.25,
                'reliability_error_rate': -0.20,  # Negative weight for errors
                'latency_mean_latency': -0.15,    # Negative weight for latency 
                'throughput_throughput_ops_per_sec': 0.20,
                'accuracy_accuracy': 0.20
            }
            
            # Add response time metric if available
            has_response_time = any('response_time_mean_response_time' in col for col in data.columns)
            if has_response_time:
                metric_weights['response_time_mean_response_time'] = -0.15  # Negative weight
                # Adjust other weights
                for key in list(metric_weights.keys()):
                    if key != 'response_time_mean_response_time':
                        metric_weights[key] *= 0.85  # Rescale to make weights sum close to 1.0
        
        # Ensure metrics exist in the data
        valid_metrics = [m for m in metric_weights.keys() if m in data.columns]
        if not valid_metrics:
            logger.warning("None of the specified metrics found in data")
            return {}
        
        # Normalize weights to sum to 1.0
        total_weight = sum(abs(metric_weights[m]) for m in valid_metrics)
        if total_weight == 0:
            logger.warning("Total weight is zero, unable to calculate contributions")
            return {}
        
        normalized_weights = {m: metric_weights[m] / total_weight for m in valid_metrics}
        
        # Group by agent and calculate metrics
        agent_metrics = {}
        
        for agent_id, group in data.groupby('agent_id'):
            metrics_data = {}
            
            for metric in valid_metrics:
                if metric in group.columns:
                    # Calculate average for this metric
                    avg_value = group[metric].mean()
                    if not pd.isna(avg_value):
                        metrics_data[metric] = avg_value
            
            if metrics_data:
                agent_metrics[agent_id] = metrics_data
        
        # Calculate normalized scores for each metric
        normalized_scores = {}
        
        for metric in valid_metrics:
            # Extract values for this metric across all agents
            values = [agent_data.get(metric) for agent_data in agent_metrics.values() 
                     if metric in agent_data and not pd.isna(agent_data[metric])]
            
            if not values:
                continue
                
            # Calculate min, max for normalization
            min_val = min(values)
            max_val = max(values)
            
            # Skip if all values are the same
            if min_val == max_val:
                continue
                
            # Normalize based on whether higher is better
            is_higher_better = not any(keyword in metric.lower() 
                                     for keyword in ['error', 'latency', 'time', 'failure'])
            
            # For each agent, calculate normalized score for this metric
            for agent_id, agent_data in agent_metrics.items():
                if metric in agent_data and not pd.isna(agent_data[metric]):
                    value = agent_data[metric]
                    
                    if is_higher_better:
                        # Higher is better, normalize to 0-1 range
                        norm_score = (value - min_val) / (max_val - min_val)
                    else:
                        # Lower is better, invert normalization
                        norm_score = 1 - ((value - min_val) / (max_val - min_val))
                    
                    if agent_id not in normalized_scores:
                        normalized_scores[agent_id] = {}
                    
                    normalized_scores[agent_id][metric] = norm_score
        
        # Calculate weighted average scores
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
        
        # Normalize contributions to sum to 100%
        total_contribution = sum(contribution_scores.values())
        if total_contribution > 0:
            contribution_scores = {agent_id: (score / total_contribution) * 100 
                                 for agent_id, score in contribution_scores.items()}
        
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
        
        # Create windows
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
    
    def calculate_adaptive_weights(self, lookback_days: int = 30, 
                                performance_metrics: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Calculate adaptive weights for agents based on their historical performance.
        
        Args:
            lookback_days: Number of days to look back
            performance_metrics: List of metrics to consider for weighting
            
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
        
        # Default performance metrics if not provided
        if performance_metrics is None:
            performance_metrics = [
                'reliability_success_rate',
                'latency_mean_latency',
                'throughput_throughput_ops_per_sec',
                'accuracy_accuracy'
            ]
        
        # Filter to metrics available in the data
        available_metrics = [m for m in performance_metrics if m in data.columns]
        if not available_metrics:
            logger.warning("None of the specified metrics found in data")
            return {}
        
        # Group by agent and calculate performance scores
        agent_scores = {}
        
        for agent_id, group in data.groupby('agent_id'):
            # Calculate score for each available metric
            metric_scores = {}
            
            for metric in available_metrics:
                if metric in group.columns:
                    values = group[metric].dropna()
                    if len(values) > 0:
                        # For metrics where lower is better, invert the value
                        is_lower_better = any(keyword in metric.lower() 
                                            for keyword in ['error', 'latency', 'time', 'failure'])
                        
                        avg_value = values.mean()
                        if is_lower_better:
                            # For metrics where lower is better, use inverse
                            if avg_value != 0:  # Avoid division by zero
                                metric_scores[metric] = 1 / avg_value
                            else:
                                metric_scores[metric] = float('inf')  # Perfect score
                        else:
                            metric_scores[metric] = avg_value
            
            # Calculate overall score as average of metric scores
            if metric_scores:
                agent_scores[agent_id] = sum(metric_scores.values()) / len(metric_scores)
        
        # If no scores calculated, return equal weights
        if not agent_scores:
            return {}
        
        # Convert scores to weights that sum to 1.0
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
        
        # Convert to DataFrame for plotting
        contrib_data = pd.DataFrame([(agent_id, score) for agent_id, score in contributions.items()],
                                 columns=['agent_id', 'contribution'])
        
        # Sort by contribution (highest first)
        contrib_data = contrib_data.sort_values('contribution', ascending=False)
        
        # Create the plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 7), gridspec_kw={'width_ratios': [2, 1]})
        
        # Bar chart
        bars = ax1.bar(
            contrib_data['agent_id'], 
            contrib_data['contribution'],
            color=plt.cm.viridis(np.linspace(0, 0.8, len(contrib_data)))
        )
        
        # Add percentage labels
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, height,
                   f"{height:.1f}%", ha='center', va='bottom')
        
        ax1.set_title("Agent Contribution Breakdown")
        ax1.set_xlabel("Agent ID")
        ax1.set_ylabel("Contribution (%)")
        plt.xticks(rotation=45, ha='right')
        
        # Pie chart
        ax2.pie(
            contrib_data['contribution'],
            labels=contrib_data['agent_id'],
            autopct='%1.1f%%',
            startangle=90,
            colors=plt.cm.viridis(np.linspace(0, 0.8, len(contrib_data)))
        )
        ax2.set_title("Contribution Share")
        
        plt.tight_layout()
        
        # Save the plot
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.data_dir, f"contribution_breakdown_{timestamp}.png")
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Contribution breakdown visualization saved to {output_file}")
        return output_file
    
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
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Get unique agents, sorted by average contribution
        agent_avg_contrib = history_df.groupby('agent_id')['contribution'].mean()
        agents = agent_avg_contrib.sort_values(ascending=False).index
        
        # Create a pivot table for plotting
        pivot_df = history_df.pivot(index='window_mid', columns='agent_id', values='contribution')
        
        # Plot stacked area chart
        pivot_df[agents].plot.area(
            ax=ax,
            stacked=True,
            alpha=0.7,
            colormap='viridis'
        )
        
        ax.set_title(f"Agent Contribution Over Time (Window Size: {window_size} days)")
        ax.set_xlabel("Time")
        ax.set_ylabel("Contribution (%)")
        ax.legend(title="Agent ID")
        ax.grid(True, alpha=0.3)
        
        # Save the plot
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.data_dir, f"contribution_time_{timestamp}.png")
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Contribution over time visualization saved to {output_file}")
        return output_file
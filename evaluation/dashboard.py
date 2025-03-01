#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluation Dashboard Module

Provides a Streamlit-based dashboard for visualizing agent performance metrics.
"""

import os
import sys
import json
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from typing import Dict
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_metrics_data(data_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Load metrics data for all agents from the data directory.
    
    Args:
        data_dir: Directory containing metrics files
    
    Returns:
        Dictionary mapping agent IDs to their metrics DataFrames
    """
    metrics_data = {}
    file_pattern = os.path.join(data_dir, "*.json")
    metric_files = glob.glob(file_pattern)

    for file_path in metric_files:
        try:
            if "system_report" in os.path.basename(file_path):
                continue

            with open(file_path, 'r') as f:
                data = json.load(f)

            # Extract agent ID and timestamp
            if 'agent_id' in data and 'timestamp' in data:
                agent_id = data['agent_id']
                timestamp = datetime.fromisoformat(data['timestamp'])

                # Initialize DataFrame for this agent if needed
                if agent_id not in metrics_data:
                    metrics_data[agent_id] = []

                # Extract metrics and flatten for DataFrame
                record = {'timestamp': timestamp}
                record['agent_type'] = data.get('agent_type', 'unknown')

                # Flatten metrics structure
                metrics = data.get('metrics', {})
                for category, category_metrics in metrics.items():
                    for metric_name, metric_value in category_metrics.items():
                        record[f"{category}_{metric_name}"] = metric_value

                metrics_data[agent_id].append(record)

        except Exception as e:
            logger.error(f"Error parsing metrics file {file_path}: {e}")

    # Convert lists to DataFrames
    for agent_id in metrics_data:
        if metrics_data[agent_id]:
            metrics_data[agent_id] = pd.DataFrame(metrics_data[agent_id])

    return metrics_data


def run_dashboard(data_dir: str):
    """
    Run the Streamlit dashboard application.
    
    Args:
        data_dir: Directory containing metrics data
    """
    # Set page configuration
    st.set_page_config(
        page_title="Alpha-Agent Performance Dashboard",
        page_icon=":bar_chart:",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Page title
    st.title("Alpha-Agent Performance Dashboard")

    # Load metrics data
    metrics_data = load_metrics_data(data_dir)

    if not metrics_data:
        st.warning("No metrics data found. Please ensure data files exist in the specified directory.")
        return

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select View",
        ["System Overview", "Agent Performance", "Agent Comparison", "Performance Trends"]
    )

    # Time range filter
    st.sidebar.title("Time Filter")
    today = datetime.now().date()
    start_date = st.sidebar.date_input("Start Date", today - timedelta(days=30))
    end_date = st.sidebar.date_input("End Date", today)

    if start_date > end_date:
        st.sidebar.error("Start date should be before end date.")
        return

    # Convert to datetime
    start_datetime = datetime.combine(start_date, datetime.min.time())
    end_datetime = datetime.combine(end_date, datetime.max.time())

    # Filter data by time range
    filtered_data = {}
    for agent_id, df in metrics_data.items():
        if not df.empty and 'timestamp' in df.columns:
            filtered_df = df[(df['timestamp'] >= start_datetime) & (df['timestamp'] <= end_datetime)]
            if not filtered_df.empty:
                filtered_data[agent_id] = filtered_df

    if not filtered_data:
        st.warning("No data found for the selected time period.")
        return

    # Render the selected page
    if page == "System Overview":
        render_system_overview(filtered_data)
    elif page == "Agent Performance":
        render_agent_performance(filtered_data)
    elif page == "Agent Comparison":
        render_agent_comparison(filtered_data)
    elif page == "Performance Trends":
        render_performance_trends(filtered_data)


def render_system_overview(data: Dict[str, pd.DataFrame]):
    """
    Render system overview page.
    
    Args:
        data: Dictionary of agent metrics data
    """
    st.header("System Overview")

    # Summary metrics
    st.subheader("System Performance Summary")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Agents", len(data))

    # Calculate average success rate across agents
    success_rates = []
    for df in data.values():
        if 'reliability_success_rate' in df.columns:
            avg_rate = df['reliability_success_rate'].mean()
            if not pd.isna(avg_rate):
                success_rates.append(avg_rate)

    with col2:
        avg_success = np.mean(success_rates) if success_rates else None
        if avg_success is not None:
            st.metric("Avg Success Rate", f"{avg_success:.2f}%")
        else:
            st.metric("Avg Success Rate", "N/A")

    # Calculate average latency across agents
    latencies = []
    for df in data.values():
        if 'latency_mean_latency' in df.columns:
            avg_latency = df['latency_mean_latency'].mean()
            if not pd.isna(avg_latency):
                latencies.append(avg_latency)

    with col3:
        avg_latency = np.mean(latencies) if latencies else None
        if avg_latency is not None:
            st.metric("Avg Latency", f"{avg_latency:.2f} ms")
        else:
            st.metric("Avg Latency", "N/A")

    # Calculate total operations
    total_ops = 0
    for df in data.values():
        if 'reliability_total_successes' in df.columns and 'reliability_total_errors' in df.columns:
            # Use the latest value as the cumulative count
            successes = df['reliability_total_successes'].max() if len(df) > 0 else 0
            errors = df['reliability_total_errors'].max() if len(df) > 0 else 0
            total_ops += successes + errors

    with col4:
        st.metric("Total Operations", f"{total_ops:,}")

    # Agent type distribution
    st.subheader("Agent Type Distribution")

    agent_types = {}
    for agent_id, df in data.items():
        if not df.empty and 'agent_type' in df.columns:
            agent_type = df['agent_type'].iloc[0]
            if agent_type not in agent_types:
                agent_types[agent_type] = 0
            agent_types[agent_type] += 1

    if agent_types:
        # Convert to DataFrame for plotting
        agent_types_df = pd.DataFrame([{'Type': t, 'Count': c} for t, c in agent_types.items()])

        # Create chart
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(agent_types_df['Type'], agent_types_df['Count'], color='skyblue')
        ax.set_xlabel('Agent Type')
        ax.set_ylabel('Count')
        ax.set_title('Agent Type Distribution')

        # Add data labels
        for i, v in enumerate(agent_types_df['Count']):
            ax.text(i, v + 0.1, str(v), ha='center')

        st.pyplot(fig)
    else:
        st.write("No agent type data available.")

    # Success rate by agent type
    st.subheader("Performance by Agent Type")

    # Group metrics by agent type
    type_metrics = {}
    for agent_id, df in data.items():
        if not df.empty and 'agent_type' in df.columns:
            agent_type = df['agent_type'].iloc[0]

            if agent_type not in type_metrics:
                type_metrics[agent_type] = {
                    'success_rates': [],
                    'latencies': [],
                    'error_rates': []
                }

            if 'reliability_success_rate' in df.columns:
                success_rate = df['reliability_success_rate'].mean()
                if not pd.isna(success_rate):
                    type_metrics[agent_type]['success_rates'].append(success_rate)

            if 'latency_mean_latency' in df.columns:
                latency = df['latency_mean_latency'].mean()
                if not pd.isna(latency):
                    type_metrics[agent_type]['latencies'].append(latency)

            if 'reliability_error_rate' in df.columns:
                error_rate = df['reliability_error_rate'].mean()
                if not pd.isna(error_rate):
                    type_metrics[agent_type]['error_rates'].append(error_rate)

    if type_metrics:
        # Calculate averages by type
        avg_by_type = {
            agent_type: {
                'avg_success_rate': np.mean(metrics['success_rates']) if metrics['success_rates'] else None,
                'avg_latency': np.mean(metrics['latencies']) if metrics['latencies'] else None,
                'avg_error_rate': np.mean(metrics['error_rates']) if metrics['error_rates'] else None
            } for agent_type, metrics in type_metrics.items()
        }

        # Convert to DataFrame for display
        type_summary = []
        for agent_type, metrics in avg_by_type.items():
            record = {'Agent Type': agent_type}
            record.update({
                'Success Rate (%)': f"{metrics['avg_success_rate']:.2f}" if metrics[
                                                                                'avg_success_rate'] is not None else "N/A",
                'Latency (ms)': f"{metrics['avg_latency']:.2f}" if metrics['avg_latency'] is not None else "N/A",
                'Error Rate (%)': f"{metrics['avg_error_rate']:.2f}" if metrics['avg_error_rate'] is not None else "N/A"
            })
            type_summary.append(record)

        # Display as table
        st.table(pd.DataFrame(type_summary))
    else:
        st.write("No performance metrics available by agent type.")


def render_agent_performance(data: Dict[str, pd.DataFrame]):
    """
    Render individual agent performance page.
    
    Args:
        data: Dictionary of agent metrics data
    """
    st.header("Agent Performance")

    # Agent selection
    agent_options = list(data.keys())
    if not agent_options:
        st.warning("No agent data available.")
        return

    agent_display_names = {}
    for agent_id in agent_options:
        df = data[agent_id]
        if not df.empty and 'agent_type' in df.columns:
            agent_type = df['agent_type'].iloc[0]
            agent_display_names[agent_id] = f"{agent_id} ({agent_type})"
        else:
            agent_display_names[agent_id] = agent_id

    selected_agent = st.selectbox(
        "Select an agent:",
        options=agent_options,
        format_func=lambda x: agent_display_names[x]
    )

    # Get agent data
    agent_df = data[selected_agent]

    # Agent overview
    st.subheader("Agent Overview")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        agent_type = agent_df['agent_type'].iloc[0] if 'agent_type' in agent_df.columns else "Unknown"
        st.metric("Agent Type", agent_type)

    with col2:
        if 'reliability_success_rate' in agent_df.columns:
            success_rate = agent_df['reliability_success_rate'].mean()
            if not pd.isna(success_rate):
                st.metric("Success Rate", f"{success_rate:.2f}%")
            else:
                st.metric("Success Rate", "N/A")
        else:
            st.metric("Success Rate", "N/A")

    with col3:
        if 'latency_mean_latency' in agent_df.columns:
            mean_latency = agent_df['latency_mean_latency'].mean()
            if not pd.isna(mean_latency):
                st.metric("Avg Latency", f"{mean_latency:.2f} ms")
            else:
                st.metric("Avg Latency", "N/A")
        else:
            st.metric("Avg Latency", "N/A")

    with col4:
        if 'reliability_total_successes' in agent_df.columns and 'reliability_total_errors' in agent_df.columns:
            successes = agent_df['reliability_total_successes'].max()
            errors = agent_df['reliability_total_errors'].max()
            st.metric("Operations", f"{int(successes + errors):,}")
        else:
            st.metric("Operations", "N/A")

    # Performance metrics over time
    st.subheader("Performance Over Time")

    # Get time-series metrics
    time_metrics = [
        ('reliability_success_rate', 'Success Rate (%)', 'green'),
        ('reliability_error_rate', 'Error Rate (%)', 'red'),
        ('latency_mean_latency', 'Mean Latency (ms)', 'blue')
    ]

    available_metrics = [m for m, _, _ in time_metrics if m in agent_df.columns]

    if available_metrics and 'timestamp' in agent_df.columns:
        # Plot time series
        fig, ax = plt.subplots(figsize=(10, 6))

        for metric, label, color in time_metrics:
            if metric in agent_df.columns:
                ax.plot(agent_df['timestamp'], agent_df[metric], label=label, color=color)

        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.set_title(f'Performance Metrics for {selected_agent}')
        ax.legend()
        fig.autofmt_xdate()  # Rotate date labels

        st.pyplot(fig)
    else:
        st.write("No time-series data available for this agent.")

    # Detailed metrics table
    st.subheader("All Performance Metrics")

    # Prepare metrics table
    metrics_dict = {}
    for category in ['reliability', 'latency', 'accuracy', 'throughput', 'response_time']:
        category_metrics = [col for col in agent_df.columns if col.startswith(f"{category}_")]
        if category_metrics:
            for metric in category_metrics:
                # Extract just the metric name part (after the category prefix)
                metric_name = metric.split('_', 1)[1] if '_' in metric else metric
                metric_name = metric_name.replace('_', ' ').title()

                value = agent_df[metric].mean()
                if pd.isna(value):
                    metrics_dict[metric_name] = "N/A"
                elif 'rate' in metric.lower() or 'ratio' in metric.lower():
                    metrics_dict[metric_name] = f"{value:.2f}%"
                else:
                    metrics_dict[metric_name] = f"{value:.4f}"

    if metrics_dict:
        # Display as two columns
        metrics_df = pd.DataFrame(list(metrics_dict.items()), columns=['Metric', 'Value'])
        col1, col2 = st.columns(2)

        mid_idx = len(metrics_df) // 2
        with col1:
            st.table(metrics_df.iloc[:mid_idx])

        with col2:
            st.table(metrics_df.iloc[mid_idx:])
    else:
        st.write("No detailed metrics available.")


def render_agent_comparison(data: Dict[str, pd.DataFrame]):
    """
    Render agent comparison page.
    
    Args:
        data: Dictionary of agent metrics data
    """
    st.header("Agent Comparison")

    # Find common metrics across agents
    all_metrics = set()
    for df in data.values():
        all_metrics.update(df.columns)

    # Filter to numeric metrics
    numeric_metrics = []
    for agent_id, df in data.items():
        for column in df.columns:
            if pd.api.types.is_numeric_dtype(df[column]) and column not in ['timestamp']:
                numeric_metrics.append(column)

    numeric_metrics = list(set(numeric_metrics))

    # Create friendly metric names
    metric_names = {}
    for metric in numeric_metrics:
        parts = metric.split('_')
        if len(parts) > 1:
            # Create a readable name
            readable = ' '.join(parts[1:]).title()
            metric_names[metric] = f"{parts[0].title()} {readable}"
        else:
            metric_names[metric] = metric.title()

    # Let user select metrics to compare
    if numeric_metrics:
        selected_metric = st.selectbox(
            "Select metric to compare:",
            options=numeric_metrics,
            format_func=lambda x: metric_names.get(x, x)
        )

        # Collect data for the selected metric
        comparison_data = []
        for agent_id, df in data.items():
            if selected_metric in df.columns:
                agent_type = df['agent_type'].iloc[0] if 'agent_type' in df.columns else "Unknown"
                value = df[selected_metric].mean()

                if not pd.isna(value):
                    comparison_data.append({
                        'Agent ID': agent_id,
                        'Agent Type': agent_type,
                        'Value': value
                    })

        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)

            # Determine if lower is better for display purposes
            lower_is_better = any(keyword in selected_metric.lower()
                                  for keyword in ['error', 'latency', 'time', 'failure'])

            # Sort by value
            comparison_df = comparison_df.sort_values('Value', ascending=lower_is_better)

            # Create bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(
                comparison_df['Agent ID'],
                comparison_df['Value'],
                color=[plt.cm.RdYlGn_r(i / len(comparison_df)) if lower_is_better
                       else plt.cm.RdYlGn(i / len(comparison_df))
                       for i in range(len(comparison_df))]
            )

            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                        f'{height:.2f}', ha='center', va='bottom')

            ax.set_xlabel('Agent')
            ax.set_ylabel(metric_names.get(selected_metric, selected_metric))
            ax.set_title(f'Agent Comparison: {metric_names.get(selected_metric, selected_metric)}')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()

            st.pyplot(fig)

            # Display as table as well
            st.subheader("Comparison Data")
            st.table(comparison_df)
        else:
            st.write("No data available for the selected metric.")
    else:
        st.warning("No numeric metrics available for comparison.")


def render_performance_trends(data: Dict[str, pd.DataFrame]):
    """
    Render performance trends page.
    
    Args:
        data: Dictionary of agent metrics data
    """
    st.header("Performance Trends")

    # Agent selection
    agent_options = list(data.keys())
    if not agent_options:
        st.warning("No agent data available.")
        return

    agent_display_names = {}
    for agent_id in agent_options:
        df = data[agent_id]
        if not df.empty and 'agent_type' in df.columns:
            agent_type = df['agent_type'].iloc[0]
            agent_display_names[agent_id] = f"{agent_id} ({agent_type})"
        else:
            agent_display_names[agent_id] = agent_id

    selected_agents = st.multiselect(
        "Select agents to compare:",
        options=agent_options,
        format_func=lambda x: agent_display_names[x],
        default=agent_options[:min(3, len(agent_options))]
    )

    if not selected_agents:
        st.warning("Please select at least one agent.")
        return

    # Metric selection
    all_metrics = set()
    for agent_id in selected_agents:
        df = data[agent_id]
        for column in df.columns:
            if pd.api.types.is_numeric_dtype(df[column]) and column != 'timestamp':
                all_metrics.add(column)

    # Create friendly metric names
    metric_names = {}
    for metric in all_metrics:
        parts = metric.split('_')
        if len(parts) > 1:
            readable = ' '.join(parts[1:]).title()
            metric_names[metric] = f"{parts[0].title()} {readable}"
        else:
            metric_names[metric] = metric.title()

    selected_metric = st.selectbox(
        "Select metric to track:",
        options=list(all_metrics),
        format_func=lambda x: metric_names.get(x, x)
    )

    # Plot trend lines for selected agents
    if selected_metric in all_metrics:
        fig, ax = plt.subplots(figsize=(10, 6))

        for agent_id in selected_agents:
            df = data[agent_id]
            if selected_metric in df.columns and 'timestamp' in df.columns:
                df = df.sort_values('timestamp')
                ax.plot(df['timestamp'], df[selected_metric], marker='o', label=agent_display_names[agent_id])

        ax.set_xlabel('Time')
        ax.set_ylabel(metric_names.get(selected_metric, selected_metric))
        ax.set_title(f'Performance Trend: {metric_names.get(selected_metric, selected_metric)}')
        ax.legend()
        fig.autofmt_xdate()  # Rotate date labels
        plt.grid(True, alpha=0.3)

        st.pyplot(fig)
    else:
        st.warning("Selected metric not available for the chosen agents.")


if __name__ == "__main__":
    # Get data directory from command line or use default
    import argparse

    parser = argparse.ArgumentParser(description='Run Alpha-Agent Performance Dashboard')
    parser.add_argument('--data-dir', type=str, default='data/metrics',
                        help='Directory containing metrics data files')

    args = parser.parse_args()

    # Run the dashboard
    run_dashboard(args.data_dir)

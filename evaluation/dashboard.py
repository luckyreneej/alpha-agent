#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Optimized Visualization Dashboard Module

This module provides a real-time visualization dashboard for monitoring
agent performance and system metrics using Plotly Dash with improved
performance and memory efficiency.
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import time

# Dashboard libraries
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import metrics collectors and analyzers
from agent_metrics import AgentMetricsTracker
from agent_evaluator import AgentEvaluator
from contribution_analyzer import ContributionAnalyzer
from performance_tracker import PerformanceTracker
from metrics_utils import sample_time_series, calculate_time_window

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VisualizationDashboard:
    """
    Real-time visualization dashboard for monitoring agent performance and system metrics.
    Optimized for better performance and responsiveness.
    """

    def __init__(self,
                 metrics_tracker: AgentMetricsTracker,
                 agent_evaluator: AgentEvaluator,
                 data_dir: str = None,
                 update_interval: int = 30,
                 max_points: int = 500):  # max points for data visualization
        """
        Initialize the dashboard.

        Args:
            metrics_tracker: Agent metrics tracker instance
            agent_evaluator: Agent evaluator instance
            data_dir: Directory for metrics data
            update_interval: Dashboard refresh interval in seconds
            max_points: Maximum number of data points to plot for performance
        """
        self.metrics_tracker = metrics_tracker
        self.agent_evaluator = agent_evaluator
        self.data_dir = data_dir or os.path.join('data', 'metrics')
        self.update_interval = update_interval
        self.max_points = max_points

        # Additional analysis components
        self.contribution_analyzer = ContributionAnalyzer(self.data_dir)
        self.performance_tracker = PerformanceTracker(self.data_dir)

        # Data caching for better performance
        self.data_cache = {}
        self.last_data_update = {}
        self.data_cache_ttl = 60  # seconds

        # Initialize Dash app with improved styling
        self.app = dash.Dash(
            __name__,
            title='Alpha-Agent Performance Dashboard',
            suppress_callback_exceptions=True,
            # Use external stylesheets for better appearance
            external_stylesheets=[
                'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap',
                'https://cdnjs.cloudflare.com/ajax/libs/modern-normalize/1.1.0/modern-normalize.min.css'
            ]
        )

        # Setup dashboard layout with modern styling
        self._setup_layout()

        # Setup callbacks
        self._setup_callbacks()

    def _setup_layout(self):
        """Configure the dashboard layout with improved UI."""
        # Add custom CSS for better styling
        custom_css = """
        :root {
            --primary: #2563eb;
            --primary-light: #3b82f6;
            --secondary: #64748b;
            --success: #22c55e;
            --warning: #eab308;
            --danger: #ef4444;
            --gray-50: #f9fafb;
            --gray-100: #f3f4f6;
            --gray-200: #e5e7eb;
            --gray-300: #d1d5db;
            --gray-800: #1f2937;
        }

        body {
            font-family: 'Inter', sans-serif;
            background-color: var(--gray-50);
            color: var(--gray-800);
            line-height: 1.5;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 16px;
        }

        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 24px;
            padding-bottom: 16px;
            border-bottom: 1px solid var(--gray-200);
        }

        .dashboard-title {
            font-size: 24px;
            font-weight: 600;
            margin: 0;
            color: var(--gray-800);
        }

        .card {
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
            padding: 16px;
            margin-bottom: 16px;
        }

        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
        }

        .card-title {
            font-size: 16px;
            font-weight: 600;
            margin: 0;
        }

        .button {
            padding: 8px 16px;
            border-radius: 6px;
            font-weight: 500;
            cursor: pointer;
            text-align: center;
            transition: background-color 0.3s;
        }

        .button-primary {
            background-color: var(--primary);
            color: white;
            border: none;
        }

        .button-primary:hover {
            background-color: var(--primary-light);
        }

        .dropdown {
            background-color: white;
            border: 1px solid var(--gray-300);
            border-radius: 6px;
            padding: 8px 12px;
        }

        .tabs {
            display: flex;
            border-bottom: 1px solid var(--gray-200);
            margin-bottom: 16px;
        }

        .tab {
            padding: 12px 16px;
            cursor: pointer;
            border-bottom: 2px solid transparent;
            font-weight: 500;
        }

        .tab-active {
            border-bottom-color: var(--primary);
            color: var(--primary);
        }

        .status-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 6px;
        }

        .status-good { background-color: var(--success); }
        .status-warning { background-color: var(--warning); }
        .status-bad { background-color: var(--danger); }

        /* Grid layouts */
        .grid-2 {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 16px;
        }

        .grid-3 {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 16px;
        }

        /* Responsive adjustments */
        @media (max-width: 1200px) {
            .grid-3 {
                grid-template-columns: repeat(2, 1fr);
            }
        }

        @media (max-width: 768px) {
            .grid-2, .grid-3 {
                grid-template-columns: 1fr;
            }
        }
        """

        # Actual layout
        self.app.layout = html.Div([
            # Include custom CSS
            html.Style(custom_css),

            # Main container
            html.Div([
                # Header
                html.Div([
                    html.H1('Alpha-Agent Performance Dashboard',
                            className='dashboard-title'),

                    html.Div([
                        html.Button('Refresh Data',
                                    id='refresh-button',
                                    className='button button-primary',
                                    style={'marginRight': '10px'}),

                        dcc.Dropdown(
                            id='time-range-dropdown',
                            options=[
                                {'label': 'Last Hour', 'value': '1h'},
                                {'label': 'Last 6 Hours', 'value': '6h'},
                                {'label': 'Last 24 Hours', 'value': '24h'},
                                {'label': 'Last 7 Days', 'value': '7d'},
                                {'label': 'Last 30 Days', 'value': '30d'},
                                {'label': 'All Data', 'value': 'all'}
                            ],
                            value='24h',
                            className='dropdown',
                            style={'width': '200px'}
                        )
                    ], style={'display': 'flex', 'alignItems': 'center'})
                ], className='header'),

                # Main tabs
                html.Div([
                    dcc.Tabs(id='main-tabs', value='overview', children=[
                        dcc.Tab(label='System Overview', value='overview'),
                        dcc.Tab(label='Agent Performance', value='agents'),
                        dcc.Tab(label='Communication', value='communication'),
                        dcc.Tab(label='Analytics', value='analytics')
                    ], colors={
                        "border": "var(--gray-200)",
                        "primary": "var(--primary)",
                        "background": "white"
                    })
                ]),

                # Content area
                html.Div(id='tab-content'),

                # Hidden div for storing data
                html.Div(id='data-store', style={'display': 'none'}),

                # Refresh interval
                dcc.Interval(
                    id='interval-component',
                    interval=self.update_interval * 1000,  # in milliseconds
                    n_intervals=0
                )
            ], className='container')
        ])

    def _setup_callbacks(self):
        """Setup dashboard callbacks with improved efficiency."""

        # Tab content handling
        @self.app.callback(
            Output('tab-content', 'children'),
            [Input('main-tabs', 'value')]
        )
        def render_tab_content(tab):
            if tab == 'overview':
                return self._create_overview_tab()
            elif tab == 'agents':
                return self._create_agents_tab()
            elif tab == 'communication':
                return self._create_communication_tab()
            elif tab == 'analytics':
                return self._create_analytics_tab()
            return html.Div("Unknown tab")

        # Data refresh callback
        @self.app.callback(
            Output('data-store', 'children'),
            [Input('interval-component', 'n_intervals'),
             Input('refresh-button', 'n_clicks'),
             Input('time-range-dropdown', 'value')]
        )
        def refresh_data(n_intervals, n_clicks, time_range):
            # Generate unique key for this data configuration
            cache_key = f"data_{time_range}"

            # Check if we have cached data and it's still fresh
            current_time = time.time()
            if (cache_key in self.data_cache and
                    cache_key in self.last_data_update and
                    current_time - self.last_data_update[cache_key] < self.data_cache_ttl):
                return self.data_cache[cache_key]

            # Convert time range to datetime
            delta = calculate_time_window(time_range)
            start_date = None if time_range == 'all' else datetime.now() - delta

            try:
                # Get system data
                system_data = self.performance_tracker.get_performance_summary(time_range=time_range)

                # Get agent contributions
                contributions = self.contribution_analyzer.calculate_agent_contributions()

                # Compile data
                data = {
                    'system_data': system_data,
                    'contributions': contributions,
                    'time_range': time_range,
                    'timestamp': datetime.now().isoformat()
                }

                # Cache the data
                self.data_cache[cache_key] = json.dumps(data)
                self.last_data_update[cache_key] = current_time

                return json.dumps(data)
            except Exception as e:
                logger.error(f"Error refreshing data: {e}")
                # Return empty data
                return json.dumps({'error': str(e)})

        # System Health gauge callback
        @self.app.callback(
            Output('system-health-gauge', 'figure'),
            [Input('data-store', 'children')]
        )
        def update_system_health(data_json):
            try:
                data = json.loads(data_json) if data_json else {}
                system_data = data.get('system_data', {})

                # Get health score
                health_score = 0

                # Try to get health score from system data
                system_metrics = system_data.get('system_metrics', {})
                if system_metrics:
                    # Use avg response time and error rate to calculate health
                    avg_response_time = system_metrics.get('avg_response_time', 0)
                    error_rate = 0

                    # Find error rate in agent data
                    for agent_id, agent_data in system_data.get('agents', {}).items():
                        reliability = agent_data.get('metrics', {}).get('reliability', {})
                        if 'error_rate' in reliability:
                            error_rate += reliability['error_rate'].get('avg', 0)

                    # Normalize error rate by agent count
                    agent_count = len(system_data.get('agents', {}))
                    if agent_count > 0:
                        error_rate /= agent_count

                    # Calculate health score - lower response time and error rate is better
                    response_score = max(0, 100 - (avg_response_time / 20))  # 0-100 scale
                    error_score = max(0, 100 - error_rate)  # 0-100 scale

                    health_score = (response_score * 0.4) + (error_score * 0.6)

                # Create gauge chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=health_score,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "System Health"},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1},
                        'bar': {'color': "rgba(50, 86, 168, 0.9)"},
                        'steps': [
                            {'range': [0, 40], 'color': "rgba(255, 99, 132, 0.3)"},
                            {'range': [40, 70], 'color': "rgba(255, 205, 86, 0.3)"},
                            {'range': [70, 100], 'color': "rgba(75, 192, 192, 0.3)"}
                        ],
                        'threshold': {
                            'line': {'color': "rgba(50, 86, 168, 0.8)", 'width': 4},
                            'thickness': 0.75,
                            'value': health_score
                        }
                    }
                ))

                fig.update_layout(
                    height=250,
                    margin=dict(l=20, r=20, t=30, b=20),
                    paper_bgcolor='white',
                    font={'color': '#333'}
                )

                return fig

            except Exception as e:
                logger.error(f"Error updating system health: {e}")
                # Return empty gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=0,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "System Health"},
                    gauge={'axis': {'range': [0, 100]}}
                ))
                fig.update_layout(height=250)
                return fig

        # Agent contributions pie chart
        @self.app.callback(
            Output('agent-contributions-pie', 'figure'),
            [Input('data-store', 'children')]
        )
        def update_contributions_pie(data_json):
            try:
                data = json.loads(data_json) if data_json else {}
                contributions = data.get('contributions', {})

                if not contributions:
                    # Return empty chart
                    fig = go.Figure(go.Pie(
                        labels=['No Data'],
                        values=[1],
                        hole=.4,
                        marker={'colors': ['#f2f2f2']}
                    ))
                    fig.update_layout(
                        title="No agent data available",
                        height=300,
                        margin=dict(l=20, r=20, t=40, b=20)
                    )
                    return fig

                # Sort by contribution (highest first)
                sorted_contributions = sorted(
                    contributions.items(),
                    key=lambda x: x[1],
                    reverse=True
                )

                labels = [agent_id for agent_id, _ in sorted_contributions]
                values = [contrib for _, contrib in sorted_contributions]

                # Create pie chart
                fig = go.Figure(go.Pie(
                    labels=labels,
                    values=values,
                    hole=.4,
                    textinfo='label+percent',
                    marker=dict(colors=px.colors.qualitative.Plotly),
                    pull=[0.05 if i == 0 else 0 for i in range(len(labels))]  # Pull out largest slice
                ))

                fig.update_layout(
                    title="Agent Contributions",
                    height=300,
                    margin=dict(l=20, r=20, t=40, b=20),
                    legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
                )

                return fig

            except Exception as e:
                logger.error(f"Error updating contributions pie: {e}")
                # Return empty chart
                fig = go.Figure()
                fig.update_layout(title="Error loading contributions data")
                return fig

        # System metrics timeline
        @self.app.callback(
            Output('system-metrics-timeline', 'figure'),
            [Input('data-store', 'children')]
        )
        def update_system_metrics_timeline(data_json):
            try:
                # Load data from data store
                data = json.loads(data_json) if data_json else {}

                # Get agent data
                system_data = data.get('system_data', {})
                agents_data = system_data.get('agents', {})

                if not agents_data:
                    # Return empty chart
                    fig = go.Figure()
                    fig.update_layout(
                        title="No system data available",
                        xaxis=dict(title="Time"),
                        yaxis=dict(title="Value")
                    )
                    return fig

                # Collect metrics data points
                timeline_data = []

                for agent_id, agent_data in agents_data.items():
                    # Get agent metrics history
                    agent_df = self.performance_tracker.load_agent_history(agent_id)

                    if not agent_df.empty and 'timestamp' in agent_df.columns:
                        # Sample data to improve performance
                        if len(agent_df) > self.max_points:
                            agent_df = sample_time_series(agent_df, self.max_points)

                        # Add success rate data
                        success_rate_col = next((col for col in agent_df.columns
                                                 if 'success_rate' in col.lower()), None)
                        if success_rate_col and success_rate_col in agent_df.columns:
                            for timestamp, value in zip(agent_df['timestamp'], agent_df[success_rate_col]):
                                timeline_data.append({
                                    'timestamp': timestamp,
                                    'agent_id': agent_id,
                                    'metric': 'Success Rate',
                                    'value': value
                                })

                        # Add response time data
                        response_time_col = next((col for col in agent_df.columns
                                                  if 'response_time' in col.lower() and 'mean' in col.lower()), None)
                        if response_time_col and response_time_col in agent_df.columns:
                            for timestamp, value in zip(agent_df['timestamp'], agent_df[response_time_col]):
                                timeline_data.append({
                                    'timestamp': timestamp,
                                    'agent_id': agent_id,
                                    'metric': 'Response Time',
                                    'value': value
                                })

                if not timeline_data:
                    # Return empty chart
                    fig = go.Figure()
                    fig.update_layout(
                        title="No timeline data available",
                        xaxis=dict(title="Time"),
                        yaxis=dict(title="Value")
                    )
                    return fig

                # Convert to DataFrame
                timeline_df = pd.DataFrame(timeline_data)

                # Create figure with two Y axes
                fig = make_subplots(specs=[[{"secondary_y": True}]])

                # Add success rate lines
                success_df = timeline_df[timeline_df['metric'] == 'Success Rate']
                if not success_df.empty:
                    for agent_id, group in success_df.groupby('agent_id'):
                        fig.add_trace(
                            go.Scatter(
                                x=group['timestamp'],
                                y=group['value'],
                                name=f"{agent_id} Success Rate",
                                mode='lines',
                                line=dict(width=2)
                            ),
                            secondary_y=False
                        )

                # Add response time lines
                response_df = timeline_df[timeline_df['metric'] == 'Response Time']
                if not response_df.empty:
                    for agent_id, group in response_df.groupby('agent_id'):
                        fig.add_trace(
                            go.Scatter(
                                x=group['timestamp'],
                                y=group['value'],
                                name=f"{agent_id} Response Time",
                                mode='lines',
                                line=dict(dash='dot', width=2)
                            ),
                            secondary_y=True
                        )

                # Update axis labels
                fig.update_xaxes(title_text="Time")
                fig.update_yaxes(title_text="Success Rate (%)", secondary_y=False)
                fig.update_yaxes(title_text="Response Time (ms)", secondary_y=True)

                # Update layout
                fig.update_layout(
                    title="Agent Performance Over Time",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    height=400,
                    margin=dict(l=60, r=60, t=80, b=60)
                )

                return fig

            except Exception as e:
                logger.error(f"Error updating metrics timeline: {e}")
                # Return empty chart
                fig = go.Figure()
                fig.update_layout(
                    title=f"Error loading data: {str(e)}",
                    height=400
                )
                return fig

    def _create_overview_tab(self):
        """Create content for system overview tab."""
        return html.Div([
            # System health and contributions row
            html.Div([
                html.Div([
                    html.Div([
                        html.H3("System Health", className="card-title"),
                        dcc.Graph(id='system-health-gauge')
                    ], className="card")
                ], className="six columns"),

                html.Div([
                    html.Div([
                        html.H3("Agent Contributions", className="card-title"),
                        dcc.Graph(id='agent-contributions-pie')
                    ], className="card")
                ], className="six columns")
            ], className="row"),

            # System metrics row
            html.Div([
                html.Div([
                    html.H3("Performance Metrics", className="card-title"),
                    dcc.Graph(id='system-metrics-timeline')
                ], className="card")
            ], className="row"),

            # System stats row
            html.Div([
                html.Div(id='system-stats', className="card")
            ], className="row"),
        ])

    def _create_agents_tab(self):
        """Create content for agent performance tab."""
        return html.Div([
            # Agent selector
            html.Div([
                html.Div([
                    html.H3("Agent Performance", className="card-title"),
                    html.P("Select an agent to view detailed metrics"),

                    dcc.Dropdown(
                        id='agent-selector-dropdown',
                        options=[],  # Will be populated by callback
                        placeholder='Select an agent',
                        className='dropdown',
                        style={'marginTop': '10px'}
                    )
                ], className="card")
            ], className="row"),

            # Agent details
            html.Div(id='agent-details')
        ])

    def _create_communication_tab(self):
        """Create content for communication tab."""
        return html.Div([
            html.Div([
                html.H3("Agent Communication Network", className="card-title"),
                html.P("Visualization of communication patterns between agents"),
                html.Div(id='communication-graph', style={'height': '600px'})
            ], className="card"),

            html.Div([
                html.H3("Communication Statistics", className="card-title"),
                html.Div(id='communication-stats')
            ], className="card")
        ])

    def _create_analytics_tab(self):
        """Create content for analytics tab."""
        return html.Div([
            html.Div([
                html.H3("Performance Analysis", className="card-title"),
                html.P("Select metrics to analyze"),

                dcc.Dropdown(
                    id='metrics-dropdown',
                    options=[
                        {'label': 'Success Rate', 'value': 'success_rate'},
                        {'label': 'Error Rate', 'value': 'error_rate'},
                        {'label': 'Response Time', 'value': 'response_time'},
                        {'label': 'Latency', 'value': 'latency'},
                        {'label': 'Throughput', 'value': 'throughput'},
                        {'label': 'Accuracy', 'value': 'accuracy'}
                    ],
                    value=['success_rate', 'response_time'],
                    multi=True,
                    className='dropdown',
                    style={'marginTop': '10px'}
                ),

                html.Div(id='performance-analysis-results', style={'marginTop': '20px'})
            ], className="card"),

            html.Div([
                html.H3("Anomaly Detection", className="card-title"),
                html.Div(id='anomalies-display')
            ], className="card")
        ])

    def run_server(self, debug=False, port=8050, host='0.0.0.0'):
        """
        Run the dashboard server.

        Args:
            debug: Whether to run in debug mode
            port: Server port
            host: Server host
        """
        self.app.run_server(debug=debug, port=port, host=host)


# Add callback for populating agent selector dropdown
def add_agent_dropdown_callback(dashboard):
    """
    Add callback for agent selector dropdown.
    This needs to be separate due to Dash's callback requirement.

    Args:
        dashboard: Dashboard instance
    """

    @dashboard.app.callback(
        Output('agent-selector-dropdown', 'options'),
        [Input('data-store', 'children')]
    )
    def update_agent_dropdown(data_json):
        try:
            data = json.loads(data_json) if data_json else {}
            system_data = data.get('system_data', {})
            agents = system_data.get('agents', {})

            options = []
            for agent_id, agent_data in agents.items():
                agent_type = agent_data.get('agent_type', 'Unknown')
                options.append({'label': f"{agent_id} ({agent_type})", 'value': agent_id})

            return options

        except Exception as e:
            logger.error(f"Error updating agent dropdown: {e}")
            return []


# Add callback for agent details
def add_agent_details_callback(dashboard):
    """
    Add callback for agent details section.

    Args:
        dashboard: Dashboard instance
    """

    @dashboard.app.callback(
        Output('agent-details', 'children'),
        [Input('agent-selector-dropdown', 'value'),
         Input('data-store', 'children')]
    )
    def display_agent_details(selected_agent, data_json):
        if not selected_agent:
            return html.Div("Select an agent to view details", style={'margin': '20px', 'color': '#888'})

        try:
            data = json.loads(data_json) if data_json else {}
            system_data = data.get('system_data', {})
            agents = system_data.get('agents', {})

            if selected_agent not in agents:
                return html.Div(f"No data available for agent {selected_agent}",
                                style={'margin': '20px', 'color': '#888'})

            agent_data = agents[selected_agent]

            # Create metrics cards
            cards = []

            for category, metrics in agent_data.get('metrics', {}).items():
                metric_items = []

                for metric_name, metric_values in metrics.items():
                    # Format metric value
                    if 'latest' in metric_values:
                        value = metric_values['latest']
                        formatted_value = f"{value:.2f}" if isinstance(value, float) else str(value)

                        metric_items.append(html.Div([
                            html.Span(metric_name.replace('_', ' ').title(), style={'fontWeight': '500'}),
                            html.Span(formatted_value, style={'float': 'right', 'fontFamily': 'monospace'})
                        ], style={'margin': '8px 0', 'borderBottom': '1px solid #eee', 'paddingBottom': '8px'}))

                if metric_items:
                    cards.append(html.Div([
                        html.H4(category.title(), style={'marginTop': '0', 'marginBottom': '16px', 'color': '#555'}),
                        html.Div(metric_items)
                    ], className='card'))

            # Create layout with cards in grid
            return html.Div([
                html.Div([
                    html.Div(cards, className='grid-2')
                ], className='row')
            ])

        except Exception as e:
            logger.error(f"Error displaying agent details: {e}")
            return html.Div(f"Error loading agent details: {str(e)}", style={'margin': '20px', 'color': '#d32f2f'})


# Main execution
if __name__ == '__main__':
    # Initialize components
    metrics_tracker = AgentMetricsTracker()
    agent_evaluator = AgentEvaluator()

    # Create dashboard
    dashboard = VisualizationDashboard(metrics_tracker, agent_evaluator)

    # Add callbacks
    add_agent_dropdown_callback(dashboard)
    add_agent_details_callback(dashboard)

    # Run server
    dashboard.run_server(debug=True)
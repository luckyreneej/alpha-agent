import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import networkx as nx
import seaborn as sns
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def plot_metric_timeline(timestamps, values, title, ylabel, xlabel="Time",
                         color='blue', marker='o', output_path=None):
    """
    Plot metric values over time.

    Args:
        timestamps: List of timestamps
        values: List of metric values
        title: Plot title
        ylabel: Y-axis label
        xlabel: X-axis label
        color: Line color
        marker: Point marker
        output_path: Optional path to save the plot

    Returns:
        Figure object or output path if saved
    """
    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, values, marker=marker, linestyle='-', color=color)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.grid(True, alpha=0.3)
    plt.gcf().autofmt_xdate()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return output_path
    else:
        return plt.gcf()


def plot_multi_metric_timeline(df, x_column, y_columns, labels=None, title="Metrics Over Time",
                               xlabel="Time", ylabel="Value", output_path=None):
    """
    Plot multiple metrics on the same timeline.

    Args:
        df: DataFrame with data
        x_column: Column to use for x-axis
        y_columns: List of columns to plot on y-axis
        labels: Optional list of labels for y-columns
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        output_path: Optional path to save the plot

    Returns:
        Figure object or output path if saved
    """
    plt.figure(figsize=(12, 6))

    if labels is None:
        labels = y_columns

    # Get a color palette for the lines
    colors = sns.color_palette("husl", len(y_columns))

    # Plot each metric
    for i, col in enumerate(y_columns):
        plt.plot(df[x_column], df[col], label=labels[i], color=colors[i])

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gcf().autofmt_xdate()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return output_path
    else:
        return plt.gcf()


def plot_communication_network(graph, node_size_attr=None, edge_weight_attr=None,
                               output_file=None, title="Agent Communication Network"):
    """
    Plot communication network graph.

    Args:
        graph: NetworkX graph
        node_size_attr: Optional node attribute to use for sizing
        edge_weight_attr: Optional edge attribute to use for line thickness
        output_file: Optional path to save the visualization
        title: Plot title

    Returns:
        Figure object or output path if saved
    """
    plt.figure(figsize=(12, 10))

    # Calculate node sizes
    if node_size_attr and nx.get_node_attributes(graph, node_size_attr):
        # Use the specified attribute
        node_attrs = nx.get_node_attributes(graph, node_size_attr)
        # Scale to reasonable sizes
        max_attr = max(node_attrs.values()) if node_attrs else 1
        node_sizes = {node: 100 + (attr / max_attr * 900)
                      for node, attr in node_attrs.items()}
    else:
        # Use degree as default size metric
        node_sizes = {node: 100 + (graph.degree(node) * 20) for node in graph.nodes()}

    # Calculate edge weights
    if edge_weight_attr and nx.get_edge_attributes(graph, edge_weight_attr):
        # Use the specified attribute
        edge_attrs = nx.get_edge_attributes(graph, edge_weight_attr)
        edge_weights = [edge_attrs.get((u, v), 1) for u, v in graph.edges()]
    else:
        # Use 'weight' attribute or default to 1
        edge_weights = [graph.get_edge_data(u, v).get('weight', 1) for u, v in graph.edges()]

    # Normalize edge weights for visualization
    if edge_weights:
        max_weight = max(edge_weights)
        normalized_weights = [w / max_weight * 5 for w in edge_weights]
    else:
        normalized_weights = []

    # Calculate layout
    pos = nx.spring_layout(graph, k=0.3, iterations=50)

    # Create colormap for nodes based on out-degree
    out_degrees = dict(graph.out_degree())
    node_colors = [plt.cm.viridis(out_degrees.get(node, 0) / max(out_degrees.values(), 1))
                   for node in graph.nodes()]

    # Draw nodes
    nx.draw_networkx_nodes(
        graph, pos,
        node_size=[node_sizes.get(node, 100) for node in graph.nodes()],
        node_color=node_colors,
        alpha=0.8
    )

    # Draw edges
    nx.draw_networkx_edges(
        graph, pos,
        width=normalized_weights if normalized_weights else 1.0,
        edge_color='gray',
        alpha=0.6,
        arrows=True,
        arrowsize=15,
        arrowstyle='-|>'
    )

    # Add labels
    nx.draw_networkx_labels(graph, pos, font_size=10, font_color='black')

    plt.title(title)
    plt.axis('off')

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        return output_file
    else:
        return plt.gcf()


def plot_heatmap(data, title="Correlation Heatmap", output_path=None):
    """
    Plot a heatmap for correlation or other matrix data.

    Args:
        data: DataFrame or 2D array
        title: Plot title
        output_path: Optional path to save the visualization

    Returns:
        Figure object or output path if saved
    """
    plt.figure(figsize=(10, 8))

    # Create the heatmap
    sns.heatmap(data, annot=True, cmap='viridis', center=0,
                linewidths=.5, cbar_kws={"shrink": .8})

    plt.title(title)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return output_path
    else:
        return plt.gcf()


def plot_contribution_breakdown(contributions, title="Agent Contribution Breakdown",
                                output_path=None):
    """
    Create a visualization of agent contributions.

    Args:
        contributions: Dictionary mapping agent IDs to contribution scores
        title: Plot title
        output_path: Optional path to save the visualization

    Returns:
        Figure object or output path if saved
    """
    if not contributions:
        logger.warning("No contribution data available for visualization")
        return None

    # Convert to DataFrame for plotting
    contrib_data = pd.DataFrame([
        (agent_id, score) for agent_id, score in contributions.items()
    ], columns=['agent_id', 'contribution'])

    # Sort by contribution (highest first)
    contrib_data = contrib_data.sort_values('contribution', ascending=False)

    # Create the plot with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 7),
                                   gridspec_kw={'width_ratios': [2, 1]})

    # Color mapping
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(contrib_data)))

    # Bar chart
    bars = ax1.bar(
        contrib_data['agent_id'],
        contrib_data['contribution'],
        color=colors
    )

    # Add percentage labels
    for bar in bars:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.1f}%",
            ha='center',
            va='bottom'
        )

    ax1.set_title("Agent Contribution Breakdown")
    ax1.set_xlabel("Agent ID")
    ax1.set_ylabel("Contribution (%)")
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Pie chart
    ax2.pie(
        contrib_data['contribution'],
        labels=contrib_data['agent_id'],
        autopct='%1.1f%%',
        startangle=90,
        colors=colors
    )
    ax2.set_title("Contribution Share")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return output_path
    else:
        return fig


def plot_multi_agent_metrics(metrics_df, agent_column, metric_columns,
                             time_column='timestamp', title=None, output_path=None):
    """
    Plot metrics for multiple agents over time.

    Args:
        metrics_df: DataFrame with metrics data
        agent_column: Column containing agent identifiers
        metric_columns: List of metric columns to plot
        time_column: Column containing timestamps
        title: Optional title for the plot
        output_path: Optional path to save the visualization

    Returns:
        Figure object or output path if saved
    """
    if metrics_df.empty:
        logger.warning("Empty DataFrame provided for metrics visualization")
        return None

    # Ensure time column is datetime
    if not pd.api.types.is_datetime64_dtype(metrics_df[time_column]):
        metrics_df[time_column] = pd.to_datetime(metrics_df[time_column])

    # Get unique agents
    agents = metrics_df[agent_column].unique()

    # Create subplots for each metric
    n_metrics = len(metric_columns)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 4 * n_metrics), sharex=True)

    # Use single axes if only one metric
    if n_metrics == 1:
        axes = [axes]

    # Get a color palette for the agents
    colors = sns.color_palette("husl", len(agents))

    for i, metric in enumerate(metric_columns):
        ax = axes[i]

        for j, agent in enumerate(agents):
            agent_data = metrics_df[metrics_df[agent_column] == agent]
            if not agent_data.empty and metric in agent_data.columns:
                ax.plot(
                    agent_data[time_column],
                    agent_data[metric],
                    label=agent,
                    color=colors[j],
                    marker='o' if len(agent_data) < 10 else None
                )

        ax.set_title(f"{metric}")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)

        if i == 0:  # Only add legend to first subplot
            ax.legend(loc='upper right')

    # Set overall title if provided
    if title:
        fig.suptitle(title, fontsize=16)

    # Format x-axis dates
    fig.autofmt_xdate()

    plt.tight_layout()
    if title:
        plt.subplots_adjust(top=0.95)  # Adjust for suptitle

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return output_path
    else:
        return fig


def create_performance_dashboard(metrics_data, output_dir, prefix="dashboard"):
    """
    Create a set of visualizations for a performance dashboard.

    Args:
        metrics_data: Dictionary of metrics data
        output_dir: Directory to save visualizations
        prefix: Prefix for file names

    Returns:
        Dictionary mapping chart names to file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    plots = {}

    # System overview plot
    if 'system_metrics' in metrics_data and metrics_data['system_metrics']:
        sys_metrics = pd.DataFrame(metrics_data['system_metrics'])
        if not sys_metrics.empty and 'timestamp' in sys_metrics.columns:
            plots['system_overview'] = plot_multi_metric_timeline(
                sys_metrics,
                'timestamp',
                ['overall_score', 'prediction_accuracy', 'trading_performance'],
                labels=['Overall Score', 'Prediction Accuracy', 'Trading Performance'],
                title="System Performance Over Time",
                output_path=os.path.join(output_dir, f"{prefix}_system_overview.png")
            )

    # Agent contributions plot
    if 'agent_contributions' in metrics_data and metrics_data['agent_contributions']:
        plots['agent_contributions'] = plot_contribution_breakdown(
            metrics_data['agent_contributions'],
            title="Agent Contribution Analysis",
            output_path=os.path.join(output_dir, f"{prefix}_agent_contributions.png")
        )

    # Prediction accuracy plot
    if 'prediction_metrics' in metrics_data and metrics_data['prediction_metrics']:
        pred_metrics = pd.DataFrame(metrics_data['prediction_metrics'])
        if not pred_metrics.empty and 'timestamp' in pred_metrics.columns:
            accuracy_cols = [col for col in pred_metrics.columns if 'accuracy' in col.lower()]
            if accuracy_cols:
                plots['prediction_accuracy'] = plot_multi_metric_timeline(
                    pred_metrics,
                    'timestamp',
                    accuracy_cols,
                    title="Prediction Accuracy Metrics",
                    output_path=os.path.join(output_dir, f"{prefix}_prediction_accuracy.png")
                )

            error_cols = [col for col in pred_metrics.columns if
                          any(err in col.lower() for err in ['error', 'mse', 'rmse', 'mae'])]
            if error_cols:
                plots['prediction_errors'] = plot_multi_metric_timeline(
                    pred_metrics,
                    'timestamp',
                    error_cols,
                    title="Prediction Error Metrics",
                    output_path=os.path.join(output_dir, f"{prefix}_prediction_errors.png")
                )

    # Trading performance plot
    if 'trading_metrics' in metrics_data and metrics_data['trading_metrics']:
        trade_metrics = pd.DataFrame(metrics_data['trading_metrics'])
        if not trade_metrics.empty and 'timestamp' in trade_metrics.columns:
            perf_cols = [col for col in trade_metrics.columns
                         if any(term in col.lower() for term in ['return', 'profit', 'sharpe', 'calmar'])]
            if perf_cols:
                plots['trading_performance'] = plot_multi_metric_timeline(
                    trade_metrics,
                    'timestamp',
                    perf_cols,
                    title="Trading Performance Metrics",
                    output_path=os.path.join(output_dir, f"{prefix}_trading_performance.png")
                )

    # Communication network plot
    if 'communication_graph' in metrics_data:
        graph = metrics_data['communication_graph']
        if graph and graph.nodes():
            plots['communication_network'] = plot_communication_network(
                graph,
                title="Agent Communication Network",
                output_file=os.path.join(output_dir, f"{prefix}_communication_network.png")
            )

    return plots


def plot_gauge_chart(value, min_val=0, max_val=100, title="Metric", thresholds=None, output_path=None):
    """
    Create a gauge chart for a single value.

    Args:
        value: Value to display
        min_val: Minimum scale value
        max_val: Maximum scale value
        title: Chart title
        thresholds: Optional dictionary with threshold values and colors
                    e.g., {50: 'red', 70: 'yellow', 90: 'green'}
        output_path: Optional path to save the visualization

    Returns:
        Figure object or output path if saved
    """
    if thresholds is None:
        thresholds = {
            0: "red",
            50: "yellow",
            70: "green"
        }

    # Create figure
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, polar=True)

    # Define gauge range (in radians)
    gauge_min = np.pi / 6  # 30 degrees
    gauge_max = np.pi * 11 / 6  # 330 degrees
    gauge_range = gauge_max - gauge_min

    # Calculate value in radians
    val_rad = gauge_min + (value - min_val) / (max_val - min_val) * gauge_range

    # Create background arcs for thresholds
    sorted_thresholds = sorted(thresholds.items())
    for i in range(len(sorted_thresholds)):
        threshold, color = sorted_thresholds[i]
        next_threshold = max_val
        if i < len(sorted_thresholds) - 1:
            next_threshold = sorted_thresholds[i + 1][0]

        # Calculate arc bounds in radians
        thresh_min_rad = gauge_min + (threshold - min_val) / (max_val - min_val) * gauge_range
        thresh_max_rad = gauge_min + (next_threshold - min_val) / (max_val - min_val) * gauge_range

        # Draw the arc
        ax.barh(0, thresh_max_rad - thresh_min_rad, left=thresh_min_rad, height=0.1, color=color, alpha=0.5)

    # Draw the needle
    ax.plot([0, val_rad], [0, 0], 'k-', linewidth=3)
    ax.scatter(val_rad, 0, s=100, color='black', zorder=5)

    # Add value text
    fig.text(0.5, 0.25, f"{value:.1f}", ha='center', va='center', fontsize=24)

    # Add title
    plt.title(title, pad=20)

    # Configure axes
    ax.set_ylim(-0.1, 0.1)
    ax.set_theta_offset(np.pi / 2)
    ax.grid(False)
    ax.set_yticklabels([])
    ax.set_xticks(np.linspace(gauge_min, gauge_max, 5))
    ax.set_xticklabels([str(int(min_val)),
                        str(int(min_val + (max_val - min_val) * 0.25)),
                        str(int(min_val + (max_val - min_val) * 0.5)),
                        str(int(min_val + (max_val - min_val) * 0.75)),
                        str(int(max_val))])

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return output_path
    else:
        return fig


def plot_stacked_area_chart(df, x_column, y_columns, labels=None, colors=None,
                            title="Stacked Area Chart", xlabel="Time", ylabel="Value",
                            output_path=None):
    """
    Create a stacked area chart for time series data.

    Args:
        df: DataFrame with data
        x_column: Column to use for x-axis
        y_columns: List of columns to stack
        labels: Optional list of labels for the areas
        colors: Optional list of colors for the areas
        title: Chart title
        xlabel: X-axis label
        ylabel: Y-axis label
        output_path: Optional path to save the visualization

    Returns:
        Figure object or output path if saved
    """
    plt.figure(figsize=(12, 8))

    if labels is None:
        labels = y_columns

    if colors is None:
        colors = plt.cm.viridis(np.linspace(0, 0.9, len(y_columns)))

    plt.stackplot(df[x_column], [df[col] for col in y_columns],
                  labels=labels, colors=colors, alpha=0.8)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)

    # Format x-axis dates if x is datetime
    if pd.api.types.is_datetime64_dtype(df[x_column]):
        plt.gcf().autofmt_xdate()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return output_path
    else:
        return plt.gcf()


def plot_agent_metrics_radar(metrics_dict, agent_ids=None, metric_keys=None,
                             title="Agent Metrics Comparison", output_path=None):
    """
    Create a radar chart comparing multiple agents across metrics.

    Args:
        metrics_dict: Dictionary mapping agent IDs to their metrics
        agent_ids: Optional list of agent IDs to include
        metric_keys: Optional list of metric keys to include
        title: Chart title
        output_path: Optional path to save the visualization

    Returns:
        Figure object or output path if saved
    """
    if not metrics_dict:
        logger.warning("Empty metrics dictionary provided for radar chart")
        return None

    # Filter agents if specified
    if agent_ids:
        metrics_dict = {agent_id: metrics for agent_id, metrics in metrics_dict.items()
                        if agent_id in agent_ids}

    if not metrics_dict:
        logger.warning("No agent data available after filtering")
        return None

    # Get common metrics across all agents
    if metric_keys is None:
        # Get all metric keys from the first agent
        first_agent = next(iter(metrics_dict.values()))
        metric_keys = list(first_agent.keys())

    # Filter to metrics that exist for all agents
    common_metrics = []
    for key in metric_keys:
        if all(key in agent_metrics for agent_metrics in metrics_dict.values()):
            common_metrics.append(key)

    if not common_metrics:
        logger.warning("No common metrics found across agents")
        return None

    # Number of metrics
    N = len(common_metrics)

    # Create angle for each metric
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    # Add metric labels
    plt.xticks(angles[:-1], common_metrics, size=8)

    # Colors for each agent
    colors = plt.cm.tab10(np.linspace(0, 1, len(metrics_dict)))

    # Plot each agent
    for i, (agent_id, metrics) in enumerate(metrics_dict.items()):
        values = [metrics.get(metric, 0) for metric in common_metrics]
        values += values[:1]  # Close the polygon

        ax.plot(angles, values, linewidth=2, linestyle='solid', label=agent_id, color=colors[i])
        ax.fill(angles, values, color=colors[i], alpha=0.1)

    plt.title(title)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return output_path
    else:
        return fig

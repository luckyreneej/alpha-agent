# Alpha-Agent Evaluation Framework

## Overview

The Alpha-Agent Evaluation Framework provides comprehensive tools for monitoring, analyzing, and visualizing the performance of multi-agent systems. It enables detailed tracking of individual agents and system-wide metrics to identify strengths, weaknesses, and opportunities for optimization.

## Key Components

### Agent Metrics (`agent_metrics.py`)

Tracks and calculates performance metrics for individual agents:

- **Accuracy Metrics**: Prediction accuracy, error rates, directional accuracy
- **Latency Metrics**: Execution time, response time, processing delays  
- **Throughput Metrics**: Operations per second, message processing rates
- **Reliability Metrics**: Success rates, error counts, stability measurements

### Agent Evaluator (`agent_evaluator.py`)

Evaluates the performance of agents and the overall system:

- **Prediction Evaluation**: Assesses prediction accuracy and reliability
- **Trading Performance**: Evaluates trading strategy effectiveness
- **System-wide Metrics**: Tracks overall system health and performance
- **Metric Visualization**: Generates performance visualizations

### Communication Metrics (`communication_metrics.py`)

Analyzes communication patterns between agents:

- **Message Tracking**: Records message volume and types
- **Network Analysis**: Visualizes agent communication networks
- **Bottleneck Detection**: Identifies communication inefficiencies
- **Response Time Analysis**: Measures agent responsiveness

### Contribution Analyzer (`contribution_analyzer.py`)

Measures each agent's contribution to overall system performance:

- **Contribution Scoring**: Quantifies agent impacts on system performance
- **Historical Tracking**: Monitors contribution changes over time
- **Comparative Analysis**: Compares different agents' contributions
- **Performance Attribution**: Identifies key performance drivers

### Performance Tracker (`performance_tracker.py`)

Tracks and analyzes performance over time:

- **Historical Analysis**: Records and compares performance metrics
- **Trend Identification**: Detects performance patterns
- **Anomaly Detection**: Identifies unusual performance behavior
- **Baseline Comparison**: Measures performance against established baselines

### Visualization Dashboard (`dashboard.py`)

Interactive visualization of system and agent metrics:

- **System Overview**: High-level system performance view
- **Agent Details**: Individual agent performance metrics
- **Communication Visualization**: Agent interaction patterns
- **Time-based Analysis**: Performance trends over time

## Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/alpha-agent-evaluation.git
cd alpha-agent-evaluation

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

#### 1. Track Individual Agent Metrics

```python
from agent_metrics import AgentMetricsTracker

# Initialize tracker
metrics_tracker = AgentMetricsTracker()

# Register an agent
agent_metrics = metrics_tracker.register_agent("prediction_agent", "PredictionModel")

# Record metrics
agent_metrics.record_execution_time("task_123", 150.5)  # milliseconds
agent_metrics.record_success("task_123")
agent_metrics.record_prediction(predicted_value=10.5, actual_value=10.7)

# Save metrics
agent_metrics.save_metrics()
```

#### 2. Evaluate Prediction Performance

```python
from agent_evaluator import AgentEvaluator

# Initialize evaluator
evaluator = AgentEvaluator()

# Evaluate predictions
predictions = {"AAPL": [155.3, 156.7, 158.2]}
actual_values = {"AAPL": [155.5, 157.0, 159.1]}

metrics = evaluator.evaluate_prediction_agent(predictions, actual_values)
print(f"Prediction accuracy: {metrics['avg_directional_accuracy']:.2f}")
print(f"Overall score: {metrics['overall_score']:.2f}/100")
```

#### 3. Analyze Communication Patterns

```python
from communication_metrics import CommunicationMetrics

# Initialize communication metrics
comm_metrics = CommunicationMetrics()

# Record messages
comm_metrics.record_message(
    sender_id="data_agent",
    receiver_id="prediction_agent",
    message_type="data_update"
)

# Record response time
comm_metrics.record_response_time(
    sender_id="data_agent",
    receiver_id="prediction_agent",
    request_id="req123",
    response_time_ms=45.2
)

# Identify bottlenecks
bottlenecks = comm_metrics.identify_bottlenecks()

# Visualize network
network_image = comm_metrics.plot_communication_network()
```

#### 4. Analyze Agent Contributions

```python
from contribution_analyzer import ContributionAnalyzer

# Initialize analyzer
analyzer = ContributionAnalyzer()

# Calculate contributions
contributions = analyzer.calculate_agent_contributions()
print(f"Agent contributions: {contributions}")

# Track changes over time
history = analyzer.calculate_contribution_history(window_size=7)

# Visualize contributions
analyzer.plot_contribution_breakdown()
```

#### 5. Track Performance Over Time

```python
from performance_tracker import PerformanceTracker

# Initialize tracker
tracker = PerformanceTracker()

# Load agent history
history = tracker.load_agent_history("prediction_agent", start_date="2023-01-01")

# Compare agents
comparison = tracker.compare_agents(
    agent_ids=["prediction_agent", "sentiment_agent"],
    metric_names=["success_rate", "response_time"]
)

# Detect anomalies
anomalies = tracker.detect_performance_anomalies("prediction_agent")
```

#### 6. Launch Visualization Dashboard

```python
from dashboard import VisualizationDashboard
from agent_metrics import AgentMetricsTracker
from agent_evaluator import AgentEvaluator

# Initialize components
metrics_tracker = AgentMetricsTracker()
agent_evaluator = AgentEvaluator()

# Create dashboard
dashboard = VisualizationDashboard(metrics_tracker, agent_evaluator)

# Run server
dashboard.run_server(debug=True, port=8050)
```

## Advanced Usage

### Creating Performance Baselines

```python
from performance_tracker import PerformanceTracker

tracker = PerformanceTracker()

# Create baseline from recent data
baseline = tracker.create_performance_baseline("trading_agent", baseline_days=30)

# Compare current performance to baseline
comparison = tracker.compare_to_baseline("trading_agent", time_range="7d")
```

### Detecting Performance Shifts

```python
from performance_tracker import PerformanceTracker

tracker = PerformanceTracker()

# Detect gradual performance changes
shifts = tracker.detect_performance_shifts("prediction_agent", window_days=90)

for metric, data in shifts['shifts'].items():
    print(f"{metric}: {data['status']} by {data['overall_change']:.2f}%")
```

### Analyzing Communication Efficiency

```python
from communication_metrics import CommunicationMetrics

comm_metrics = CommunicationMetrics()

# Get communication summary
summary = comm_metrics.get_communication_summary(start_time="2023-01-01")

# Get network graph data
network = comm_metrics.analyze_information_flow()

# Identify central agents
for agent_id in network['hubs']:
    print(f"Hub agent: {agent_id['agent_id']} (centrality: {agent_id['centrality']:.2f})")
```

### Generating System Reports

```python
from agent_metrics import AgentMetricsTracker

tracker = AgentMetricsTracker()

# Generate system-wide report
report = tracker.generate_system_report(output_format="html")

# Save report to file
with open("system_report.html", "w") as f:
    f.write(report)
```

## Integrating with Agents

To integrate metrics tracking into your agent implementations:

```python
class MyAgent:
    def __init__(self, agent_id, agent_type, metrics_tracker):
        self.agent_id = agent_id
        self.metrics = metrics_tracker.register_agent(agent_id, agent_type)
        
    def process_request(self, request_id, data):
        # Start timing
        self.metrics.record_request_start(request_id)
        
        try:
            # Process the request
            result = self._process_data(data)
            
            # Record success
            self.metrics.record_success(request_id)
            return result
            
        except Exception as e:
            # Record error
            self.metrics.record_error(request_id, type(e).__name__, {"details": str(e)})
            raise
            
        finally:
            # Record response time
            self.metrics.record_request_end(request_id)
```

## Customizing Metrics and Evaluation

### Custom Metric Weights

```python
from contribution_analyzer import ContributionAnalyzer

analyzer = ContributionAnalyzer()

# Custom weights for different metrics
custom_weights = {
    'reliability_success_rate': 0.3,
    'latency_mean_latency': -0.2,
    'throughput_throughput_ops_per_sec': 0.2,
    'accuracy_accuracy': 0.3
}

# Calculate with custom weights
contributions = analyzer.calculate_agent_contributions(metric_weights=custom_weights)
```

### Custom Performance Scoring

```python
from agent_evaluator import AgentEvaluator

evaluator = AgentEvaluator()

# Custom scoring function (example: prioritize directional accuracy)
def custom_scoring(metrics):
    directional_weight = 0.8
    error_weight = 0.2
    
    dir_acc_score = metrics['avg_directional_accuracy'] * 100
    error_score = 100 - (metrics['avg_rmse'] * 20)
    
    return directional_weight * dir_acc_score + error_weight * error_score

# Apply custom scoring
predictions = {"AAPL": [155.3, 156.7, 158.2]}
actual_values = {"AAPL": [155.5, 157.0, 159.1]}

metrics = evaluator.evaluate_prediction_agent(predictions, actual_values)
custom_score = custom_scoring(metrics)
```

## Best Practices

1. **Regular Metrics Collection**: Configure agents to record metrics consistently for accurate analysis

2. **Baseline Creation**: Establish performance baselines during stable operation periods for comparison

3. **Anomaly Thresholds**: Adjust anomaly detection thresholds based on your system's normal variability

4. **Sampling for Large Systems**: Use appropriate sampling for systems with numerous agents or high message volumes

5. **Periodic Cleanup**: Configure automatic cleanup of old metrics data to manage storage requirements

6. **Report Automation**: Schedule regular system reports for monitoring trends

## Troubleshooting

- **Missing Metrics**: Ensure all agents are correctly registered with the metrics tracker

- **Visualization Issues**: Check that the data for the selected time range is available

- **Performance Problems**: Use sampling for large datasets and consider using the database backend

- **Inconsistent Results**: Validate that all agents use consistent timekeeping and metric recording

## Advanced Configuration

The evaluation framework supports various configuration options:

```python
# Example configuration
config = {
    'data_dir': 'metrics_data',
    'update_interval': 60,  # seconds
    'max_history': 10000,   # maximum metrics to keep in memory
    'auto_save_interval': 300,  # seconds between auto-saves
    'baseline_days': 30,    # days of data for baseline
    'anomaly_threshold': 2.0,  # standard deviations for anomaly detection
    'sampling_enabled': True,  # enable data sampling for large datasets
    'db_backend': False      # use database instead of file storage
}

# Apply configuration to components
metrics_tracker = AgentMetricsTracker(metrics_dir=config['data_dir'])
evaluator = AgentEvaluator(evaluation_dir=config['data_dir'])
```

## Additional Resources

- **Visualization Guide**: See `visualization_guide.md` for customizing charts and plots
- **Metrics Reference**: Detailed explanation of available metrics in `metrics_reference.md`
- **Example Scripts**: Check the `examples/` directory for complete usage examples
- **Dashboard Guide**: Instructions for customizing the visualization dashboard in `dashboard_guide.md`
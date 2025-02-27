# Alpha-Agent Evaluation Metrics Module

## Overview

The evaluation module provides comprehensive tools for tracking, analyzing and visualizing agent performance in the Alpha-Agent system. It enables detailed monitoring of individual agents and the overall system to identify strengths, weaknesses, and opportunities for optimization.

## Key Components

### Agent Metrics Tracker (`agent_metrics.py`)

- **Purpose**: Track individual agent performance metrics including accuracy, latency, throughput and reliability
- **Key Features**:
  - Execution time tracking
  - Success/error rate monitoring 
  - Prediction accuracy measurement
  - Response time analysis

### Communication Metrics (`communication_metrics.py`)

- **Purpose**: Analyze inter-agent communication patterns and effectiveness
- **Key Features**:
  - Message volume tracking
  - Communication network analysis
  - Bottleneck identification
  - Response time measurements
  - Information flow visualization

### Contribution Analyzer (`contribution_analyzer.py`)

- **Purpose**: Measure each agent's contribution to overall system performance
- **Key Features**:
  - Contribution scoring based on multiple metrics
  - Adaptive weight calculation
  - Historical contribution tracking
  - Visualization of agent contributions

### Visualization Dashboard (`dashboard.py`)

- **Purpose**: Interactive visualization of system and agent metrics
- **Key Features**:
  - System overview
  - Individual agent performance analysis
  - Agent comparison
  - Performance trends over time

## Usage Examples

### Basic Metrics Tracking

```python
from evaluation.agent_metrics import AgentMetricsTracker

# Initialize metrics tracker
metrics_tracker = AgentMetricsTracker("data/metrics")

# Register an agent
agent_metrics = metrics_tracker.register_agent("agent_1", "MarketAnalyst")

# Record agent activities
agent_metrics.record_execution_time("task_123", 150.5)  # 150.5ms
agent_metrics.record_success("task_123")

# For prediction agents
agent_metrics.record_prediction(
    predicted_value=10.5, 
    actual_value=10.7, 
    metadata={"confidence": 0.85}
)

# Save metrics
agent_metrics.save_metrics()
```

### Communication Tracking

```python
from evaluation.communication_metrics import CommunicationMetrics

# Initialize communication metrics
comm_metrics = CommunicationMetrics("data/metrics/communication")

# Record messages between agents
comm_metrics.record_message(
    sender_id="agent_1",
    receiver_id="agent_2",
    message_type="query",
    request_id="req_123"
)

# Record response time
comm_metrics.record_response_time(
    sender_id="agent_1", 
    receiver_id="agent_2",
    request_id="req_123", 
    response_time_ms=45.2
)

# Identify communication bottlenecks
bottlenecks = comm_metrics.identify_bottlenecks()

# Visualize communication network
comm_metrics.plot_communication_network()
```

### Contribution Analysis

```python
from evaluation.contribution_analyzer import ContributionAnalyzer

# Initialize analyzer
analyzer = ContributionAnalyzer("data/metrics")

# Calculate agent contributions
contributions = analyzer.calculate_agent_contributions()

# Get adaptive weights for agents
weights = analyzer.calculate_adaptive_weights(lookback_days=30)

# Visualize contributions
analyzer.plot_contribution_breakdown()
```

### Running the Dashboard

To launch the interactive dashboard:

```bash
streamlit run evaluation/dashboard.py -- --data-dir data/metrics
```

## Integration with Agents

Typically, you would integrate the metrics tracking into your agent's operation cycle:

```python
class MyAgent:
    def __init__(self, agent_id, agent_type):
        # Initialize metrics
        self.metrics = metrics_tracker.register_agent(agent_id, agent_type)
        
    def process_request(self, request):
        # Generate unique task ID
        task_id = f"task_{uuid.uuid4()}"
        
        start_time = time.time()
        try:
            # Process the request
            result = self._perform_processing(request)
            
            # Record success
            self.metrics.record_success(task_id)
            return result
            
        except Exception as e:
            # Record error
            self.metrics.record_error(task_id, str(type(e)), {"details": str(e)})
            raise
        finally:
            # Record execution time
            execution_time = (time.time() - start_time) * 1000  # Convert to ms
            self.metrics.record_execution_time(task_id, execution_time)
```

## Example Script

See the included example script that demonstrates all components working together:

```bash
python examples/evaluation_example.py
```

This will simulate multiple agents performing tasks and communicating, then analyze and visualize the results.

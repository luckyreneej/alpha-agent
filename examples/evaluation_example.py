#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluation Metrics Example

This example demonstrates how to use the evaluation metrics modules to track,
analyze, and visualize agent performance.
"""

import os
import sys
import time
import random
from datetime import datetime, timedelta
import logging

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import evaluation modules
from evaluation.agent_metrics import AgentMetrics, AgentMetricsTracker
from evaluation.communication_metrics import CommunicationMetrics
from evaluation.contribution_analyzer import ContributionAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def simulate_agents(num_agents=5, duration_seconds=60, metrics_dir='data/metrics'):
    """
    Simulate multiple agents performing tasks and track their metrics.
    
    Args:
        num_agents: Number of agents to simulate
        duration_seconds: Duration of simulation in seconds
        metrics_dir: Directory for storing metrics
    """
    # Ensure metrics directory exists
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Initialize metrics tracker
    metrics_tracker = AgentMetricsTracker(metrics_dir)
    
    # Initialize communication metrics
    comm_metrics = CommunicationMetrics(os.path.join(metrics_dir, 'communication'))
    
    # Create agent types
    agent_types = [
        'MarketAnalyst',
        'PredictionModel',
        'StrategyGenerator',
        'PortfolioManager',
        'RiskManager'
    ]
    
    # Create agents with different types
    agents = []
    for i in range(num_agents):
        agent_id = f"agent_{i+1}"
        agent_type = agent_types[i % len(agent_types)]
        agent_metrics = metrics_tracker.register_agent(agent_id, agent_type)
        agents.append((agent_id, agent_type, agent_metrics))
    
    print(f"Simulating {num_agents} agents for {duration_seconds} seconds...")
    
    # Run simulation
    start_time = time.time()
    end_time = start_time + duration_seconds
    task_id = 0
    
    while time.time() < end_time:
        # Simulate agent operations
        for agent_id, agent_type, agent_metrics in agents:
            # Generate a unique task ID
            task_id += 1
            task_id_str = f"task_{task_id}"
            
            # Simulate execution time (between 10ms and 1000ms)
            execution_time = random.uniform(10, 1000)
            agent_metrics.record_execution_time(task_id_str, execution_time)
            
            # Simulate success/failure (90% success rate)
            if random.random() < 0.9:
                agent_metrics.record_success(task_id_str)
            else:
                agent_metrics.record_error(task_id_str, "simulation_error", 
                                         {"reason": "Random simulation error"})
            
            # For prediction agents, simulate predictions and actuals
            if agent_type == 'PredictionModel' or agent_type == 'StrategyGenerator':
                # Simulate a prediction between 0 and 100
                predicted = random.uniform(0, 100)
                # Actual is predicted +/- some error
                error = random.normalvariate(0, 10)  # Normal distribution with mean 0, std 10
                actual = predicted + error
                
                agent_metrics.record_prediction(predicted, actual, metadata={
                    "prediction_target": "stock_price",
                    "confidence": random.uniform(0.5, 0.95)
                })
            
            # Simulate communications between agents
            if random.random() < 0.3:  # 30% chance of communication
                # Select random receiver
                receiver_idx = random.randint(0, num_agents - 1)
                receiver_id = f"agent_{receiver_idx+1}"
                
                if receiver_id != agent_id:  # Don't communicate with self
                    # Pick message type
                    msg_type = random.choice(['query', 'response', 'update', 'alert'])
                    
                    # Generate random message size
                    message_size = random.randint(100, 10000)  # bytes
                    
                    # Record the message
                    request_id = f"req_{task_id}"
                    comm_metrics.record_message(
                        sender_id=agent_id,
                        receiver_id=receiver_id,
                        message_type=msg_type,
                        size_bytes=message_size,
                        request_id=request_id,
                        content_summary=f"Simulated {msg_type} message"
                    )
                    
                    # Record response time
                    response_time = random.uniform(5, 500)  # ms
                    comm_metrics.record_response_time(
                        sender_id=agent_id,
                        receiver_id=receiver_id,
                        request_id=request_id,
                        response_time_ms=response_time
                    )
        
        # Small delay between iterations
        time.sleep(0.01)
    
    # Save all metrics
    metrics_tracker.save_all_metrics()
    comm_metrics.save_metrics()
    
    print(f"Simulation completed. Metrics saved to {metrics_dir}")
    return metrics_dir

def analyze_results(metrics_dir):
    """
    Analyze the results of the simulation.
    
    Args:
        metrics_dir: Directory containing metrics data
    """
    print("\nAnalyzing agent metrics...")
    
    # Initialize analyzer
    contribution_analyzer = ContributionAnalyzer(metrics_dir)
    
    # Calculate agent contributions
    contributions = contribution_analyzer.calculate_agent_contributions()
    
    # Print contributions
    print("\nAgent Contributions:")
    for agent_id, score in sorted(contributions.items(), key=lambda x: x[1], reverse=True):
        print(f"  {agent_id}: {score:.2f}%")
    
    # Generate visualization
    print("\nGenerating visualizations...")
    viz_file = contribution_analyzer.plot_contribution_breakdown()
    if viz_file:
        print(f"  Contribution breakdown chart saved to: {viz_file}")
    
    # Get communication stats
    comm_metrics = CommunicationMetrics(os.path.join(metrics_dir, 'communication'))
    comm_stats = comm_metrics.get_message_stats()
    
    print("\nCommunication Statistics:")
    print(f"  Total Messages: {comm_stats.get('total_messages', 0)}")
    print(f"  Active Agents: {comm_stats.get('active_agents', 0)}")
    print(f"  Avg Response Time: {comm_stats.get('avg_response_time', 0):.2f} ms")
    
    # Identify bottlenecks
    bottlenecks = comm_metrics.identify_bottlenecks()
    if bottlenecks:
        print("\nCommunication Bottlenecks:")
        for bottleneck in bottlenecks[:3]:  # Show top 3
            print(f"  {bottleneck['sender_id']} → {bottleneck['receiver_id']}: "
                 f"{bottleneck['avg_response_time']:.2f} ms ({bottleneck['ratio']:.2f}x avg)")
    
    # Generate communication network visualization
    network_viz = comm_metrics.plot_communication_network()
    if network_viz:
        print(f"\nCommunication network visualization saved to: {network_viz}")
    
    print("\nTo view interactive dashboard, run:")
    print(f"  streamlit run evaluation/dashboard.py -- --data-dir {metrics_dir}")

def main():
    """
    Main function to run the example.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Run evaluation metrics example")
    parser.add_argument('--agents', type=int, default=5, help='Number of agents to simulate')
    parser.add_argument('--duration', type=int, default=60, help='Duration in seconds')
    parser.add_argument('--metrics-dir', type=str, default='data/metrics', 
                      help='Directory for storing metrics')
    parser.add_argument('--analyze-only', action='store_true', 
                      help='Skip simulation and only analyze existing results')
    
    args = parser.parse_args()
    
    if not args.analyze_only:
        metrics_dir = simulate_agents(
            num_agents=args.agents,
            duration_seconds=args.duration,
            metrics_dir=args.metrics_dir
        )
    else:
        metrics_dir = args.metrics_dir
        print(f"Skipping simulation, analyzing existing results in {metrics_dir}")
    
    analyze_results(metrics_dir)
    
    print("\nExample completed successfully!")

if __name__ == "__main__":
    main()
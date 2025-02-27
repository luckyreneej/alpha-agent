#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Communication Metrics Module

This module tracks and analyzes inter-agent communication patterns and effectiveness.
"""

import os
import json
import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import logging
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CommunicationMetrics:
    """
    Tracks and analyzes communication metrics between agents.
    """
    
    def __init__(self, metrics_dir: str = None):
        """
        Initialize the communication metrics tracker.
        
        Args:
            metrics_dir: Directory to store metrics data
        """
        self.metrics_dir = metrics_dir or os.path.join('data', 'metrics', 'communication')
        os.makedirs(self.metrics_dir, exist_ok=True)
        
        # Initialize metrics storage
        self.message_counts = {}
        self.response_times = {}
        self.message_sizes = {}
        self.communication_graph = nx.DiGraph()
        self.message_history = []
    
    def record_message(self, sender_id: str, receiver_id: str, message_type: str, 
                      timestamp: datetime = None, size_bytes: int = None, 
                      request_id: str = None, content_summary: str = None) -> None:
        """
        Record a message between agents.
        
        Args:
            sender_id: ID of the sending agent
            receiver_id: ID of the receiving agent
            message_type: Type of message
            timestamp: Message timestamp
            size_bytes: Size of message in bytes
            request_id: ID for request-response tracking
            content_summary: Brief summary of message content
        """
        timestamp = timestamp or datetime.now()
        
        # Track message counts
        agent_pair = f"{sender_id}:{receiver_id}"
        if agent_pair not in self.message_counts:
            self.message_counts[agent_pair] = {}
        
        if message_type not in self.message_counts[agent_pair]:
            self.message_counts[agent_pair][message_type] = 0
        
        self.message_counts[agent_pair][message_type] += 1
        
        # Track message sizes if provided
        if size_bytes is not None:
            if agent_pair not in self.message_sizes:
                self.message_sizes[agent_pair] = []
            
            self.message_sizes[agent_pair].append(size_bytes)
        
        # Update network graph
        if not self.communication_graph.has_edge(sender_id, receiver_id):
            self.communication_graph.add_edge(sender_id, receiver_id, weight=0, 
                                           messages={}, response_times=[])
        
        # Get the edge data
        edge_data = self.communication_graph.get_edge_data(sender_id, receiver_id)
        
        # Update edge weight (message count)
        edge_data['weight'] += 1
        
        # Update message type counts
        if message_type not in edge_data['messages']:
            edge_data['messages'][message_type] = 0
        edge_data['messages'][message_type] += 1
        
        # Add to message history
        message_record = {
            'sender_id': sender_id,
            'receiver_id': receiver_id,
            'message_type': message_type,
            'timestamp': timestamp,
            'size_bytes': size_bytes,
            'request_id': request_id,
            'content_summary': content_summary
        }
        self.message_history.append(message_record)
        
        # Save metrics periodically
        self._check_auto_save()
    
    def record_response_time(self, sender_id: str, receiver_id: str, 
                           request_id: str, response_time_ms: float) -> None:
        """
        Record response time for request-response patterns.
        
        Args:
            sender_id: ID of the requesting agent
            receiver_id: ID of the responding agent
            request_id: ID linking request and response
            response_time_ms: Response time in milliseconds
        """
        # Track response times by agent pair
        agent_pair = f"{sender_id}:{receiver_id}"
        if agent_pair not in self.response_times:
            self.response_times[agent_pair] = []
        
        self.response_times[agent_pair].append(response_time_ms)
        
        # Update network graph
        if self.communication_graph.has_edge(sender_id, receiver_id):
            edge_data = self.communication_graph.get_edge_data(sender_id, receiver_id)
            edge_data['response_times'].append(response_time_ms)
        
        # Save metrics periodically
        self._check_auto_save()
    
    def get_message_stats(self, start_time: Optional[datetime] = None, 
                         end_time: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Get message statistics for the specified time period.
        
        Args:
            start_time: Optional start time filter
            end_time: Optional end time filter
            
        Returns:
            Dictionary with message statistics
        """
        # Filter message history by time if specified
        filtered_history = self.message_history
        if start_time or end_time:
            start_time = start_time or datetime.min
            end_time = end_time or datetime.max
            
            filtered_history = [
                msg for msg in self.message_history 
                if start_time <= msg['timestamp'] <= end_time
            ]
        
        if not filtered_history:
            return {
                'total_messages': 0,
                'active_agents': 0,
                'avg_response_time': 0.0,
                'message_types': {}
            }
        
        # Calculate overall statistics
        unique_senders = set(msg['sender_id'] for msg in filtered_history)
        unique_receivers = set(msg['receiver_id'] for msg in filtered_history)
        unique_agents = unique_senders.union(unique_receivers)
        
        # Count message types
        message_types = {}
        for msg in filtered_history:
            msg_type = msg['message_type']
            if msg_type not in message_types:
                message_types[msg_type] = 0
            message_types[msg_type] += 1
        
        # Calculate response times if available
        valid_response_times = []
        for agent_pair, times in self.response_times.items():
            valid_response_times.extend(times)
        
        avg_response_time = np.mean(valid_response_times) if valid_response_times else 0.0
        
        return {
            'total_messages': len(filtered_history),
            'active_agents': len(unique_agents),
            'unique_senders': len(unique_senders),
            'unique_receivers': len(unique_receivers),
            'avg_response_time': avg_response_time,
            'message_types': message_types,
            'time_period': {
                'start': start_time,
                'end': end_time
            }
        }
    
    def calculate_agent_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Calculate communication metrics for individual agents.
        
        Returns:
            Dictionary mapping agent IDs to their metrics
        """
        agent_metrics = {}
        
        # Get all unique agents from the graph
        all_agents = set(self.communication_graph.nodes())
        
        for agent_id in all_agents:
            # Calculate outgoing metrics (messages sent)
            out_edges = self.communication_graph.out_edges(agent_id, data=True)
            out_message_count = sum(data['weight'] for _, _, data in out_edges)
            
            out_response_times = []
            for _, _, data in out_edges:
                out_response_times.extend(data.get('response_times', []))
            
            # Calculate incoming metrics (messages received)
            in_edges = self.communication_graph.in_edges(agent_id, data=True)
            in_message_count = sum(data['weight'] for _, _, data in in_edges)
            
            in_response_times = []
            for _, _, data in in_edges:
                in_response_times.extend(data.get('response_times', []))
            
            # Calculate various metrics
            metrics = {
                'messages_sent': out_message_count,
                'messages_received': in_message_count,
                'total_messages': out_message_count + in_message_count,
                'outgoing_connections': len(out_edges),
                'incoming_connections': len(in_edges),
                'avg_outgoing_response_time': np.mean(out_response_times) if out_response_times else 0.0,
                'avg_incoming_response_time': np.mean(in_response_times) if in_response_times else 0.0
            }
            
            agent_metrics[agent_id] = metrics
        
        return agent_metrics
    
    def identify_bottlenecks(self, threshold_factor: float = 2.0) -> List[Dict[str, Any]]:
        """
        Identify communication bottlenecks using response time analysis.
        
        Args:
            threshold_factor: Factor above average to consider as bottleneck
            
        Returns:
            List of identified bottlenecks
        """
        bottlenecks = []
        
        # Calculate overall average response time
        all_response_times = []
        for times in self.response_times.values():
            all_response_times.extend(times)
        
        if not all_response_times:
            return bottlenecks
        
        overall_avg = np.mean(all_response_times)
        threshold = overall_avg * threshold_factor
        
        # Check for agent pairs with high response times
        for agent_pair, times in self.response_times.items():
            if not times:
                continue
                
            avg_time = np.mean(times)
            if avg_time > threshold:
                sender, receiver = agent_pair.split(':')
                
                bottlenecks.append({
                    'sender_id': sender,
                    'receiver_id': receiver,
                    'avg_response_time': avg_time,
                    'overall_avg_time': overall_avg,
                    'ratio': avg_time / overall_avg,
                    'message_count': len(times)
                })
        
        # Sort by response time ratio (highest first)
        bottlenecks.sort(key=lambda x: x['ratio'], reverse=True)
        
        return bottlenecks
    
    def analyze_information_flow(self) -> Dict[str, Any]:
        """
        Analyze information flow patterns in the communication network.
        
        Returns:
            Dictionary with flow analysis metrics
        """
        if not self.communication_graph.nodes():
            return {
                'centrality': {},
                'hubs': [],
                'isolated_agents': []
            }
        
        # Calculate network metrics
        try:
            # Degree centrality measures
            out_degree_centrality = nx.out_degree_centrality(self.communication_graph)
            in_degree_centrality = nx.in_degree_centrality(self.communication_graph)
            
            # Betweenness centrality (how often an agent is on the shortest path)
            betweenness_centrality = nx.betweenness_centrality(self.communication_graph, weight='weight')
            
            # Identify hubs (agents with high out-degree)
            hubs = []
            if out_degree_centrality:
                avg_out_degree = np.mean(list(out_degree_centrality.values()))
                hubs = [{
                    'agent_id': agent_id,
                    'centrality': centrality,
                    'ratio_to_avg': centrality / avg_out_degree if avg_out_degree > 0 else float('inf')
                } for agent_id, centrality in out_degree_centrality.items() 
                  if centrality > avg_out_degree * 1.5]  # 50% more connections than average
            
            # Identify isolated agents (no incoming or outgoing connections)
            isolated_agents = [node for node in self.communication_graph.nodes() 
                             if self.communication_graph.degree(node) == 0]
            
            return {
                'centrality': {
                    'out_degree': out_degree_centrality,
                    'in_degree': in_degree_centrality,
                    'betweenness': betweenness_centrality
                },
                'hubs': sorted(hubs, key=lambda x: x['centrality'], reverse=True),
                'isolated_agents': isolated_agents
            }
            
        except Exception as e:
            logger.error(f"Error analyzing information flow: {e}")
            return {
                'centrality': {},
                'hubs': [],
                'isolated_agents': [],
                'error': str(e)
            }
    
    def plot_communication_network(self, output_file: str = None) -> str:
        """
        Generate a visualization of the communication network.
        
        Args:
            output_file: Optional output file path
            
        Returns:
            Path to the saved visualization file
        """
        if not self.communication_graph.nodes():
            logger.warning("No communication data available for visualization")
            return ""
        
        # Create the plot
        plt.figure(figsize=(12, 10))
        
        # Calculate node sizes based on activity
        node_sizes = {}
        for node in self.communication_graph.nodes():
            # Size based on total messages sent and received
            sent = sum(data['weight'] for _, _, data in self.communication_graph.out_edges(node, data=True))
            received = sum(data['weight'] for _, _, data in self.communication_graph.in_edges(node, data=True))
            node_sizes[node] = 100 + (sent + received) * 10  # Base size + scaling factor
        
        # Calculate edge weights
        edge_weights = [data['weight'] for _, _, data in self.communication_graph.edges(data=True)]
        max_weight = max(edge_weights) if edge_weights else 1
        normalized_weights = [w / max_weight * 5 for w in edge_weights]  # Scale for visibility
        
        # Calculate network layout
        pos = nx.spring_layout(self.communication_graph)
        
        # Draw nodes
        nx.draw_networkx_nodes(
            self.communication_graph, pos,
            node_size=[node_sizes.get(node, 100) for node in self.communication_graph.nodes()],
            node_color='lightblue',
            alpha=0.8
        )
        
        # Draw edges with varying thickness
        nx.draw_networkx_edges(
            self.communication_graph, pos,
            width=normalized_weights,
            edge_color='gray',
            alpha=0.6,
            arrows=True,
            arrowsize=15,
            arrowstyle='-|>'
        )
        
        # Add labels
        nx.draw_networkx_labels(self.communication_graph, pos, font_size=10)
        
        plt.title('Agent Communication Network')
        plt.axis('off')
        
        # Save the plot
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = os.path.join(self.metrics_dir, f"communication_network_{timestamp}.png")
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Communication network visualization saved to {output_file}")
        return output_file
    
    def save_metrics(self) -> str:
        """
        Save current metrics to a file.
        
        Returns:
            Path to the saved file
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_path = os.path.join(self.metrics_dir, f"communication_metrics_{timestamp}.json")
        
        # Prepare data for serialization
        data = {
            'timestamp': datetime.now().isoformat(),
            'message_counts': self.message_counts,
            'agent_metrics': self.calculate_agent_metrics(),
            'bottlenecks': self.identify_bottlenecks(),
            'network_info': {
                'node_count': len(self.communication_graph.nodes()),
                'edge_count': len(self.communication_graph.edges())
            },
            'message_stats': self.get_message_stats()
        }
        
        # Information flow analysis can fail if the graph doesn't meet requirements
        try:
            data['information_flow'] = self.analyze_information_flow()
        except Exception as e:
            logger.error(f"Error including information flow analysis: {e}")
            data['information_flow'] = {'error': str(e)}
        
        # Convert non-serializable objects
        serializable_history = []
        for msg in self.message_history:
            msg_copy = msg.copy()
            if isinstance(msg_copy.get('timestamp'), datetime):
                msg_copy['timestamp'] = msg_copy['timestamp'].isoformat()
            serializable_history.append(msg_copy)
        
        data['message_history'] = serializable_history
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Communication metrics saved to {file_path}")
        return file_path
    
    def load_metrics(self, file_path: str) -> bool:
        """
        Load metrics from a file.
        
        Args:
            file_path: Path to the metrics file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            self.message_counts = data.get('message_counts', {})
            
            # Recreate the graph
            self.communication_graph = nx.DiGraph()
            
            # Add nodes and edges based on message history
            message_history = []
            for msg in data.get('message_history', []):
                # Convert timestamp string back to datetime
                if isinstance(msg.get('timestamp'), str):
                    try:
                        msg['timestamp'] = datetime.fromisoformat(msg['timestamp'])
                    except ValueError:
                        msg['timestamp'] = datetime.now()  # Fallback
                
                message_history.append(msg)
                
                # Add to graph
                sender = msg.get('sender_id')
                receiver = msg.get('receiver_id')
                
                if sender and receiver:
                    if not self.communication_graph.has_edge(sender, receiver):
                        self.communication_graph.add_edge(sender, receiver, weight=0, 
                                                       messages={}, response_times=[])
                    
                    # Update edge weight
                    edge_data = self.communication_graph.get_edge_data(sender, receiver)
                    edge_data['weight'] += 1
                    
                    # Update message type counts
                    msg_type = msg.get('message_type', 'unknown')
                    if msg_type not in edge_data['messages']:
                        edge_data['messages'][msg_type] = 0
                    edge_data['messages'][msg_type] += 1
            
            self.message_history = message_history
            
            logger.info(f"Successfully loaded communication metrics from {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading communication metrics: {e}")
            return False
    
    def _check_auto_save(self) -> None:
        """
        Auto-save metrics if the message count reaches a threshold.
        """
        # Save every 1000 messages
        if len(self.message_history) % 1000 == 0 and len(self.message_history) > 0:
            self.save_metrics()
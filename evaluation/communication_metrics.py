#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Optimized Communication Metrics Module

This module tracks and analyzes inter-agent communication patterns and effectiveness
with improved memory efficiency and performance.
"""

import os
import json
import numpy as np
import networkx as nx
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
from collections import defaultdict, Counter

# Import utility modules
from metrics_utils import sample_time_series
from data_cache import DataCache
from visualization import plot_communication_network

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CommunicationMetrics:
    """
    Tracks and analyzes communication metrics between agents with optimized
    storage and algorithms.
    """

    def __init__(self, metrics_dir: str = None, max_history: int = 10000):
        """
        Initialize the communication metrics tracker.

        Args:
            metrics_dir: Directory to store metrics data
            max_history: Maximum number of messages to keep in memory
        """
        self.metrics_dir = metrics_dir or os.path.join('data', 'metrics', 'communication')
        os.makedirs(self.metrics_dir, exist_ok=True)

        # Initialize data cache
        self.data_cache = DataCache(self.metrics_dir)

        # Initialize metrics storage with more efficient data structures
        self.message_counts = defaultdict(Counter)  # {agent_pair: {message_type: count}}
        self.response_times = defaultdict(list)  # {agent_pair: [response_times]}
        self.message_sizes = defaultdict(list)  # {agent_pair: [message_sizes]}

        # Initialize graph with metadata for efficiency
        self.communication_graph = nx.DiGraph()

        # Keep message history with max size
        self.message_history = []
        self.max_history = max_history

        # Track last save timestamp
        self.last_save_time = datetime.now()
        self._auto_save_interval = 300  # 5 minutes

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

        # Track message counts more efficiently
        agent_pair = f"{sender_id}:{receiver_id}"
        self.message_counts[agent_pair][message_type] += 1

        # Track message sizes if provided
        if size_bytes is not None:
            self.message_sizes[agent_pair].append(size_bytes)
            # Limit list size to prevent memory issues
            if len(self.message_sizes[agent_pair]) > self.max_history:
                self.message_sizes[agent_pair] = self.message_sizes[agent_pair][-self.max_history:]

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

        # Add to message history with memory constraints
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

        # Limit message history size
        if len(self.message_history) > self.max_history:
            # Keep most recent messages
            self.message_history = self.message_history[-self.max_history:]

        # Check for auto-save
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
        self.response_times[agent_pair].append(response_time_ms)

        # Limit list size to prevent memory issues
        if len(self.response_times[agent_pair]) > self.max_history:
            self.response_times[agent_pair] = self.response_times[agent_pair][-self.max_history:]

        # Update network graph
        if self.communication_graph.has_edge(sender_id, receiver_id):
            edge_data = self.communication_graph.get_edge_data(sender_id, receiver_id)
            edge_data['response_times'].append(response_time_ms)

            # Limit response times in graph
            if len(edge_data['response_times']) > 1000:  # Keep fewer in graph for memory efficiency
                edge_data['response_times'] = edge_data['response_times'][-1000:]

        # Check for auto-save
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
                if (isinstance(msg['timestamp'], datetime) and
                    start_time <= msg['timestamp'] <= end_time)
            ]

        if not filtered_history:
            return {
                'total_messages': 0,
                'active_agents': 0,
                'avg_response_time': 0.0,
                'message_types': {}
            }

        # Calculate overall statistics efficiently
        sender_set = set()
        receiver_set = set()
        message_type_counter = Counter()

        for msg in filtered_history:
            sender_set.add(msg['sender_id'])
            receiver_set.add(msg['receiver_id'])
            message_type_counter[msg['message_type']] += 1

        unique_agents = sender_set.union(receiver_set)

        # Calculate response times if available
        valid_response_times = []
        for agent_pair, times in self.response_times.items():
            if times:  # Only include if we have data
                valid_response_times.extend(times)

        avg_response_time = np.mean(valid_response_times) if valid_response_times else 0.0

        return {
            'total_messages': len(filtered_history),
            'active_agents': len(unique_agents),
            'unique_senders': len(sender_set),
            'unique_receivers': len(receiver_set),
            'avg_response_time': avg_response_time,
            'message_types': dict(message_type_counter),
            'time_period': {
                'start': start_time.isoformat() if isinstance(start_time, datetime) else str(start_time),
                'end': end_time.isoformat() if isinstance(end_time, datetime) else str(end_time)
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
            # Calculate outgoing metrics (messages sent) efficiently
            out_edges = list(self.communication_graph.out_edges(agent_id, data=True))
            out_message_count = sum(data['weight'] for _, _, data in out_edges)

            # Extract response times in one pass
            out_response_times = []
            for _, _, data in out_edges:
                if 'response_times' in data:
                    out_response_times.extend(data['response_times'])

            # Calculate incoming metrics (messages received) efficiently
            in_edges = list(self.communication_graph.in_edges(agent_id, data=True))
            in_message_count = sum(data['weight'] for _, _, data in in_edges)

            # Extract incoming response times in one pass
            in_response_times = []
            for _, _, data in in_edges:
                if 'response_times' in data:
                    in_response_times.extend(data['response_times'])

            # Calculate metrics
            metrics = {
                'messages_sent': out_message_count,
                'messages_received': in_message_count,
                'total_messages': out_message_count + in_message_count,
                'outgoing_connections': len(out_edges),
                'incoming_connections': len(in_edges),
            }

            # Add response time metrics if available
            if out_response_times:
                metrics['avg_outgoing_response_time'] = float(np.mean(out_response_times))

            if in_response_times:
                metrics['avg_incoming_response_time'] = float(np.mean(in_response_times))

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

        # Collect all response times in one pass
        all_response_times = []
        for times in self.response_times.values():
            if times:  # Only include if we have data
                all_response_times.extend(times)

        if not all_response_times:
            return bottlenecks

        # Calculate overall average
        overall_avg = np.mean(all_response_times)
        threshold = overall_avg * threshold_factor

        # Check each agent pair
        for agent_pair, times in self.response_times.items():
            if not times:
                continue

            avg_time = np.mean(times)
            if avg_time > threshold:
                sender, receiver = agent_pair.split(':')

                bottlenecks.append({
                    'sender_id': sender,
                    'receiver_id': receiver,
                    'avg_response_time': float(avg_time),
                    'overall_avg_time': float(overall_avg),
                    'ratio': float(avg_time / overall_avg),
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

        # Calculate network metrics efficiently
        try:
            # Degree centrality
            out_degree = dict(self.communication_graph.out_degree(weight='weight'))
            in_degree = dict(self.communication_graph.in_degree(weight='weight'))

            # Convert to centrality (normalized by n-1)
            n = len(self.communication_graph)
            norm_factor = n - 1 if n > 1 else 1  # Avoid division by zero

            out_degree_centrality = {node: count / norm_factor for node, count in out_degree.items()}
            in_degree_centrality = {node: count / norm_factor for node, count in in_degree.items()}

            # Betweenness centrality (how often an agent is on the shortest path)
            # Only calculate if graph is not too large (expensive calculation)
            if n <= 100:  # Arbitrary threshold
                betweenness_centrality = nx.betweenness_centrality(self.communication_graph, weight='weight')
            else:
                # For large graphs, use approximate or skip
                betweenness_centrality = {}
                logger.info("Graph too large for betweenness centrality calculation")

            # Identify hubs (agents with high out-degree)
            hubs = []
            if out_degree_centrality:
                avg_out_degree = np.mean(list(out_degree_centrality.values()))

                # Consider agents with significantly higher than average out-degree
                for agent_id, centrality in out_degree_centrality.items():
                    if centrality > avg_out_degree * 1.5:  # 50% more connections than average
                        hubs.append({
                            'agent_id': agent_id,
                            'centrality': float(centrality),
                            'ratio_to_avg': float(centrality / avg_out_degree) if avg_out_degree > 0 else float('inf')
                        })

            # Sort hubs by centrality
            hubs.sort(key=lambda x: x['centrality'], reverse=True)

            # Identify isolated agents (no incoming or outgoing connections)
            isolated_agents = [node for node in self.communication_graph.nodes()
                               if self.communication_graph.degree(node) == 0]

            return {
                'centrality': {
                    'out_degree': out_degree_centrality,
                    'in_degree': in_degree_centrality,
                    'betweenness': betweenness_centrality
                },
                'hubs': hubs,
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

        # Use the visualization utility to create the plot
        return plot_communication_network(
            self.communication_graph,
            node_size_attr='weight',
            edge_weight_attr='weight',
            output_file=output_file,
            title="Agent Communication Network"
        )

    def save_metrics(self, file_path: str = None) -> str:
        """
        Save current metrics to a file.

        Args:
            file_path: Optional path for the saved file

        Returns:
            Path to the saved file
        """
        if file_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_path = os.path.join(self.metrics_dir, f"communication_metrics_{timestamp}.json")

        # Prepare serializable data
        data = {
            'timestamp': datetime.now().isoformat(),
            'message_counts': {k: dict(v) for k, v in self.message_counts.items()},
            'agent_metrics': self.calculate_agent_metrics(),
            'bottlenecks': self.identify_bottlenecks(),
            'network_info': {
                'node_count': len(self.communication_graph.nodes()),
                'edge_count': len(self.communication_graph.edges())
            },
            'message_stats': self.get_message_stats()
        }

        # Add information flow analysis
        try:
            data['information_flow'] = self.analyze_information_flow()
        except Exception as e:
            logger.error(f"Error including information flow analysis: {e}")
            data['information_flow'] = {'error': str(e)}

        # Convert non-serializable objects for message history
        serializable_history = []

        # Sample message history if too large
        history_to_save = sample_time_series(self.message_history, 1000) if len(
            self.message_history) > 1000 else self.message_history

        for msg in history_to_save:
            msg_copy = msg.copy()
            if isinstance(msg_copy.get('timestamp'), datetime):
                msg_copy['timestamp'] = msg_copy['timestamp'].isoformat()
            serializable_history.append(msg_copy)

        data['message_history'] = serializable_history

        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Save to file
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Communication metrics saved to {file_path}")
        self.last_save_time = datetime.now()

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

            # Clear existing data
            self.message_counts = defaultdict(Counter)
            self.response_times = defaultdict(list)
            self.message_sizes = defaultdict(list)
            self.communication_graph = nx.DiGraph()
            self.message_history = []

            # Load message counts
            for k, v in data.get('message_counts', {}).items():
                self.message_counts[k] = Counter(v)

            # Rebuild message history and graph
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
                message_type = msg.get('message_type', 'unknown')

                if sender and receiver:
                    if not self.communication_graph.has_edge(sender, receiver):
                        self.communication_graph.add_edge(sender, receiver, weight=0,
                                                          messages={}, response_times=[])

                    # Update edge weight
                    edge_data = self.communication_graph.get_edge_data(sender, receiver)
                    edge_data['weight'] += 1

                    # Update message type counts
                    if message_type not in edge_data['messages']:
                        edge_data['messages'][message_type] = 0
                    edge_data['messages'][message_type] += 1

            # Limit history size
            if len(message_history) > self.max_history:
                message_history = message_history[-self.max_history:]

            self.message_history = message_history

            logger.info(f"Successfully loaded communication metrics from {file_path}")
            return True

        except Exception as e:
            logger.error(f"Error loading communication metrics: {e}")
            return False

    def _check_auto_save(self) -> None:
        """
        Auto-save metrics if needed.
        """
        # Save every 5 minutes (defined by self._auto_save_interval)
        if (datetime.now() - self.last_save_time).total_seconds() > self._auto_save_interval:
            try:
                self.save_metrics()
            except Exception as e:
                logger.error(f"Error during auto-save: {e}")

    def get_communication_summary(self, start_time: Optional[datetime] = None,
                                  end_time: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Get a summary of communication patterns.

        Args:
            start_time: Optional start time filter
            end_time: Optional end time filter

        Returns:
            Dictionary with communication summary
        """
        # Get basic message stats
        message_stats = self.get_message_stats(start_time, end_time)

        # Get agent metrics
        agent_metrics = self.calculate_agent_metrics()

        # Identify most active agents
        active_agents = []
        for agent_id, metrics in agent_metrics.items():
            active_agents.append({
                'agent_id': agent_id,
                'total_messages': metrics.get('total_messages', 0),
                'messages_sent': metrics.get('messages_sent', 0),
                'messages_received': metrics.get('messages_received', 0)
            })

        # Sort by total messages
        active_agents.sort(key=lambda x: x['total_messages'], reverse=True)

        # Get top 5 most active
        top_active_agents = active_agents[:5] if active_agents else []

        # Get bottlenecks
        bottlenecks = self.identify_bottlenecks()

        # Calculate common communication paths
        common_paths = []
        for sender_id, receiver_id, data in self.communication_graph.edges(data=True):
            common_paths.append({
                'sender_id': sender_id,
                'receiver_id': receiver_id,
                'message_count': data.get('weight', 0),
                'message_types': data.get('messages', {})
            })

        # Sort by message count
        common_paths.sort(key=lambda x: x['message_count'], reverse=True)

        # Get top 5 most common paths
        top_common_paths = common_paths[:5] if common_paths else []

        # Create summary
        summary = {
            'time_period': {
                'start': start_time.isoformat() if start_time else None,
                'end': end_time.isoformat() if end_time else None
            },
            'total_messages': message_stats.get('total_messages', 0),
            'active_agents': message_stats.get('active_agents', 0),
            'message_types': message_stats.get('message_types', {}),
            'avg_response_time': message_stats.get('avg_response_time', 0),
            'top_active_agents': top_active_agents,
            'top_communication_paths': top_common_paths,
            'bottlenecks': bottlenecks[:3] if bottlenecks else []
        }

        return summary

    def reset_metrics(self) -> None:
        """
        Reset all metrics data.
        """
        # Save current data before resetting
        self.save_metrics()

        # Clear all data structures
        self.message_counts = defaultdict(Counter)
        self.response_times = defaultdict(list)
        self.message_sizes = defaultdict(list)
        self.communication_graph = nx.DiGraph()
        self.message_history = []

        logger.info("Communication metrics have been reset")

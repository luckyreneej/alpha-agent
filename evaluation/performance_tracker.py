#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Performance Tracker Module

This module provides functionality for tracking and analyzing agent performance over time,
including historical performance logging, time-based comparisons, and contribution analysis.
"""

import os
import json
import pandas as pd
import numpy as np
import glob
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from datetime import datetime, timedelta
import logging
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PerformanceTracker:
    """
    Tracks and analyzes agent performance over time.
    """
    
    def __init__(self, data_dir: str = None):
        """
        Initialize the performance tracker.
        
        Args:
            data_dir: Directory containing performance data files
        """
        self.data_dir = data_dir or os.path.join('data', 'metrics')
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Cache for loaded data
        self._performance_cache = {}
    
    def load_agent_history(self, agent_id: str, start_date: Optional[datetime] = None, 
                           end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Load historical performance data for an agent.
        
        Args:
            agent_id: Agent identifier
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            DataFrame with historical performance metrics
        """
        # Define date filters
        if start_date is None:
            start_date = datetime.min
        if end_date is None:
            end_date = datetime.max
            
        # Cache key for this query
        cache_key = f"{agent_id}_{start_date.isoformat()}_{end_date.isoformat()}"
        if cache_key in self._performance_cache:
            return self._performance_cache[cache_key]
        
        # Find all metric files for this agent
        file_pattern = os.path.join(self.data_dir, f"{agent_id}_*.json")
        metric_files = glob.glob(file_pattern)
        
        if not metric_files:
            logger.warning(f"No metric files found for agent {agent_id}")
            return pd.DataFrame()
        
        # Load and combine data
        data_list = []
        for file_path in metric_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Parse timestamp
                timestamp = datetime.fromisoformat(data['timestamp'])
                
                # Skip if outside date range
                if timestamp < start_date or timestamp > end_date:
                    continue
                
                # Extract metrics and flatten for DataFrame
                flattened = {'timestamp': timestamp}
                flattened['agent_id'] = data['agent_id']
                flattened['agent_type'] = data['agent_type']
                
                # Flatten metrics structure
                metrics = data['metrics']
                for category, category_metrics in metrics.items():
                    for metric_name, metric_value in category_metrics.items():
                        flattened[f"{category}_{metric_name}"] = metric_value
                
                data_list.append(flattened)
                
            except Exception as e:
                logger.error(f"Error loading metrics from {file_path}: {e}")
        
        # Create DataFrame
        if not data_list:
            logger.warning(f"No data found for agent {agent_id} in specified date range")
            return pd.DataFrame()
            
        df = pd.DataFrame(data_list)
        df = df.sort_values('timestamp')
        
        # Cache the result
        self._performance_cache[cache_key] = df
        
        return df
    
    def load_system_history(self, start_date: Optional[datetime] = None, 
                           end_date: Optional[datetime] = None) -> Dict[str, pd.DataFrame]:
        """
        Load historical performance data for all agents in the system.
        
        Args:
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            Dictionary mapping agent IDs to performance DataFrames
        """
        # Find all unique agent IDs from metric files
        file_pattern = os.path.join(self.data_dir, "*.json")
        metric_files = glob.glob(file_pattern)
        
        # Extract unique agent IDs
        agent_ids = set()
        for file_path in metric_files:
            file_name = os.path.basename(file_path)
            if file_name.startswith("system_report"):
                continue
                
            parts = file_name.split('_')
            if len(parts) >= 1:
                agent_id = parts[0]  # Assuming filename format is agent_id_date.json
                agent_ids.add(agent_id)
        
        # Load data for each agent
        system_data = {}
        for agent_id in agent_ids:
            df = self.load_agent_history(agent_id, start_date, end_date)
            if not df.empty:
                system_data[agent_id] = df
        
        return system_data
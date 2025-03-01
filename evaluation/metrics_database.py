import sqlite3
import pandas as pd
import json
import os
import logging
from datetime import datetime
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MetricsDatabase:
    """
    Database backend for storing and retrieving agent metrics.
    """

    def __init__(self, db_path):
        """
        Initialize the metrics database.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._initialize_db()

    def _initialize_db(self):
        """Create database tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create tables
        # Agent metrics table - for numeric metrics
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS agent_metrics (
            id INTEGER PRIMARY KEY,
            agent_id TEXT,
            agent_type TEXT,
            timestamp TEXT,
            metric_type TEXT,
            metric_name TEXT,
            metric_value REAL,
            UNIQUE(agent_id, timestamp, metric_type, metric_name)
        )
        ''')

        # Agent metadata table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS agent_metadata (
            agent_id TEXT PRIMARY KEY,
            agent_type TEXT,
            created_at TEXT,
            last_updated TEXT
        )
        ''')

        # Communication metrics table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS communication_metrics (
            id INTEGER PRIMARY KEY,
            sender_id TEXT,
            receiver_id TEXT,
            message_type TEXT,
            timestamp TEXT,
            response_time REAL
        )
        ''')

        # System metrics table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS system_metrics (
            id INTEGER PRIMARY KEY,
            timestamp TEXT,
            metric_name TEXT,
            metric_value REAL,
            UNIQUE(timestamp, metric_name)
        )
        ''')

        # Complex metrics table (for JSON storage)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS complex_metrics (
            id INTEGER PRIMARY KEY,
            agent_id TEXT,
            timestamp TEXT,
            metric_type TEXT,
            metric_data TEXT,  -- JSON data
            UNIQUE(agent_id, timestamp, metric_type)
        )
        ''')

        # Create indices for performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_agent_metrics_agent_id ON agent_metrics(agent_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_agent_metrics_timestamp ON agent_metrics(timestamp)')
        cursor.execute(
            'CREATE INDEX IF NOT EXISTS idx_communication_metrics_sender ON communication_metrics(sender_id)')
        cursor.execute(
            'CREATE INDEX IF NOT EXISTS idx_communication_metrics_timestamp ON communication_metrics(timestamp)')

        conn.commit()
        conn.close()

    def store_agent_metrics(self, agent_id, agent_type, metrics_dict, timestamp=None):
        """
        Store metrics for an agent.

        Args:
            agent_id: Agent identifier
            agent_type: Type of agent
            metrics_dict: Dictionary of metrics organized by type
            timestamp: Optional timestamp (defaults to current time)

        Returns:
            Number of metrics stored
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Update agent metadata
        cursor.execute(
            "INSERT OR REPLACE INTO agent_metadata (agent_id, agent_type, last_updated) VALUES (?, ?, ?)",
            (agent_id, agent_type, timestamp)
        )

        # Store metrics
        count = 0
        for metric_type, metrics in metrics_dict.items():
            # Handle complex metrics (non-scalar values)
            if any(not isinstance(v, (int, float)) for v in metrics.values()):
                # Store as JSON in complex_metrics table
                cursor.execute(
                    "INSERT OR REPLACE INTO complex_metrics (agent_id, timestamp, metric_type, metric_data) VALUES (?, ?, ?, ?)",
                    (agent_id, timestamp, metric_type, json.dumps(metrics))
                )
                count += 1
                continue

            # Store scalar metrics individually
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    try:
                        cursor.execute(
                            """INSERT OR REPLACE INTO agent_metrics 
                               (agent_id, agent_type, timestamp, metric_type, metric_name, metric_value) 
                               VALUES (?, ?, ?, ?, ?, ?)""",
                            (agent_id, agent_type, timestamp, metric_type, metric_name, value)
                        )
                        count += 1
                    except Exception as e:
                        logger.error(f"Error storing metric {metric_name}: {e}")

        conn.commit()
        conn.close()

        logger.info(f"Stored {count} metrics for agent {agent_id}")
        return count

    def store_system_metrics(self, metrics_dict, timestamp=None):
        """
        Store system-wide metrics.

        Args:
            metrics_dict: Dictionary of system metrics
            timestamp: Optional timestamp (defaults to current time)

        Returns:
            Number of metrics stored
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        count = 0
        for metric_name, value in metrics_dict.items():
            if isinstance(value, (int, float)):
                try:
                    cursor.execute(
                        "INSERT OR REPLACE INTO system_metrics (timestamp, metric_name, metric_value) VALUES (?, ?, ?)",
                        (timestamp, metric_name, value)
                    )
                    count += 1
                except Exception as e:
                    logger.error(f"Error storing system metric {metric_name}: {e}")

        conn.commit()
        conn.close()

        logger.info(f"Stored {count} system metrics")
        return count

    def store_communication_metrics(self, sender_id, receiver_id, message_type, response_time=None, timestamp=None):
        """
        Store communication metrics between agents.

        Args:
            sender_id: ID of sending agent
            receiver_id: ID of receiving agent
            message_type: Type of message
            response_time: Optional response time in milliseconds
            timestamp: Optional timestamp (defaults to current time)

        Returns:
            True if successful
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute(
                """INSERT INTO communication_metrics 
                   (sender_id, receiver_id, message_type, timestamp, response_time) 
                   VALUES (?, ?, ?, ?, ?)""",
                (sender_id, receiver_id, message_type, timestamp, response_time)
            )

            conn.commit()
            conn.close()
            return True

        except Exception as e:
            logger.error(f"Error storing communication metric: {e}")
            conn.close()
            return False

    def get_agent_metrics(self, agent_id, metric_types=None, start_time=None, end_time=None):
        """
        Retrieve metrics for an agent.

        Args:
            agent_id: Agent identifier
            metric_types: Optional list of metric types to include
            start_time: Optional start time filter
            end_time: Optional end time filter

        Returns:
            DataFrame with metrics data
        """
        conn = sqlite3.connect(self.db_path)

        query = "SELECT timestamp, metric_type, metric_name, metric_value FROM agent_metrics WHERE agent_id = ?"
        params = [agent_id]

        if metric_types:
            placeholders = ','.join(['?' for _ in metric_types])
            query += f" AND metric_type IN ({placeholders})"
            params.extend(metric_types)

        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)

        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)

        query += " ORDER BY timestamp"

        try:
            df = pd.read_sql_query(query, conn, params=params)

            # Also get complex metrics
            complex_query = "SELECT timestamp, metric_type, metric_data FROM complex_metrics WHERE agent_id = ?"
            complex_params = [agent_id]

            if metric_types:
                placeholders = ','.join(['?' for _ in metric_types])
                complex_query += f" AND metric_type IN ({placeholders})"
                complex_params.extend(metric_types)

            if start_time:
                complex_query += " AND timestamp >= ?"
                complex_params.append(start_time)

            if end_time:
                complex_query += " AND timestamp <= ?"
                complex_params.append(end_time)

            complex_df = pd.read_sql_query(complex_query, conn, params=complex_params)

            conn.close()

            # Process complex metrics
            if not complex_df.empty:
                # Parse JSON data
                complex_metrics = []
                for _, row in complex_df.iterrows():
                    metrics_data = json.loads(row['metric_data'])
                    for metric_name, value in metrics_data.items():
                        # Only process scalar values
                        if isinstance(value, (int, float)):
                            complex_metrics.append({
                                'timestamp': row['timestamp'],
                                'metric_type': row['metric_type'],
                                'metric_name': metric_name,
                                'metric_value': value
                            })

                if complex_metrics:
                    # Append complex metrics to regular metrics
                    complex_metrics_df = pd.DataFrame(complex_metrics)
                    df = pd.concat([df, complex_metrics_df], ignore_index=True)

            return df

        except Exception as e:
            logger.error(f"Error retrieving agent metrics: {e}")
            conn.close()
            return pd.DataFrame()

    def get_system_metrics(self, metric_names=None, start_time=None, end_time=None):
        """
        Retrieve system metrics.

        Args:
            metric_names: Optional list of metric names to include
            start_time: Optional start time filter
            end_time: Optional end time filter

        Returns:
            DataFrame with system metrics
        """
        conn = sqlite3.connect(self.db_path)

        query = "SELECT timestamp, metric_name, metric_value FROM system_metrics"
        params = []

        conditions = []
        if metric_names:
            placeholders = ','.join(['?' for _ in metric_names])
            conditions.append(f"metric_name IN ({placeholders})")
            params.extend(metric_names)

        if start_time:
            conditions.append("timestamp >= ?")
            params.append(start_time)

        if end_time:
            conditions.append("timestamp <= ?")
            params.append(end_time)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY timestamp"

        try:
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()

            # Pivot to wide format if there are multiple metric names
            if not df.empty and 'metric_name' in df.columns:
                # Convert timestamp to datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'])

                # Pivot to wide format
                df_wide = df.pivot(index='timestamp', columns='metric_name', values='metric_value')
                df_wide.reset_index(inplace=True)

                return df_wide

            return df

        except Exception as e:
            logger.error(f"Error retrieving system metrics: {e}")
            conn.close()
            return pd.DataFrame()

    def get_communication_metrics(self, agent_id=None, as_sender=True, as_receiver=False,
                                  start_time=None, end_time=None):
        """
        Retrieve communication metrics.

        Args:
            agent_id: Optional agent ID to filter by
            as_sender: Whether to include metrics where agent is sender
            as_receiver: Whether to include metrics where agent is receiver
            start_time: Optional start time filter
            end_time: Optional end time filter

        Returns:
            DataFrame with communication metrics
        """
        conn = sqlite3.connect(self.db_path)

        query = "SELECT sender_id, receiver_id, message_type, timestamp, response_time FROM communication_metrics"
        params = []

        conditions = []
        if agent_id:
            agent_conditions = []
            if as_sender:
                agent_conditions.append("sender_id = ?")
                params.append(agent_id)
            if as_receiver:
                agent_conditions.append("receiver_id = ?")
                params.append(agent_id)

            if agent_conditions:
                conditions.append("(" + " OR ".join(agent_conditions) + ")")

        if start_time:
            conditions.append("timestamp >= ?")
            params.append(start_time)

        if end_time:
            conditions.append("timestamp <= ?")
            params.append(end_time)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY timestamp"

        try:
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()
            return df

        except Exception as e:
            logger.error(f"Error retrieving communication metrics: {e}")
            conn.close()
            return pd.DataFrame()

    def get_agent_list(self):
        """
        Get a list of all agents with their metadata.

        Returns:
            DataFrame with agent metadata
        """
        conn = sqlite3.connect(self.db_path)

        query = "SELECT agent_id, agent_type, created_at, last_updated FROM agent_metadata"

        try:
            df = pd.read_sql_query(query, conn)
            conn.close()
            return df

        except Exception as e:
            logger.error(f"Error retrieving agent list: {e}")
            conn.close()
            return pd.DataFrame()

    def get_metrics_summary(self, start_time=None, end_time=None):
        """
        Get a summary of metrics in the database.

        Args:
            start_time: Optional start time filter
            end_time: Optional end time filter

        Returns:
            Dictionary with metrics summary
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        time_condition = ""
        params = []

        if start_time:
            time_condition += " AND timestamp >= ?"
            params.append(start_time)

        if end_time:
            time_condition += " AND timestamp <= ?"
            params.append(end_time)

        # Count agents
        cursor.execute("SELECT COUNT(DISTINCT agent_id) FROM agent_metadata")
        agent_count = cursor.fetchone()[0]

        # Count metrics by type
        cursor.execute(
            f"SELECT metric_type, COUNT(*) FROM agent_metrics WHERE 1=1 {time_condition} GROUP BY metric_type",
            params
        )
        metric_counts = dict(cursor.fetchall())

        # Count system metrics
        cursor.execute(
            f"SELECT COUNT(*) FROM system_metrics WHERE 1=1 {time_condition}",
            params
        )
        system_metric_count = cursor.fetchone()[0]

        # Count communication metrics
        cursor.execute(
            f"SELECT COUNT(*) FROM communication_metrics WHERE 1=1 {time_condition}",
            params
        )
        comm_metric_count = cursor.fetchone()[0]

        # Get date range
        cursor.execute(
            "SELECT MIN(timestamp), MAX(timestamp) FROM agent_metrics"
        )
        date_range = cursor.fetchone()

        conn.close()

        return {
            'agent_count': agent_count,
            'metric_counts': metric_counts,
            'system_metric_count': system_metric_count,
            'communication_metric_count': comm_metric_count,
            'date_range': {
                'start': date_range[0] if date_range else None,
                'end': date_range[1] if date_range else None
            }
        }

    def export_to_csv(self, output_dir):
        """
        Export database tables to CSV files.

        Args:
            output_dir: Directory to save CSV files

        Returns:
            Dictionary mapping table names to file paths
        """
        os.makedirs(output_dir, exist_ok=True)

        conn = sqlite3.connect(self.db_path)

        # Get list of tables
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]

        # Export each table
        exported_files = {}
        for table in tables:
            try:
                output_file = os.path.join(output_dir, f"{table}.csv")
                df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
                df.to_csv(output_file, index=False)
                exported_files[table] = output_file
                logger.info(f"Exported table {table} to {output_file}")
            except Exception as e:
                logger.error(f"Error exporting table {table}: {e}")

        conn.close()

        return exported_files

    def clear_old_data(self, days_to_keep=30):
        """
        Clear data older than the specified number of days.

        Args:
            days_to_keep: Number of days of data to keep

        Returns:
            Number of rows deleted
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Calculate cutoff date
        cutoff_date = (datetime.now() - pd.Timedelta(days=days_to_keep)).isoformat()

        # Delete old agent metrics
        cursor.execute(
            "DELETE FROM agent_metrics WHERE timestamp < ?",
            (cutoff_date,)
        )
        agent_metrics_deleted = cursor.rowcount

        # Delete old system metrics
        cursor.execute(
            "DELETE FROM system_metrics WHERE timestamp < ?",
            (cutoff_date,)
        )
        system_metrics_deleted = cursor.rowcount

        # Delete old communication metrics
        cursor.execute(
            "DELETE FROM communication_metrics WHERE timestamp < ?",
            (cutoff_date,)
        )
        comm_metrics_deleted = cursor.rowcount

        # Delete old complex metrics
        cursor.execute(
            "DELETE FROM complex_metrics WHERE timestamp < ?",
            (cutoff_date,)
        )
        complex_metrics_deleted = cursor.rowcount

        # Commit changes
        conn.commit()

        # Vacuum the database to reclaim space
        cursor.execute("VACUUM")

        conn.close()

        total_deleted = (agent_metrics_deleted + system_metrics_deleted +
                         comm_metrics_deleted + complex_metrics_deleted)

        logger.info(f"Cleared {total_deleted} old metrics older than {cutoff_date}")

        return total_deleted

    def calculate_agent_statistics(self, agent_id, metric_type, metric_name,
                                   window=None, start_time=None, end_time=None):
        """
        Calculate statistics for a specific agent metric.

        Args:
            agent_id: Agent identifier
            metric_type: Type of metric
            metric_name: Name of metric
            window: Optional rolling window size in days
            start_time: Optional start time filter
            end_time: Optional end time filter

        Returns:
            Dictionary with statistics
        """
        conn = sqlite3.connect(self.db_path)

        query = """
        SELECT timestamp, metric_value 
        FROM agent_metrics 
        WHERE agent_id = ? 
          AND metric_type = ? 
          AND metric_name = ?
        """

        params = [agent_id, metric_type, metric_name]

        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)

        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)

        query += " ORDER BY timestamp"

        try:
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()

            if df.empty:
                return {
                    'count': 0,
                    'mean': None,
                    'std': None,
                    'min': None,
                    'max': None,
                    'latest': None
                }

            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Calculate basic statistics
            stats = {
                'count': len(df),
                'mean': float(df['metric_value'].mean()),
                'std': float(df['metric_value'].std()),
                'min': float(df['metric_value'].min()),
                'max': float(df['metric_value'].max()),
                'latest': float(df['metric_value'].iloc[-1])
            }

            # Calculate rolling statistics if window specified
            if window and len(df) > 1:
                window_days = pd.Timedelta(days=window)
                df = df.set_index('timestamp')

                # Create rolling window by time
                rolling = df['metric_value'].rolling(window_days)

                stats['rolling_mean'] = (
                    rolling.mean().dropna().tolist() if not rolling.mean().empty else []
                )
                stats['rolling_std'] = (
                    rolling.std().dropna().tolist() if not rolling.std().empty else []
                )
                stats['rolling_timestamps'] = (
                    rolling.mean().dropna().index.strftime(
                        '%Y-%m-%d %H:%M:%S').tolist() if not rolling.mean().empty else []
                )

            return stats

        except Exception as e:
            logger.error(f"Error calculating agent statistics: {e}")
            conn.close()
            return {
                'count': 0,
                'error': str(e)
            }

    def compare_agents(self, agent_ids, metric_types=None, start_time=None, end_time=None):
        """
        Compare metrics across multiple agents.

        Args:
            agent_ids: List of agent identifiers
            metric_types: Optional list of metric types to include
            start_time: Optional start time filter
            end_time: Optional end time filter

        Returns:
            Dictionary with comparison data
        """
        if not agent_ids:
            return {'error': 'No agent IDs provided'}

        # Get agent metadata
        agent_metadata = {}
        for agent_id in agent_ids:
            agent_data = self.get_agent_metrics(
                agent_id, metric_types, start_time, end_time
            )

            if not agent_data.empty:
                # Group metrics by type and name
                metrics_by_type = {}
                for _, row in agent_data.iterrows():
                    metric_type = row['metric_type']
                    metric_name = row['metric_name']

                    if metric_type not in metrics_by_type:
                        metrics_by_type[metric_type] = {}

                    if metric_name not in metrics_by_type[metric_type]:
                        metrics_by_type[metric_type][metric_name] = []

                    metrics_by_type[metric_type][metric_name].append(
                        (row['timestamp'], row['metric_value'])
                    )

                # Calculate statistics for each metric
                agent_stats = {}
                for metric_type, metrics in metrics_by_type.items():
                    agent_stats[metric_type] = {}

                    for metric_name, values in metrics.items():
                        timestamps = [t for t, _ in values]
                        metric_values = [v for _, v in values]

                        agent_stats[metric_type][metric_name] = {
                            'count': len(metric_values),
                            'mean': np.mean(metric_values),
                            'std': np.std(metric_values),
                            'min': np.min(metric_values),
                            'max': np.max(metric_values),
                            'latest': metric_values[-1] if metric_values else None
                        }

                agent_metadata[agent_id] = agent_stats

        # Find common metrics across all agents
        common_metrics = {}

        if agent_metadata:
            # Get all metrics from first agent
            first_agent = next(iter(agent_metadata.values()))

            for metric_type, metrics in first_agent.items():
                for metric_name in metrics:
                    # Check if this metric exists for all agents
                    metric_exists = all(
                        metric_type in agent_stats and metric_name in agent_stats[metric_type]
                        for agent_stats in agent_metadata.values()
                    )

                    if metric_exists:
                        if metric_type not in common_metrics:
                            common_metrics[metric_type] = []
                        common_metrics[metric_type].append(metric_name)

        # Build comparison data
        comparison = {
            'agent_metadata': agent_metadata,
            'common_metrics': common_metrics,
            'summary': {}
        }

        # Add summary statistics for common metrics
        for metric_type, metric_names in common_metrics.items():
            comparison['summary'][metric_type] = {}

            for metric_name in metric_names:
                comparison['summary'][metric_type][metric_name] = {
                    'best_agent': None,
                    'worst_agent': None,
                    'agent_rankings': {}
                }

                # Determine if higher is better based on metric name
                higher_is_better = not any(
                    keyword in metric_name.lower()
                    for keyword in ['error', 'latency', 'time', 'failure']
                )

                # Collect latest values for each agent
                agent_values = {}
                for agent_id, agent_stats in agent_metadata.items():
                    if metric_type in agent_stats and metric_name in agent_stats[metric_type]:
                        latest = agent_stats[metric_type][metric_name]['latest']
                        agent_values[agent_id] = latest

                if agent_values:
                    # Sort agents by value
                    sorted_agents = sorted(
                        agent_values.items(),
                        key=lambda x: x[1],
                        reverse=higher_is_better
                    )

                    # Set best and worst
                    comparison['summary'][metric_type][metric_name]['best_agent'] = sorted_agents[0][0]
                    comparison['summary'][metric_type][metric_name]['worst_agent'] = sorted_agents[-1][0]

                    # Set rankings
                    for i, (agent_id, value) in enumerate(sorted_agents):
                        comparison['summary'][metric_type][metric_name]['agent_rankings'][agent_id] = {
                            'rank': i + 1,
                            'value': value
                        }

        return comparison

    def build_network_graph(self, start_time=None, end_time=None):
        """
        Build a network graph of agent communications.

        Args:
            start_time: Optional start time filter
            end_time: Optional end time filter

        Returns:
            Dictionary with network graph data
        """
        conn = sqlite3.connect(self.db_path)

        query = """
        SELECT sender_id, receiver_id, message_type, COUNT(*) as message_count,
               AVG(response_time) as avg_response_time
        FROM communication_metrics
        WHERE 1=1
        """

        params = []

        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)

        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)

        query += " GROUP BY sender_id, receiver_id, message_type"

        try:
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()

            if df.empty:
                return {
                    'nodes': [],
                    'edges': []
                }

            # Build nodes (unique agents)
            nodes = set()
            for sender in df['sender_id'].unique():
                nodes.add(sender)
            for receiver in df['receiver_id'].unique():
                nodes.add(receiver)

            # Get message counts for each agent
            node_data = {}
            for node in nodes:
                # Outgoing messages
                outgoing = df[df['sender_id'] == node]['message_count'].sum()

                # Incoming messages
                incoming = df[df['receiver_id'] == node]['message_count'].sum()

                node_data[node] = {
                    'id': node,
                    'outgoing_messages': int(outgoing),
                    'incoming_messages': int(incoming),
                    'total_messages': int(outgoing + incoming)
                }

            # Build edges
            edges = []
            for _, row in df.iterrows():
                edges.append({
                    'source': row['sender_id'],
                    'target': row['receiver_id'],
                    'message_type': row['message_type'],
                    'message_count': int(row['message_count']),
                    'avg_response_time': float(row['avg_response_time']) if not pd.isna(
                        row['avg_response_time']) else None
                })

            return {
                'nodes': list(node_data.values()),
                'edges': edges
            }

        except Exception as e:
            logger.error(f"Error building network graph: {e}")
            conn.close()
            return {
                'error': str(e),
                'nodes': [],
                'edges': []
            }

    def backup_database(self, backup_path=None):
        """
        Create a backup of the database.

        Args:
            backup_path: Optional backup file path

        Returns:
            Path to the backup file
        """
        if backup_path is None:
            # Create backup path based on current date
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_dir = os.path.dirname(self.db_path)
            backup_file = os.path.basename(self.db_path).split('.')[0]
            backup_path = os.path.join(backup_dir, f"{backup_file}_backup_{timestamp}.db")

        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(backup_path), exist_ok=True)

            # Connect to source database
            source_conn = sqlite3.connect(self.db_path)

            # Create backup
            backup_conn = sqlite3.connect(backup_path)
            source_conn.backup(backup_conn)

            # Close connections
            backup_conn.close()
            source_conn.close()

            logger.info(f"Database backup created at {backup_path}")
            return backup_path

        except Exception as e:
            logger.error(f"Error creating database backup: {e}")
            return None

import os
import pandas as pd
import glob
import logging
from functools import lru_cache
from collections import OrderedDict

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataCache:
    """
    Efficient caching system for metrics data files.
    """

    def __init__(self, data_dir, max_size=100):
        """
        Initialize the data cache.

        Args:
            data_dir: Base directory for data files
            max_size: Maximum number of items to keep in cache
        """
        self.data_dir = data_dir
        self.max_size = max_size
        self._cache = OrderedDict()  # Using OrderedDict for LRU functionality
        self._file_list_cache = {}

    @lru_cache(maxsize=32)
    def list_files(self, pattern):
        """
        List files matching pattern with caching.

        Args:
            pattern: Glob pattern for files

        Returns:
            List of file paths
        """
        full_pattern = os.path.join(self.data_dir, pattern)
        files = glob.glob(full_pattern)
        return sorted(files)

    def get_dataframe(self, file_path, force_reload=False):
        """
        Get DataFrame from cache or load from disk.

        Args:
            file_path: Path to the data file
            force_reload: Whether to force reload from disk

        Returns:
            DataFrame with data
        """
        # Check if in cache and not forcing reload
        if file_path in self._cache and not force_reload:
            # Move to end of OrderedDict to mark as recently used
            value = self._cache.pop(file_path)
            self._cache[file_path] = value
            return value

        # Load from disk
        try:
            # Determine file type based on extension
            ext = os.path.splitext(file_path)[1].lower()

            if ext == '.json':
                df = pd.read_json(file_path)
            elif ext == '.csv':
                df = pd.read_csv(file_path)
            elif ext == '.parquet':
                df = pd.read_parquet(file_path)
            else:
                logger.warning(f"Unsupported file extension: {ext}")
                return pd.DataFrame()

            # Add to cache
            self._add_to_cache(file_path, df)
            return df

        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
            return pd.DataFrame()

    def _add_to_cache(self, key, value):
        """Add an item to cache with LRU eviction policy."""
        # If cache is full, remove least recently used item
        if len(self._cache) >= self.max_size:
            self._cache.popitem(last=False)

        # Add new item
        self._cache[key] = value

    def clear_cache(self):
        """Clear the entire cache."""
        self._cache.clear()
        self._file_list_cache.clear()
        # Also clear the lru_cache for list_files
        self.list_files.cache_clear()

    def remove_from_cache(self, file_path):
        """Remove a specific file from cache."""
        if file_path in self._cache:
            del self._cache[file_path]

    def load_data_in_date_range(self, file_pattern, start_date=None, end_date=None, date_column='timestamp'):
        """
        Load and combine data files in a date range.

        Args:
            file_pattern: Glob pattern for files
            start_date: Optional start date filter
            end_date: Optional end date filter
            date_column: Column name for date filtering

        Returns:
            Combined DataFrame with data in date range
        """
        # List all matching files
        files = self.list_files(file_pattern)

        if not files:
            logger.warning(f"No files found matching pattern: {file_pattern}")
            return pd.DataFrame()

        # Load all files
        dfs = []
        for file in files:
            df = self.get_dataframe(file)
            if not df.empty:
                dfs.append(df)

        if not dfs:
            return pd.DataFrame()

        # Combine all data
        combined = pd.concat(dfs, ignore_index=True)

        # Apply date filtering if needed
        if not combined.empty and date_column in combined.columns:
            # Ensure datetime type
            if not pd.api.types.is_datetime64_dtype(combined[date_column]):
                combined[date_column] = pd.to_datetime(combined[date_column])

            # Apply filters
            if start_date:
                combined = combined[combined[date_column] >= start_date]
            if end_date:
                combined = combined[combined[date_column] <= end_date]

        return combined

    def get_latest_data(self, file_pattern):
        """
        Get the most recent data file matching a pattern.

        Args:
            file_pattern: Glob pattern for files

        Returns:
            DataFrame with latest data
        """
        files = self.list_files(file_pattern)

        if not files:
            return pd.DataFrame()

        # Sort by modification time (newest first)
        files.sort(key=lambda x: os.path.getmtime(x), reverse=True)

        # Get newest file
        newest_file = files[0]
        return self.get_dataframe(newest_file)

    def save_dataframe(self, df, file_path, file_format='json'):
        """
        Save DataFrame to disk and update cache.

        Args:
            df: DataFrame to save
            file_path: Path to save to
            file_format: Format to use ('json', 'csv', 'parquet')

        Returns:
            Path to saved file
        """
        # Create directory if needed
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        try:
            # Save based on format
            if file_format == 'json':
                df.to_json(file_path, orient='records', date_format='iso')
            elif file_format == 'csv':
                df.to_csv(file_path, index=False)
            elif file_format == 'parquet':
                df.to_parquet(file_path, index=False)
            else:
                logger.warning(f"Unsupported file format: {file_format}")
                return None

            # Update cache
            self._add_to_cache(file_path, df)

            # Clear file list cache to ensure new file is found
            self._file_list_cache.clear()
            self.list_files.cache_clear()

            return file_path

        except Exception as e:
            logger.error(f"Error saving DataFrame to {file_path}: {e}")
            return None
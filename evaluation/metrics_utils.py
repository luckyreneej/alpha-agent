import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def calculate_error_metrics(predictions, actuals):
    """
    Calculate standard error metrics from predictions and actuals.

    Args:
        predictions: Array-like of predicted values
        actuals: Array-like of actual values

    Returns:
        Dictionary of error metrics
    """
    # Convert to numpy arrays if they're not already
    if isinstance(predictions, list):
        predictions = np.array(predictions)
    if isinstance(actuals, list):
        actuals = np.array(actuals)

    error = predictions - actuals
    squared_error = error ** 2
    abs_error = np.abs(error)

    # Calculate metrics, handling possible division by zero
    mse = float(np.mean(squared_error))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(abs_error))

    # Calculate MAPE, handling zeros in actuals
    non_zero_actuals = actuals != 0
    if np.any(non_zero_actuals):
        mape = float(np.mean(np.abs(error[non_zero_actuals] / actuals[non_zero_actuals]) * 100))
    else:
        mape = float('nan')

    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape
    }


def calculate_directional_accuracy(predictions, actuals):
    """
    Calculate directional accuracy from predictions and actuals.

    Args:
        predictions: Array-like of predicted values
        actuals: Array-like of actual values

    Returns:
        Directional accuracy as a float (0-1)
    """
    if len(predictions) <= 1 or len(actuals) <= 1:
        return 0.0

    pred_direction = np.sign(np.diff(predictions))
    actual_direction = np.sign(np.diff(actuals))
    matches = pred_direction == actual_direction

    # Filter out cases where actual direction is zero (no change)
    valid_directions = actual_direction != 0
    if np.any(valid_directions):
        return float(np.mean(matches[valid_directions]))
    else:
        return 0.0


def normalize_metric(values, higher_is_better=True, min_val=None, max_val=None):
    """
    Normalize metric values to 0-1 range.

    Args:
        values: List or array of values to normalize
        higher_is_better: Whether higher values are better
        min_val: Optional minimum value for normalization
        max_val: Optional maximum value for normalization

    Returns:
        List of normalized values
    """
    if len(values) == 0:
        return []

    # Convert to numpy array for efficient operations
    values_array = np.array(values)

    # Use provided min/max or calculate from data
    min_val = min_val if min_val is not None else np.min(values_array)
    max_val = max_val if max_val is not None else np.max(values_array)

    # Handle the case where all values are the same
    if min_val == max_val:
        return [0.5] * len(values)

    # Normalize based on whether higher is better
    if higher_is_better:
        normalized = (values_array - min_val) / (max_val - min_val)
    else:
        normalized = 1 - ((values_array - min_val) / (max_val - min_val))

    # Ensure values are within [0, 1]
    normalized = np.clip(normalized, 0, 1)

    return normalized.tolist()


def calculate_rolling_metrics(values, window_size):
    """
    Calculate rolling metrics efficiently using pandas.

    Args:
        values: Series or array of values
        window_size: Size of the rolling window

    Returns:
        DataFrame with rolling metrics
    """
    if len(values) <= window_size:
        return pd.DataFrame()

    # Convert to pandas Series for rolling operations
    if not isinstance(values, pd.Series):
        values = pd.Series(values)

    # Calculate rolling metrics
    rolling_mean = values.rolling(window=window_size).mean()
    rolling_std = values.rolling(window=window_size).std()
    rolling_min = values.rolling(window=window_size).min()
    rolling_max = values.rolling(window=window_size).max()

    # Combine into DataFrame
    df = pd.DataFrame({
        'mean': rolling_mean,
        'std': rolling_std,
        'min': rolling_min,
        'max': rolling_max
    }).dropna()

    return df


def calculate_performance_score(metrics, weights=None):
    """
    Calculate a weighted performance score from multiple metrics.

    Args:
        metrics: Dictionary of metrics
        weights: Optional dictionary of weights for each metric

    Returns:
        Weighted score (0-100)
    """
    if not metrics:
        return 0

    # Default equal weights if not provided
    if weights is None:
        weights = {key: 1.0 / len(metrics) for key in metrics}

    # Ensure we only use weights for metrics that exist
    valid_keys = set(metrics.keys()) & set(weights.keys())
    if not valid_keys:
        return 0

    # Normalize weights to sum to 1
    total_weight = sum(weights[key] for key in valid_keys)
    if total_weight == 0:
        return 0

    normalized_weights = {key: weights[key] / total_weight for key in valid_keys}

    # Calculate weighted sum
    weighted_sum = sum(metrics[key] * normalized_weights[key] for key in valid_keys)

    # Scale to 0-100
    return max(0, min(100, weighted_sum * 100))


def is_metric_better(metric_name):
    """
    Determine if higher values are better for a given metric.

    Args:
        metric_name: Name of the metric

    Returns:
        Boolean indicating if higher is better
    """
    lower_is_better = any(keyword in metric_name.lower()
                          for keyword in ['error', 'latency', 'time', 'failure',
                                          'drawdown', 'loss', 'negative'])
    return not lower_is_better


def sample_time_series(data, max_points=1000):
    """
    Sample a time series to a maximum number of points.

    Args:
        data: List, array or DataFrame to sample
        max_points: Maximum number of points to include

    Returns:
        Sampled data of the same type as input
    """
    if len(data) <= max_points:
        return data

    # Handle different data types
    if isinstance(data, pd.DataFrame):
        # Calculate sampling interval
        interval = len(data) // max_points

        # Sample with linear indices
        indices = list(range(0, len(data), interval))

        # Ensure the last point is included
        if indices[-1] != len(data) - 1:
            indices.append(len(data) - 1)

        # Limit to max_points
        if len(indices) > max_points:
            indices = indices[:max_points - 1] + [len(data) - 1]

        return data.iloc[indices]

    elif isinstance(data, pd.Series):
        return sample_time_series(data.values, max_points)

    else:  # List or array
        # Calculate sampling interval
        interval = len(data) // max_points

        # Sample with linear indices
        indices = list(range(0, len(data), interval))

        # Ensure the last point is included
        if indices[-1] != len(data) - 1:
            indices.append(len(data) - 1)

        # Limit to max_points
        if len(indices) > max_points:
            indices = indices[:max_points - 1] + [len(data) - 1]

        # Convert to list/array like the input
        if isinstance(data, list):
            return [data[i] for i in indices]
        else:  # numpy array
            return data[indices]


def calculate_time_window(time_range):
    """
    Convert a time range string to a timedelta.

    Args:
        time_range: String like '1h', '6h', '24h', '7d', '30d', 'all'

    Returns:
        timedelta object or None for 'all'
    """
    if time_range == '1h':
        return timedelta(hours=1)
    elif time_range == '6h':
        return timedelta(hours=6)
    elif time_range == '24h':
        return timedelta(hours=24)
    elif time_range == '7d':
        return timedelta(days=7)
    elif time_range == '30d':
        return timedelta(days=30)
    else:  # 'all'
        return None

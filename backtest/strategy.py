import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Any, Callable
import inspect

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Strategy:
    """
    Base class for trading strategies to be used in backtests.
    Strategies generate signals that are interpreted by the backtest engine.
    """

    def __init__(self, name: Optional[str] = None):
        """
        Initialize the strategy.

        Args:
            name: Strategy name (defaults to class name)
        """
        self.name = name or self.__class__.__name__
        self.parameters = {}
        self.description = f"{self.name} Strategy"

    def generate_signals(self, data: pd.DataFrame, current_date: pd.Timestamp,
                         positions: Dict[str, int]) -> Dict[str, float]:
        """
        Generate trading signals for the current date.
        Must be implemented by subclasses.

        Args:
            data: Historical data up to current_date
            current_date: Current date in the backtest
            positions: Current positions {symbol: quantity}

        Returns:
            Dictionary mapping symbols to signal values (-1.0 to 1.0 where -1 is full short, 1 is full long)
        """
        raise NotImplementedError("Subclasses must implement generate_signals")

    def set_parameters(self, **kwargs) -> None:
        """
        Set strategy parameters.

        Args:
            **kwargs: Parameter name-value pairs
        """
        self.parameters.update(kwargs)
        self._update_description()

    def _update_description(self) -> None:
        """Update strategy description based on parameters."""
        params_str = ", ".join(f"{k}={v}" for k, v in self.parameters.items())
        self.description = f"{self.name} Strategy ({params_str})"

    def get_parameters(self) -> Dict[str, Any]:
        """
        Get current strategy parameters.

        Returns:
            Dictionary of parameter name-value pairs
        """
        return self.parameters.copy()


class MovingAverageCrossStrategy(Strategy):
    """
    Moving Average Crossover strategy. Generates buy signals when fast MA crosses above slow MA,
    and sell signals when fast MA crosses below slow MA.
    """

    def __init__(self, fast_period: int = 50, slow_period: int = 200):
        """
        Initialize the Moving Average Cross strategy.

        Args:
            fast_period: Fast moving average period
            slow_period: Slow moving average period
        """
        super().__init__()
        self.set_parameters(fast_period=fast_period, slow_period=slow_period)

    def generate_signals(self, data: pd.DataFrame, current_date: pd.Timestamp,
                         positions: Dict[str, int]) -> Dict[str, float]:
        """
        Generate trading signals using moving average crossover.

        Args:
            data: Historical data up to current_date
            current_date: Current date in the backtest
            positions: Current positions {symbol: quantity}

        Returns:
            Dictionary mapping symbols to signal values
        """
        signals = {}
        fast_period = self.parameters['fast_period']
        slow_period = self.parameters['slow_period']

        # Get unique tickers in the data
        tickers = data['ticker'].unique() if 'ticker' in data.columns else [None]

        for ticker in tickers:
            # Filter data for this ticker
            ticker_data = data[data['ticker'] == ticker] if ticker is not None else data

            # Check if we have enough data for calculation
            if len(ticker_data) < slow_period:
                signals[ticker or 'default'] = 0
                continue

            # Calculate moving averages
            try:
                fast_ma = ticker_data['close'].rolling(window=fast_period).mean()
                slow_ma = ticker_data['close'].rolling(window=slow_period).mean()

                # Get the most recent values
                latest_fast = fast_ma.iloc[-1]
                latest_slow = slow_ma.iloc[-1]

                # Previous values for crossover detection
                prev_fast = fast_ma.iloc[-2] if len(fast_ma) > 1 else None
                prev_slow = slow_ma.iloc[-2] if len(slow_ma) > 1 else None

                # Generate signal
                if prev_fast is not None and prev_slow is not None:
                    # Check for crossover
                    if prev_fast <= prev_slow and latest_fast > latest_slow:
                        # Bullish crossover
                        signals[ticker or 'default'] = 1.0
                    elif prev_fast >= prev_slow and latest_fast < latest_slow:
                        # Bearish crossover
                        signals[ticker or 'default'] = -1.0
                    else:
                        # No crossover, maintain position
                        current_position = positions.get(ticker or 'default', 0)
                        signals[ticker or 'default'] = 1.0 if current_position > 0 else (
                            -1.0 if current_position < 0 else 0)
                else:
                    # Not enough data for crossover detection
                    signals[ticker or 'default'] = 0
            except Exception as e:
                logger.error(f"Error calculating MA crossover for {ticker}: {e}")
                signals[ticker or 'default'] = 0

        return signals


class RSIStrategy(Strategy):
    """
    Relative Strength Index (RSI) strategy. Generates buy signals when RSI is below oversold level,
    and sell signals when RSI is above overbought level.
    """

    def __init__(self, period: int = 14, oversold: int = 30, overbought: int = 70):
        """
        Initialize the RSI strategy.

        Args:
            period: RSI calculation period
            oversold: RSI level below which to buy (oversold)
            overbought: RSI level above which to sell (overbought)
        """
        super().__init__()
        self.set_parameters(period=period, oversold=oversold, overbought=overbought)

    def generate_signals(self, data: pd.DataFrame, current_date: pd.Timestamp,
                         positions: Dict[str, int]) -> Dict[str, float]:
        """
        Generate trading signals using RSI indicator.

        Args:
            data: Historical data up to current_date
            current_date: Current date in the backtest
            positions: Current positions {symbol: quantity}

        Returns:
            Dictionary mapping symbols to signal values
        """
        signals = {}
        period = self.parameters['period']
        oversold = self.parameters['oversold']
        overbought = self.parameters['overbought']

        # Get unique tickers in the data
        tickers = data['ticker'].unique() if 'ticker' in data.columns else [None]

        for ticker in tickers:
            # Filter data for this ticker
            ticker_data = data[data['ticker'] == ticker] if ticker is not None else data

            if len(ticker_data) < period + 1:
                # Not enough data for calculation
                signals[ticker or 'default'] = 0
                continue

            try:
                # Calculate RSI
                price_changes = ticker_data['close'].diff()

                # Calculate gains and losses
                gains = price_changes.copy()
                gains[gains < 0] = 0
                losses = -price_changes.copy()
                losses[losses < 0] = 0

                # Calculate average gains and losses
                avg_gain = gains.rolling(window=period).mean()
                avg_loss = losses.rolling(window=period).mean()

                # Calculate RS and RSI
                # Handle division by zero
                rs = avg_gain / avg_loss.replace(0, np.nan)
                rs = rs.fillna(100)  # If avg_loss is 0, RSI should be near 100
                rsi = 100 - (100 / (1 + rs))

                # Get the latest RSI value
                latest_rsi = rsi.iloc[-1]

                # Generate signal based on RSI level
                if latest_rsi <= oversold:
                    # Oversold condition - buy signal
                    signals[ticker or 'default'] = 1.0
                elif latest_rsi >= overbought:
                    # Overbought condition - sell signal
                    signals[ticker or 'default'] = -1.0
                else:
                    # Neutral zone - maintain current position
                    current_position = positions.get(ticker or 'default', 0)
                    signals[ticker or 'default'] = 1.0 if current_position > 0 else (
                        -1.0 if current_position < 0 else 0)
            except Exception as e:
                logger.error(f"Error calculating RSI for {ticker}: {e}")
                signals[ticker or 'default'] = 0

        return signals


class MACDStrategy(Strategy):
    """
    Moving Average Convergence Divergence (MACD) strategy.
    Generates signals based on MACD line crossing the signal line.
    """

    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        """
        Initialize the MACD strategy.

        Args:
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line period
        """
        super().__init__()
        self.set_parameters(fast_period=fast_period, slow_period=slow_period, signal_period=signal_period)

    def generate_signals(self, data: pd.DataFrame, current_date: pd.Timestamp,
                         positions: Dict[str, int]) -> Dict[str, float]:
        """
        Generate trading signals using MACD indicator.

        Args:
            data: Historical data up to current_date
            current_date: Current date in the backtest
            positions: Current positions {symbol: quantity}

        Returns:
            Dictionary mapping symbols to signal values
        """
        signals = {}
        fast_period = self.parameters['fast_period']
        slow_period = self.parameters['slow_period']
        signal_period = self.parameters['signal_period']

        # Get unique tickers in the data
        tickers = data['ticker'].unique() if 'ticker' in data.columns else [None]

        for ticker in tickers:
            # Filter data for this ticker
            ticker_data = data[data['ticker'] == ticker] if ticker is not None else data

            if len(ticker_data) < slow_period:
                # Not enough data for calculation
                signals[ticker or 'default'] = 0
                continue

            try:
                # Calculate MACD components
                fast_ema = ticker_data['close'].ewm(span=fast_period, adjust=False).mean()
                slow_ema = ticker_data['close'].ewm(span=slow_period, adjust=False).mean()
                macd_line = fast_ema - slow_ema
                signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()

                # Get the most recent values
                latest_macd = macd_line.iloc[-1]
                latest_signal = signal_line.iloc[-1]

                # Previous values for crossover detection
                prev_macd = macd_line.iloc[-2] if len(macd_line) > 1 else None
                prev_signal = signal_line.iloc[-2] if len(signal_line) > 1 else None

                # Generate signal
                if prev_macd is not None and prev_signal is not None:
                    # Check for crossover
                    if prev_macd <= prev_signal and latest_macd > latest_signal:
                        # Bullish crossover
                        signals[ticker or 'default'] = 1.0
                    elif prev_macd >= prev_signal and latest_macd < latest_signal:
                        # Bearish crossover
                        signals[ticker or 'default'] = -1.0
                    else:
                        # No crossover, maintain position
                        current_position = positions.get(ticker or 'default', 0)
                        signals[ticker or 'default'] = 1.0 if current_position > 0 else (
                            -1.0 if current_position < 0 else 0)
                else:
                    # Not enough data for crossover detection
                    signals[ticker or 'default'] = 0
            except Exception as e:
                logger.error(f"Error calculating MACD for {ticker}: {e}")
                signals[ticker or 'default'] = 0

        return signals


def create_combined_strategy(strategies: List[Strategy], weights: Optional[List[float]] = None) -> Strategy:
    """
    Create a combined strategy that averages signals from multiple strategies.

    Args:
        strategies: List of strategies to combine
        weights: Optional list of weights for each strategy (must sum to 1.0)

    Returns:
        A new Strategy instance that combines the provided strategies
    """
    if not strategies:
        raise ValueError("At least one strategy must be provided")

    if weights is None:
        # Equal weighting
        weights = [1.0 / len(strategies)] * len(strategies)
    elif len(weights) != len(strategies):
        raise ValueError("Number of weights must match number of strategies")
    elif abs(sum(weights) - 1.0) > 1e-6:
        raise ValueError("Weights must sum to 1.0")

    class CombinedStrategy(Strategy):
        def __init__(self, component_strategies, component_weights):
            # Use a meaningful name based on component strategies
            strategy_names = [s.name for s in component_strategies]
            name = f"Combined({', '.join(strategy_names)})"
            super().__init__(name=name)

            self.component_strategies = component_strategies
            self.component_weights = component_weights
            self.description = f"Combined strategy using {len(component_strategies)} sub-strategies with custom weights"

        def generate_signals(self, data, current_date, positions):
            combined_signals = {}

            try:
                # Get signals from each component strategy
                all_signals = []
                for strategy in self.component_strategies:
                    signals = strategy.generate_signals(data, current_date, positions)
                    all_signals.append(signals)

                # Combine signals using weights
                all_symbols = set()
                for signals in all_signals:
                    all_symbols.update(signals.keys())

                for symbol in all_symbols:
                    weighted_sum = 0.0
                    for i, signals in enumerate(all_signals):
                        if symbol in signals:
                            weighted_sum += signals[symbol] * self.component_weights[i]

                    # Normalize combined signal to [-1, 1] range
                    combined_signals[symbol] = max(-1.0, min(1.0, weighted_sum))
            except Exception as e:
                logger.error(f"Error combining signals: {e}")
                # Provide a safe default for all symbols
                for symbol in positions.keys():
                    combined_signals[symbol] = 0.0

            return combined_signals

    return CombinedStrategy(strategies, weights)


def create_strategy_from_function(func: Callable, name: Optional[str] = None) -> Strategy:
    """
    Create a strategy from a function that generates signals.

    Args:
        func: Function with signature (data, current_date, positions) -> Dict[str, float]
        name: Optional name for the strategy

    Returns:
        A Strategy instance wrapping the provided function
    """
    # Verify the function has the correct signature
    sig = inspect.signature(func)
    param_names = list(sig.parameters.keys())
    if len(param_names) < 3:
        raise ValueError("Function must accept at least three parameters: data, current_date, positions")

    class FunctionStrategy(Strategy):
        def __init__(self, signal_func, func_name):
            super().__init__(name=func_name)
            self.signal_func = signal_func
            self.description = f"Custom function-based strategy: {func_name}"

        def generate_signals(self, data, current_date, positions):
            try:
                return self.signal_func(data, current_date, positions)
            except Exception as e:
                logger.error(f"Error in function strategy {self.name}: {e}")
                # Return empty signals on error
                return {symbol: 0.0 for symbol in positions.keys()}

    strategy_name = name or func.__name__
    return FunctionStrategy(func, strategy_name)

import numpy as np
import pandas as pd
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def mean_reversion(df: pd.DataFrame) -> pd.Series:
    """Identifies potential mean reversion opportunities."""
    close = df['close']

    # Calculate distance from moving average
    ma_20 = close.rolling(window=20).mean()
    distance = (close - ma_20) / ma_20

    # Invert to get mean reversion signal (negative = overbought, positive = oversold)
    return -distance


class AlphaCalculations:
    """
    Implementation of specific alpha factor calculations.
    This is used by the AlphaFactors class for computing factor values.
    """

    def __init__(self):
        """Initialize the AlphaCalculations class."""
        pass

    # Core Alpha Factor Implementations - Practical factors that are most commonly used

    def alpha1(self, df: pd.DataFrame) -> pd.Series:
        """Rank of max return deviation over 5 days - contrarian factor."""
        returns = df['returns']
        close = df['close']

        # Select return deviation or price based on return direction
        condition = returns < 0
        returns_std = returns.rolling(window=20).std()
        result = np.where(condition, returns_std, close)

        # Square the values
        signed_power = pd.Series(np.power(result, 2), index=df.index)

        # Find which day had max value in last 5 days
        rolling_argmax = signed_power.rolling(window=5).apply(lambda x: x.argmax() if len(x) > 0 else 0, raw=False)

        # Rank and center around zero
        return rolling_argmax.rank(pct=True) - 0.5

    def alpha12(self, df: pd.DataFrame) -> pd.Series:
        """Sign of volume change multiplied by negative price change - momentum reversal."""
        volume = df['volume']
        close = df['close']

        # Get 1-day changes
        delta_volume = volume.diff(1)
        delta_close = close.diff(1)

        # Compute factor: sign(volume change) * -1 * price change
        return np.sign(delta_volume) * (-1 * delta_close)

    def alpha101(self, df: pd.DataFrame) -> pd.Series:
        """Close to open relation divided by high-low range - intraday dynamics."""
        close = df['close']
        open_price = df['open']
        high = df['high']
        low = df['low']

        # Simple ratio of close-open to high-low range
        return (close - open_price) / ((high - low) + 0.001)  # Add small value to avoid division by zero

    # Custom Factors - Additional factors focused on specific aspects of market behavior

    def momentum_factor(self, df: pd.DataFrame) -> pd.Series:
        """Measures price momentum over multiple timeframes."""
        close = df['close']

        # Calculate returns over different periods
        ret_5d = close.pct_change(5)
        ret_10d = close.pct_change(10)
        ret_20d = close.pct_change(20)

        # Combine with weights that favor more recent performance
        return 0.5 * ret_5d + 0.3 * ret_10d + 0.2 * ret_20d

    def volatility_factor(self, df: pd.DataFrame) -> pd.Series:
        """Measures changing volatility regimes."""
        returns = df['returns']

        # Calculate volatility over different periods
        vol_10d = returns.rolling(window=10).std() * np.sqrt(252)  # Annualized
        vol_30d = returns.rolling(window=30).std() * np.sqrt(252)

        # Return relative volatility measure
        return (vol_10d / vol_30d) - 1.0

    def volume_factor(self, df: pd.DataFrame) -> pd.Series:
        """Identifies unusual volume patterns."""
        volume = df['volume']

        # Calculate volume relative to its moving average
        vol_ratio = volume / volume.rolling(window=20).mean()

        # Log transform to manage outliers
        return np.log(vol_ratio + 0.001)

    # Technical Indicators as Alpha Factors

    def rsi_factor(self, df: pd.DataFrame) -> pd.Series:
        """RSI-based factor. Values close to 0 = oversold, close to 1 = overbought."""
        close = df['close']

        # Calculate RSI using pandas directly for simplicity
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)  # Replace zeros to avoid division errors
        rsi = 100 - (100 / (1 + rs))

        # Normalize to 0-1 scale
        return rsi / 100

    def macd_factor(self, df: pd.DataFrame) -> pd.Series:
        """MACD-based factor for trend following."""
        close = df['close']

        # Calculate MACD components
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()

        # Return normalized MACD histogram
        histogram = macd_line - signal_line
        return histogram / close.rolling(window=20).std()


# Utility functions that can be used by multiple alpha factors

def rolling_rank(series, window):
    """Calculate the rank of a rolling window of data."""
    return series.rolling(window=window).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])


def rolling_correlation(x, y, window):
    """Calculate the correlation of two series over a rolling window."""
    return x.rolling(window=window).corr(y)


def ts_rank(series, window):
    """Calculate the time-series rank over a window."""
    return series.rolling(window=window).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])

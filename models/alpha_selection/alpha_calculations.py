import numpy as np
import pandas as pd
import talib
from typing import Dict, List, Optional, Union, Any, Tuple
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AlphaCalculations:
    """
    Implementation of specific alpha factor calculations.
    This is meant to be used by the AlphaFactors class.
    """
    
    @staticmethod
    def alpha1(df: pd.DataFrame) -> pd.Series:
        """(rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) - 0.5)"""
        returns = df['returns']
        close = df['close']
        
        # Create the conditional array
        condition = returns < 0
        returns_std = returns.rolling(window=20).std()
        result = np.where(condition, returns_std, close)
        
        # Apply signed power
        signed_power = pd.Series(np.power(result, 2), index=df.index)
        
        # Use rolling argmax with pandas
        # For each point, look back 5 periods and find the index of max value
        rolling_argmax = signed_power.rolling(window=5).apply(lambda x: x.argmax() if len(x) > 0 else 0, raw=False)
        
        # Rank and subtract 0.5
        return rolling_argmax.rank(pct=True) - 0.5
    
    @staticmethod
    def alpha2(df: pd.DataFrame) -> pd.Series:
        """(-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))"""
        volume = df['volume']
        close = df['close']
        open_price = df['open']
        
        # delta(log(volume), 2)
        log_volume = np.log(volume)
        delta_log_volume = log_volume.diff(2)
        
        # ((close - open) / open)
        return_intraday = (close - open_price) / open_price
        
        # rank both series
        rank_delta_log_volume = delta_log_volume.rank(pct=True)
        rank_return_intraday = return_intraday.rank(pct=True)
        
        # correlation with 6-day window
        corr = rank_delta_log_volume.rolling(window=6).corr(rank_return_intraday)
        
        return -1 * corr
    
    @staticmethod
    def alpha3(df: pd.DataFrame) -> pd.Series:
        """(-1 * correlation(rank(open), rank(volume), 10))"""
        open_price = df['open']
        volume = df['volume']
        
        rank_open = open_price.rank(pct=True)
        rank_volume = volume.rank(pct=True)
        
        corr = rank_open.rolling(window=10).corr(rank_volume)
        
        return -1 * corr
    
    @staticmethod
    def alpha4(df: pd.DataFrame) -> pd.Series:
        """(-1 * Ts_Rank(rank(low), 9))"""
        low = df['low']
        rank_low = low.rank(pct=True)
        
        # Time series rank over 9 days
        ts_rank = rank_low.rolling(window=9).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
        
        return -1 * ts_rank
    
    @staticmethod
    def alpha5(df: pd.DataFrame) -> pd.Series:
        """(rank((open - (sum(vwap, 10) / 10))) * (-1 * abs(rank((close - vwap)))))"""
        vwap = df['vwap']
        open_price = df['open']
        close = df['close']
        
        vwap_mean_10 = vwap.rolling(window=10).mean()
        open_minus_vwap = open_price - vwap_mean_10
        
        rank_open_minus_vwap = open_minus_vwap.rank(pct=True)
        close_minus_vwap = close - vwap
        rank_close_minus_vwap = close_minus_vwap.rank(pct=True)
        
        return rank_open_minus_vwap * (-1 * np.abs(rank_close_minus_vwap))
    
    @staticmethod
    def alpha6(df: pd.DataFrame) -> pd.Series:
        """(-1 * correlation(open, volume, 10))"""
        open_price = df['open']
        volume = df['volume']
        
        corr = open_price.rolling(window=10).corr(volume)
        
        return -1 * corr
    
    @staticmethod
    def alpha7(df: pd.DataFrame) -> pd.Series:
        """((adv20 < volume) ? ((-1 * ts_rank(abs(delta(close, 7)), 60)) * sign(delta(close, 7))) : (-1 * 1))"""
        adv20 = df['adv20']
        volume = df['volume']
        close = df['close']
        
        # delta(close, 7)
        delta_close_7 = close.diff(7)
        
        # ts_rank(abs(delta(close, 7)), 60)
        abs_delta_close = np.abs(delta_close_7)
        ts_rank_abs_delta = abs_delta_close.rolling(window=60).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
        
        sign_delta_close = np.sign(delta_close_7)
        
        # Conditional calculation
        result = pd.Series(index=df.index)
        condition = adv20 < volume
        result[condition] = -1 * ts_rank_abs_delta[condition] * sign_delta_close[condition]
        result[~condition] = -1
        
        return result
    
    @staticmethod
    def alpha8(df: pd.DataFrame) -> pd.Series:
        """(-1 * rank(((sum(open, 5) * sum(returns, 5)) - delay((sum(open, 5) * sum(returns, 5)), 10))))"""
        open_price = df['open']
        returns = df['returns']
        
        sum_open_5 = open_price.rolling(window=5).sum()
        sum_returns_5 = returns.rolling(window=5).sum()
        
        product = sum_open_5 * sum_returns_5
        product_delay_10 = product.shift(10)
        
        diff = product - product_delay_10
        
        return -1 * diff.rank(pct=True)
    
    @staticmethod
    def alpha9(df: pd.DataFrame) -> pd.Series:
        """((0 < ts_min(delta(close, 1), 5)) ? delta(close, 1) : ((ts_max(delta(close, 1), 5) < 0) ? delta(close, 1) : (-1 * delta(close, 1))))"""
        close = df['close']
        delta_close = close.diff(1)
        
        ts_min_delta = delta_close.rolling(window=5).min()
        ts_max_delta = delta_close.rolling(window=5).max()
        
        result = pd.Series(index=df.index)
        cond1 = 0 < ts_min_delta
        cond2 = ts_max_delta < 0
        
        result[cond1] = delta_close[cond1]
        result[~cond1 & cond2] = delta_close[~cond1 & cond2]
        result[~cond1 & ~cond2] = -1 * delta_close[~cond1 & ~cond2]
        
        return result
    
    @staticmethod
    def alpha10(df: pd.DataFrame) -> pd.Series:
        """rank(((0 < ts_min(delta(close, 1), 4)) ? delta(close, 1) : ((ts_max(delta(close, 1), 4) < 0) ? delta(close, 1) : (-1 * delta(close, 1)))))"""
        close = df['close']
        delta_close = close.diff(1)
        
        ts_min_delta = delta_close.rolling(window=4).min()
        ts_max_delta = delta_close.rolling(window=4).max()
        
        result = pd.Series(index=df.index)
        cond1 = 0 < ts_min_delta
        cond2 = ts_max_delta < 0
        
        result[cond1] = delta_close[cond1]
        result[~cond1 & cond2] = delta_close[~cond1 & cond2]
        result[~cond1 & ~cond2] = -1 * delta_close[~cond1 & ~cond2]
        
        return result.rank(pct=True)
    
    @staticmethod
    def alpha11(df: pd.DataFrame) -> pd.Series:
        """((rank(ts_max((vwap - close), 3)) + rank(ts_min((vwap - close), 3))) * rank(delta(volume, 3)))"""
        vwap = df['vwap']
        close = df['close']
        volume = df['volume']
        
        vwap_minus_close = vwap - close
        ts_max_vmc = vwap_minus_close.rolling(window=3).max().rank(pct=True)
        ts_min_vmc = vwap_minus_close.rolling(window=3).min().rank(pct=True)
        
        delta_volume = volume.diff(3).rank(pct=True)
        
        return (ts_max_vmc + ts_min_vmc) * delta_volume
    
    @staticmethod
    def alpha12(df: pd.DataFrame) -> pd.Series:
        """(sign(delta(volume, 1)) * (-1 * delta(close, 1)))"""
        volume = df['volume']
        close = df['close']
        
        delta_volume = volume.diff(1)
        delta_close = close.diff(1)
        
        return np.sign(delta_volume) * (-1 * delta_close)
    
    @staticmethod
    def alpha13(df: pd.DataFrame) -> pd.Series:
        """(-1 * rank(covariance(rank(close), rank(volume), 5)))"""
        close = df['close']
        volume = df['volume']
        
        rank_close = close.rank(pct=True)
        rank_volume = volume.rank(pct=True)
        
        # Calculate the rolling covariance
        def rolling_cov(x, y, window):
            return x.rolling(window=window).cov(y)
        
        covar = rolling_cov(rank_close, rank_volume, 5)
        
        return -1 * covar.rank(pct=True)
    
    @staticmethod
    def alpha14(df: pd.DataFrame) -> pd.Series:
        """((-1 * rank(delta(returns, 3))) * correlation(open, volume, 10))"""
        returns = df['returns']
        open_price = df['open']
        volume = df['volume']
        
        delta_returns = returns.diff(3).rank(pct=True) * -1
        corr_open_volume = open_price.rolling(window=10).corr(volume)
        
        return delta_returns * corr_open_volume
    
    @staticmethod
    def alpha15(df: pd.DataFrame) -> pd.Series:
        """(-1 * sum(rank(correlation(rank(high), rank(volume), 3)), 3))"""
        high = df['high']
        volume = df['volume']
        
        # Calculate rank correlation with a custom function
        def rolling_rank_corr(x, y, window):
            x_rank = x.rolling(window=window).apply(lambda z: pd.Series(z).rank(pct=True).iloc[-1])
            y_rank = y.rolling(window=window).apply(lambda z: pd.Series(z).rank(pct=True).iloc[-1])
            return x_rank.rolling(window=window).corr(y_rank)
        
        rank_corr = rolling_rank_corr(high, volume, 3)
        rank_of_corr = rank_corr.rank(pct=True)
        sum_rank_corr = rank_of_corr.rolling(window=3).sum()
        
        return -1 * sum_rank_corr
    
    @staticmethod
    def alpha16(df: pd.DataFrame) -> pd.Series:
        """(-1 * rank(covariance(rank(high), rank(volume), 5)))"""
        high = df['high']
        volume = df['volume']
        
        rank_high = high.rank(pct=True)
        rank_volume = volume.rank(pct=True)
        
        # Calculate the rolling covariance
        def rolling_cov(x, y, window):
            return x.rolling(window=window).cov(y)
        
        covar = rolling_cov(rank_high, rank_volume, 5)
        
        return -1 * covar.rank(pct=True)
        
    @staticmethod
    def alpha101(df: pd.DataFrame) -> pd.Series:
        """((close - open) / ((high - low) + 0.001))"""
        close = df['close']
        open_price = df['open']
        high = df['high']
        low = df['low']
        
        return (close - open_price) / ((high - low) + 0.001)
    
    @staticmethod
    def election_year_momentum(df: pd.DataFrame) -> pd.Series:
        """
        Custom alpha factor that analyzes momentum patterns during election years.
        This is a simplified implementation - in a real system, it would incorporate actual
        election year data and more sophisticated analysis.
        """
        # This is a placeholder implementation
        # In a real system, you'd analyze patterns around actual election year dates
        returns = df['returns'] 
        return returns.rolling(window=20).mean() * 5
    
    @staticmethod
    def sector_rotation(df: pd.DataFrame) -> pd.Series:
        """
        Custom alpha factor for sector rotation strategies.
        This is a placeholder implementation - a real implementation would
        require sector classification data.
        """
        # This is a simplified placeholder implementation
        close = df['close']
        volume = df['volume']
        
        # Some arbitrary calculation to represent sector momentum
        return close.pct_change(20) * (volume / volume.rolling(window=20).mean())
    
    @staticmethod
    def volatility_regime(df: pd.DataFrame) -> pd.Series:
        """
        Custom alpha factor for detecting volatility regime shifts.
        """
        # Calculate historical volatility
        returns = df['returns']
        hist_vol_20 = returns.rolling(window=20).std() * np.sqrt(252)  # Annualized
        hist_vol_60 = returns.rolling(window=60).std() * np.sqrt(252)
        
        # Volatility regime shift indicator
        vol_ratio = hist_vol_20 / hist_vol_60
        
        return vol_ratio - 1.0  # Normalized around zero


# Utility functions for alpha factor calculations
def rolling_rank(series, window):
    """Calculate the rank of a rolling window of data."""
    return series.rolling(window=window).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])

def rolling_correlation(x, y, window):
    """Calculate the correlation of two series over a rolling window."""
    return x.rolling(window=window).corr(y)

def rolling_covariance(x, y, window):
    """Calculate the covariance of two series over a rolling window."""
    return x.rolling(window=window).cov(y)

def ts_rank(series, window):
    """Calculate the time-series rank over a window."""
    return series.rolling(window=window).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])

def ts_min(series, window):
    """Calculate the minimum value over a rolling window."""
    return series.rolling(window=window).min()

def ts_max(series, window):
    """Calculate the maximum value over a rolling window."""
    return series.rolling(window=window).max()
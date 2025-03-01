import pandas as pd
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HistoricalDataFetcher:
    """
    Fetches and processes historical data for backtesting.
    """

    def __init__(self, api_client=None, cache_dir: str = 'data/historical'):
        """
        Initialize the historical data fetcher.

        Args:
            api_client: API client for fetching data (dependency injection)
            cache_dir: Directory for caching historical data
        """
        self.api = api_client
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def fetch_stock_history(self,
                            tickers: List[str],
                            start_date: str,
                            end_date: str,
                            timespan: str = 'day',
                            use_cache: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical stock data for multiple tickers.

        Args:
            tickers: List of ticker symbols
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            timespan: Time interval ('minute', 'hour', 'day', etc.)
            use_cache: Whether to use cached data when available

        Returns:
            Dictionary mapping tickers to DataFrames of historical data
        """
        result = {}

        for ticker in tickers:
            logger.info(f"Fetching {timespan} data for {ticker} from {start_date} to {end_date}")

            # Check for cached data if requested
            cache_file = os.path.join(self.cache_dir,
                                      f"{ticker}_{timespan}_{start_date}_{end_date}.parquet")

            if use_cache and os.path.exists(cache_file):
                logger.info(f"Loading cached data for {ticker}")
                try:
                    df = pd.read_parquet(cache_file)
                    result[ticker] = df
                    continue
                except Exception as e:
                    logger.warning(f"Error reading cache file: {e}. Fetching from API instead.")

            # Fetch from API
            try:
                if self.api is None:
                    logger.error(f"No API client provided for {ticker}")
                    continue

                df = self.api.get_stock_bars(
                    ticker=ticker,
                    timespan=timespan,
                    from_date=start_date,
                    to_date=end_date
                )

                if df.empty:
                    logger.warning(f"No data retrieved for {ticker}")
                    continue

                result[ticker] = df

                # Cache the data
                try:
                    df.to_parquet(cache_file)
                    logger.info(f"Cached data for {ticker} to {cache_file}")
                except Exception as e:
                    logger.warning(f"Error caching data: {e}")

                # Respect API rate limits
                time.sleep(0.2)

            except Exception as e:
                logger.error(f"Error fetching data for {ticker}: {e}")

        return result

    def prepare_backtest_data(self,
                              dataset: Dict[str, Any],
                              format_type: str = 'panel',
                              resample_freq: Optional[str] = None) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Prepare and format data for backtesting.

        Args:
            dataset: Complete dataset with stock data
            format_type: 'panel' for multi-ticker DataFrame or 'dict' for separate DataFrames
            resample_freq: Optional frequency for resampling (e.g., 'W' for weekly)

        Returns:
            DataFrame or dictionary of DataFrames formatted for backtesting
        """
        if not dataset.get('stocks'):
            raise ValueError("No stock data available for backtesting")

        if format_type == 'panel':
            # Create a multi-index panel with (date, ticker) as index
            panel_data = []

            for ticker, df in dataset['stocks'].items():
                ticker_df = df.copy()
                ticker_df['ticker'] = ticker
                panel_data.append(ticker_df)

            if not panel_data:
                raise ValueError("No data available for panel format")

            # Combine all dataframes
            panel = pd.concat(panel_data, ignore_index=True)

            # Convert to datetime and set index
            panel['date'] = pd.to_datetime(panel['date'])
            panel.set_index(['date', 'ticker'], inplace=True)

            # Resample if requested
            if resample_freq:
                # Need to handle each ticker separately
                tickers = panel.index.get_level_values(1).unique()
                resampled_dfs = []

                for ticker in tickers:
                    ticker_data = panel.xs(ticker, level=1)
                    resampled = ticker_data.resample(resample_freq).agg({
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum'
                    })

                    # Recreate multi-index
                    resampled['ticker'] = ticker
                    resampled.reset_index(inplace=True)
                    resampled.set_index(['date', 'ticker'], inplace=True)

                    resampled_dfs.append(resampled)

                panel = pd.concat(resampled_dfs)

            return panel

        else:  # format_type == 'dict'
            result = {}

            for ticker, df in dataset['stocks'].items():
                # Prepare DataFrame
                ticker_df = df.copy()

                # Convert date to datetime and set as index
                ticker_df['date'] = pd.to_datetime(ticker_df['date'])
                ticker_df.set_index('date', inplace=True)

                # Resample if requested
                if resample_freq:
                    ticker_df = ticker_df.resample(resample_freq).agg({
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum'
                    })

                result[ticker] = ticker_df

            return result

    def load_local_data(self, data_dir: str, tickers: List[str] = None) -> Dict[str, Any]:
        """
        Load data from local files instead of fetching from API.

        Args:
            data_dir: Directory containing data files
            tickers: Optional list of tickers to load (loads all if None)

        Returns:
            Dictionary containing all loaded data
        """
        dataset = {
            'stocks': {},
            'metadata': {
                'source': 'local',
                'load_time': datetime.now().isoformat()
            }
        }

        if not os.path.exists(data_dir):
            logger.error(f"Data directory {data_dir} does not exist")
            return dataset

        # Find all parquet files in the directory
        files = [f for f in os.listdir(data_dir) if f.endswith('.parquet')]

        # Filter by tickers if specified
        if tickers:
            files = [f for f in files if any(f.startswith(ticker) for ticker in tickers)]

        # Load each file
        for file in files:
            try:
                ticker = file.split('_')[0]  # Assumes filename format: TICKER_*.parquet

                df = pd.read_parquet(os.path.join(data_dir, file))

                if 'date' not in df.columns:
                    logger.warning(f"File {file} does not contain a 'date' column")
                    continue

                dataset['stocks'][ticker] = df
                logger.info(f"Loaded data for {ticker} from {file}")

            except Exception as e:
                logger.error(f"Error loading file {file}: {e}")

        return dataset

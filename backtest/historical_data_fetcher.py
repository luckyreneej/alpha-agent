import pandas as pd
import numpy as np
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple

# Add parent directory to path for imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data.polygon_api import PolygonAPI

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HistoricalDataFetcher:
    """
    Fetches and processes historical data for backtesting using Polygon API.
    Supports various time ranges, asset types, and data frequencies.
    """
    
    def __init__(self, api_key: Optional[str] = None, cache_dir: str = 'data/historical'):
        """
        Initialize the historical data fetcher.
        
        Args:
            api_key: Polygon API key (optional)
            cache_dir: Directory for caching historical data
        """
        self.api = PolygonAPI(api_key)
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
    
    def fetch_options_history(self, 
                             underlying_tickers: List[str],
                             start_date: str,
                             end_date: str,
                             expiration_range_days: int = 60,
                             use_cache: bool = True) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Fetch historical options data for multiple underlying tickers.
        
        Args:
            underlying_tickers: List of underlying ticker symbols
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            expiration_range_days: Maximum days to expiration to include
            use_cache: Whether to use cached data when available
            
        Returns:
            Nested dictionary: {underlying_ticker: {option_contract: dataframe}}
        """
        result = {}
        
        for ticker in underlying_tickers:
            logger.info(f"Fetching options data for {ticker}")
            
            # Check for cached contracts data
            cache_contracts_file = os.path.join(self.cache_dir, 
                                             f"{ticker}_options_contracts_{end_date}.parquet")
            
            # First, get available options contracts
            contracts_df = None
            if use_cache and os.path.exists(cache_contracts_file):
                logger.info(f"Loading cached options contracts for {ticker}")
                try:
                    contracts_df = pd.read_parquet(cache_contracts_file)
                except Exception as e:
                    logger.warning(f"Error reading cache file: {e}. Fetching from API instead.")
            
            if contracts_df is None:
                try:
                    # Get options contracts
                    contracts_df = self.api.get_options_contracts(
                        underlying_ticker=ticker,
                        limit=500  # Adjust based on needs
                    )
                    
                    if contracts_df.empty:
                        logger.warning(f"No options contracts found for {ticker}")
                        continue
                    
                    # Cache the contracts data
                    try:
                        contracts_df.to_parquet(cache_contracts_file)
                        logger.info(f"Cached options contracts for {ticker} to {cache_contracts_file}")
                    except Exception as e:
                        logger.warning(f"Error caching options contracts: {e}")
                    
                    # Respect API rate limits
                    time.sleep(0.2)
                    
                except Exception as e:
                    logger.error(f"Error fetching options contracts for {ticker}: {e}")
                    continue
            
            # Filter contracts by expiration date if expiration_range_days is specified
            if expiration_range_days > 0 and 'expiration_date' in contracts_df.columns:
                # Convert to datetime for comparison
                end_date_dt = pd.to_datetime(end_date)
                contracts_df['expiration_dt'] = pd.to_datetime(contracts_df['expiration_date'])
                
                # Filter contracts within range
                cutoff_date = end_date_dt + timedelta(days=expiration_range_days)
                filtered_contracts = contracts_df[
                    (contracts_df['expiration_dt'] >= end_date_dt) & 
                    (contracts_df['expiration_dt'] <= cutoff_date)
                ]
                
                if filtered_contracts.empty:
                    logger.warning(f"No contracts within expiration range for {ticker}")
                    continue
            else:
                filtered_contracts = contracts_df
            
            # For each contract, fetch historical data
            ticker_options_data = {}
            
            # Limit the number of contracts to fetch (API constraints)
            max_contracts = 20  # Adjust based on needs
            sample_contracts = filtered_contracts.sample(min(max_contracts, len(filtered_contracts))) if len(filtered_contracts) > max_contracts else filtered_contracts
            
            for _, contract in sample_contracts.iterrows():
                contract_ticker = contract.get('ticker')
                if not contract_ticker:
                    continue
                    
                # Check for cached data
                cache_file = os.path.join(self.cache_dir, 
                                         f"{contract_ticker}_{start_date}_{end_date}.parquet")
                
                if use_cache and os.path.exists(cache_file):
                    logger.info(f"Loading cached data for {contract_ticker}")
                    try:
                        df = pd.read_parquet(cache_file)
                        ticker_options_data[contract_ticker] = df
                        continue
                    except Exception as e:
                        logger.warning(f"Error reading cache file: {e}. Fetching from API instead.")
                
                # Fetch from API
                try:
                    # For options, we use the regular aggs endpoint with the options ticker
                    df = self.api.get_stock_bars(
                        ticker=contract_ticker,
                        timespan='day',
                        from_date=start_date,
                        to_date=end_date
                    )
                    
                    if df.empty:
                        logger.warning(f"No data retrieved for {contract_ticker}")
                        continue
                    
                    # Add contract details for reference
                    for col in ['strike_price', 'expiration_date', 'contract_type']:
                        if col in contract:
                            df[col] = contract[col]
                    
                    ticker_options_data[contract_ticker] = df
                    
                    # Cache the data
                    try:
                        df.to_parquet(cache_file)
                        logger.info(f"Cached data for {contract_ticker} to {cache_file}")
                    except Exception as e:
                        logger.warning(f"Error caching data: {e}")
                    
                    # Respect API rate limits
                    time.sleep(0.2)
                    
                except Exception as e:
                    logger.error(f"Error fetching data for {contract_ticker}: {e}")
            
            if ticker_options_data:
                result[ticker] = ticker_options_data
        
        return result
    
    def fetch_market_news(self, 
                        start_date: str,
                        end_date: str,
                        tickers: Optional[List[str]] = None,
                        use_cache: bool = True) -> pd.DataFrame:
        """
        Fetch historical market news.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            tickers: Optional list of tickers to filter news for
            use_cache: Whether to use cached data when available
            
        Returns:
            DataFrame with news articles
        """
        cache_key = "_".join(tickers) if tickers else "general"
        cache_file = os.path.join(self.cache_dir, 
                               f"news_{cache_key}_{start_date}_{end_date}.parquet")
        
        if use_cache and os.path.exists(cache_file):
            logger.info(f"Loading cached news data")
            try:
                return pd.read_parquet(cache_file)
            except Exception as e:
                logger.warning(f"Error reading cache file: {e}. Fetching from API instead.")
        
        try:
            logger.info(f"Fetching news from {start_date} to {end_date}")
            
            # Fetch news
            news_df = self.api.get_market_news(
                ticker=tickers[0] if tickers and len(tickers) == 1 else None,  # Single ticker filter
                from_date=start_date,
                to_date=end_date,
                limit=1000  # Adjust based on needs
            )
            
            if news_df.empty:
                logger.warning("No news articles found")
                return pd.DataFrame()
            
            # For multiple tickers, filter the results
            if tickers and len(tickers) > 1 and 'tickers' in news_df.columns:
                filtered_news = news_df[news_df['tickers'].apply(
                    lambda x: any(ticker in x for ticker in tickers) if isinstance(x, list) else False
                )]
                news_df = filtered_news if not filtered_news.empty else news_df
            
            # Cache the data
            try:
                news_df.to_parquet(cache_file)
                logger.info(f"Cached news data to {cache_file}")
            except Exception as e:
                logger.warning(f"Error caching news data: {e}")
            
            return news_df
            
        except Exception as e:
            logger.error(f"Error fetching news data: {e}")
            return pd.DataFrame()
    
    def fetch_complete_dataset(self,
                             tickers: List[str],
                             start_date: str,
                             end_date: str,
                             include_options: bool = False,
                             include_news: bool = False) -> Dict[str, Any]:
        """
        Fetch a complete historical dataset for backtesting, including stocks, options, and news.
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            include_options: Whether to include options data
            include_news: Whether to include news data
            
        Returns:
            Dictionary containing all fetched data
        """
        dataset = {
            'stocks': None,
            'options': None,
            'news': None,
            'metadata': {
                'start_date': start_date,
                'end_date': end_date,
                'tickers': tickers,
                'fetch_time': datetime.now().isoformat()
            }
        }
        
        # Fetch stock data
        logger.info(f"Fetching complete dataset for {tickers} from {start_date} to {end_date}")
        dataset['stocks'] = self.fetch_stock_history(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            timespan='day'  # Daily data for backtesting
        )
        
        # Fetch options data if requested
        if include_options:
            dataset['options'] = self.fetch_options_history(
                underlying_tickers=tickers,
                start_date=start_date,
                end_date=end_date,
                expiration_range_days=60  # Contracts expiring within 60 days
            )
        
        # Fetch news data if requested
        if include_news:
            dataset['news'] = self.fetch_market_news(
                start_date=start_date,
                end_date=end_date,
                tickers=tickers
            )
        
        return dataset
    
    def prepare_backtest_data(self,
                            dataset: Dict[str, Any],
                            format_type: str = 'panel',
                            resample_freq: Optional[str] = None) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Prepare and format data for backtesting.
        
        Args:
            dataset: Complete dataset from fetch_complete_dataset()
            format_type: 'panel' for multi-ticker DataFrame or 'dict' for separate DataFrames
            resample_freq: Optional frequency for resampling (e.g., 'W' for weekly)
            
        Returns:
            DataFrame or dictionary of DataFrames formatted for backtesting
        """
        if not dataset['stocks']:
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
                # This is more complex for a panel - we need to handle each ticker separately
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
    
    def save_dataset(self, dataset: Dict[str, Any], filename: str) -> str:
        """
        Save the complete dataset to disk.
        
        Args:
            dataset: Dataset from fetch_complete_dataset()
            filename: Base filename without extension
            
        Returns:
            Path to saved files
        """
        save_dir = os.path.join(self.cache_dir, filename)
        os.makedirs(save_dir, exist_ok=True)
        
        # Save metadata
        with open(os.path.join(save_dir, 'metadata.json'), 'w') as f:
            import json
            json.dump(dataset['metadata'], f, indent=2)
        
        # Save stock data
        if dataset['stocks']:
            for ticker, df in dataset['stocks'].items():
                df.to_parquet(os.path.join(save_dir, f"{ticker}_stock.parquet"))
        
        # Save options data
        if dataset['options']:
            for underlying, contracts in dataset['options'].items():
                os.makedirs(os.path.join(save_dir, f"{underlying}_options"), exist_ok=True)
                
                for contract, df in contracts.items():
                    # Sanitize contract ticker for filename
                    contract_filename = contract.replace(':', '_')
                    df.to_parquet(os.path.join(save_dir, f"{underlying}_options", f"{contract_filename}.parquet"))
        
        # Save news data
        if dataset['news'] is not None and not dataset['news'].empty:
            dataset['news'].to_parquet(os.path.join(save_dir, "news.parquet"))
        
        logger.info(f"Dataset saved to {save_dir}")
        return save_dir
import os
import pandas as pd
import numpy as np
import logging
import time
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta
import requests

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PolygonAPI:
    """
    Client for accessing Polygon.io API to retrieve market data.
    Supports both RESTful API and WebSocket for real-time data.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Polygon API client.
        
        Args:
            api_key: Polygon.io API key. If None, will attempt to read from environment variable POLYGON_API_KEY.
        """
        self.api_key = api_key or os.environ.get('POLYGON_API_KEY')
        if not self.api_key:
            raise ValueError("API key is required. Provide it as an argument or set POLYGON_API_KEY environment variable.")
            
        self.base_url = "https://api.polygon.io"
        self.session = requests.Session()
        self.rate_limit_remaining = 5  # Conservative initial value
        self.rate_limit_reset = 0      # Time when rate limit resets
    
    def _handle_rate_limiting(self):
        """
        Handle API rate limiting by sleeping if necessary.
        """
        if self.rate_limit_remaining <= 1:
            sleep_time = max(0, self.rate_limit_reset - time.time())
            if sleep_time > 0:
                logger.info(f"Rate limit reached. Sleeping for {sleep_time:.2f} seconds.")
                time.sleep(sleep_time + 0.5)  # Add a small buffer
    
    def _make_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Make a request to the Polygon API.
        
        Args:
            endpoint: API endpoint (without base URL)
            params: Query parameters
            
        Returns:
            API response as dictionary
        """
        # Handle rate limiting
        self._handle_rate_limiting()
        
        url = f"{self.base_url}{endpoint}"
        headers = {
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # Add API key to params for endpoints that require it
        if params is None:
            params = {}
        
        if not endpoint.startswith("/v3/"):  # v3 endpoints use Bearer token
            params['apiKey'] = self.api_key
            headers = {}  # No auth header needed
        
        try:
            response = self.session.get(url, params=params, headers=headers)
            
            # Update rate limit information
            if 'X-RateLimit-Remaining' in response.headers:
                self.rate_limit_remaining = int(response.headers['X-RateLimit-Remaining'])
            if 'X-RateLimit-Reset' in response.headers:
                self.rate_limit_reset = int(response.headers['X-RateLimit-Reset'])
            
            # Handle HTTP errors
            response.raise_for_status()
            
            data = response.json()
            
            if 'error' in data and data['error']:
                logger.error(f"API Error: {data['error']}")
                return {'status': 'error', 'error': data['error']}
            
            return data
        
        except requests.RequestException as e:
            logger.error(f"Request error: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def get_stock_bars(self, 
                      ticker: str, 
                      timespan: str = 'day', 
                      from_date: str = None, 
                      to_date: str = None, 
                      limit: int = 1000, 
                      multiplier: int = 1) -> pd.DataFrame:
        """
        Get historical price bars for a stock.
        
        Args:
            ticker: Stock ticker symbol
            timespan: Time span of the bars ('minute', 'hour', 'day', 'week', 'month', 'quarter', 'year')
            from_date: Start date in YYYY-MM-DD format
            to_date: End date in YYYY-MM-DD format
            limit: Maximum number of bars to return
            multiplier: Multiplier for the timespan
            
        Returns:
            DataFrame with OHLCV data
        """
        # Default dates if not provided
        if not to_date:
            to_date = datetime.now().strftime('%Y-%m-%d')
        if not from_date:
            from_date = (datetime.strptime(to_date, '%Y-%m-%d') - timedelta(days=30)).strftime('%Y-%m-%d')
        
        endpoint = f"/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
        params = {
            "limit": limit,
            "adjusted": "true"
        }
        
        logger.info(f"Fetching {ticker} bars from {from_date} to {to_date}")
        response = self._make_request(endpoint, params)
        
        if response.get('status') == 'error' or 'results' not in response:
            logger.error(f"Error fetching bars for {ticker}: {response.get('error', 'Unknown error')}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        bars = response['results']
        if not bars:
            logger.warning(f"No bars returned for {ticker}")
            return pd.DataFrame()
        
        df = pd.DataFrame(bars)
        
        # Rename columns to standard names
        column_map = {
            't': 'timestamp',
            'o': 'open',
            'h': 'high',
            'l': 'low',
            'c': 'close',
            'v': 'volume',
            'vw': 'vwap',
            'n': 'num_trades'
        }
        
        df.rename(columns=column_map, inplace=True)
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['date'] = df['timestamp'].dt.date
        
        # Sort by timestamp
        df.sort_values('timestamp', inplace=True)
        
        return df
    
    def get_multiple_stocks_bars(self, 
                               tickers: List[str], 
                               timespan: str = 'day', 
                               from_date: str = None, 
                               to_date: str = None) -> pd.DataFrame:
        """
        Get historical price bars for multiple stocks.
        
        Args:
            tickers: List of stock ticker symbols
            timespan: Time span of the bars ('minute', 'hour', 'day', 'week', 'month')
            from_date: Start date in YYYY-MM-DD format
            to_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with OHLCV data for all tickers
        """
        all_dfs = []
        
        for ticker in tickers:
            df = self.get_stock_bars(ticker, timespan, from_date, to_date)
            if not df.empty:
                df['ticker'] = ticker
                all_dfs.append(df)
            
            # Slight delay to avoid rate limiting
            time.sleep(0.1)
        
        if not all_dfs:
            return pd.DataFrame()
        
        return pd.concat(all_dfs, ignore_index=True)
    
    def get_options_contracts(self, 
                            underlying_ticker: str, 
                            expiration_date: Optional[str] = None, 
                            strike_price: Optional[float] = None, 
                            contract_type: Optional[str] = None,
                            limit: int = 100) -> pd.DataFrame:
        """
        Get options contracts for an underlying stock.
        
        Args:
            underlying_ticker: Underlying stock ticker
            expiration_date: Option expiration date (YYYY-MM-DD)
            strike_price: Filter by strike price
            contract_type: Filter by contract type ('call', 'put')
            limit: Maximum number of contracts to return
            
        Returns:
            DataFrame with options contracts
        """
        endpoint = f"/v3/reference/options/contracts"
        params = {
            "underlying_ticker": underlying_ticker,
            "limit": limit
        }
        
        if expiration_date:
            params['expiration_date'] = expiration_date
        if strike_price:
            params['strike_price'] = strike_price
        if contract_type:
            params['contract_type'] = contract_type.lower()
        
        logger.info(f"Fetching options contracts for {underlying_ticker}")
        response = self._make_request(endpoint, params)
        
        if response.get('status') == 'error' or 'results' not in response:
            logger.error(f"Error fetching options for {underlying_ticker}: {response.get('error', 'Unknown error')}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        contracts = response['results']
        if not contracts:
            logger.warning(f"No options contracts found for {underlying_ticker}")
            return pd.DataFrame()
        
        return pd.DataFrame(contracts)
    
    def get_options_quotes(self, 
                         contract_ticker: str, 
                         from_date: Optional[str] = None, 
                         to_date: Optional[str] = None, 
                         limit: int = 100) -> pd.DataFrame:
        """
        Get historical quotes for an options contract.
        
        Args:
            contract_ticker: Options contract ticker symbol (e.g., O:AAPL210917C00125000)
            from_date: Start date in YYYY-MM-DD format
            to_date: End date in YYYY-MM-DD format
            limit: Maximum number of quotes to return
            
        Returns:
            DataFrame with options quotes
        """
        # Default dates if not provided
        if not to_date:
            to_date = datetime.now().strftime('%Y-%m-%d')
        if not from_date:
            from_date = (datetime.strptime(to_date, '%Y-%m-%d') - timedelta(days=7)).strftime('%Y-%m-%d')
        
        endpoint = f"/v3/quotes/{contract_ticker}"
        params = {
            "timestamp.gte": f"{from_date}T00:00:00Z",
            "timestamp.lte": f"{to_date}T23:59:59Z",
            "limit": limit
        }
        
        logger.info(f"Fetching options quotes for {contract_ticker}")
        response = self._make_request(endpoint, params)
        
        if response.get('status') == 'error' or 'results' not in response:
            logger.error(f"Error fetching quotes for {contract_ticker}: {response.get('error', 'Unknown error')}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        quotes = response['results']
        if not quotes:
            logger.warning(f"No quotes found for {contract_ticker}")
            return pd.DataFrame()
        
        df = pd.DataFrame(quotes)
        
        # Process timestamps
        if 'sip_timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['sip_timestamp'], unit='ns')
        
        return df
    
    def get_market_news(self, 
                      ticker: Optional[str] = None, 
                      limit: int = 100, 
                      order: str = 'desc', 
                      from_date: Optional[str] = None, 
                      to_date: Optional[str] = None) -> pd.DataFrame:
        """
        Get market news articles.
        
        Args:
            ticker: Stock ticker symbol (optional, for filtering news by ticker)
            limit: Maximum number of news articles to return
            order: Sort order ('asc' or 'desc')
            from_date: Start date in YYYY-MM-DD format
            to_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with news articles
        """
        endpoint = f"/v2/reference/news"
        params = {
            "limit": limit,
            "order": order,
        }
        
        if ticker:
            params['ticker'] = ticker
        
        if from_date:
            params['published_utc.gte'] = f"{from_date}T00:00:00Z"
        
        if to_date:
            params['published_utc.lte'] = f"{to_date}T23:59:59Z"
        
        logger.info(f"Fetching market news {'for ' + ticker if ticker else ''}")
        response = self._make_request(endpoint, params)
        
        if response.get('status') == 'error' or 'results' not in response:
            logger.error(f"Error fetching news: {response.get('error', 'Unknown error')}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        news = response['results']
        if not news:
            logger.warning("No news articles found")
            return pd.DataFrame()
        
        df = pd.DataFrame(news)
        
        # Process timestamps
        if 'published_utc' in df.columns:
            df['published_datetime'] = pd.to_datetime(df['published_utc'])
        
        return df
    
    def get_related_companies(self, ticker: str) -> Dict[str, Any]:
        """
        Get companies related to the given ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with related companies information
        """
        endpoint = f"/v1/reference/related-companies/{ticker}"
        
        logger.info(f"Fetching related companies for {ticker}")
        response = self._make_request(endpoint)
        
        if response.get('status') == 'error' or 'results' not in response:
            logger.error(f"Error fetching related companies for {ticker}: {response.get('error', 'Unknown error')}")
            return {'status': 'error'}
        
        return response['results']
    
    def get_ticker_details(self, ticker: str) -> Dict[str, Any]:
        """
        Get detailed information about a ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with ticker details
        """
        endpoint = f"/v3/reference/tickers/{ticker}"
        
        logger.info(f"Fetching details for {ticker}")
        response = self._make_request(endpoint)
        
        if response.get('status') == 'error' or 'results' not in response:
            logger.error(f"Error fetching details for {ticker}: {response.get('error', 'Unknown error')}")
            return {'status': 'error'}
        
        return response['results']
    
    def get_market_status(self) -> Dict[str, Any]:
        """
        Get current market status (open/closed).
        
        Returns:
            Dictionary with market status information
        """
        endpoint = "/v1/marketstatus/now"
        
        logger.info("Fetching market status")
        response = self._make_request(endpoint)
        
        if response.get('status') == 'error':
            logger.error(f"Error fetching market status: {response.get('error', 'Unknown error')}")
            return {'status': 'error'}
        
        return response
    
    def get_stock_splits(self, ticker: str) -> pd.DataFrame:
        """
        Get stock splits for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            DataFrame with stock split information
        """
        endpoint = f"/v3/reference/splits"
        params = {
            "ticker": ticker
        }
        
        logger.info(f"Fetching stock splits for {ticker}")
        response = self._make_request(endpoint, params)
        
        if response.get('status') == 'error' or 'results' not in response:
            logger.error(f"Error fetching splits for {ticker}: {response.get('error', 'Unknown error')}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        splits = response['results']
        if not splits:
            logger.warning(f"No splits found for {ticker}")
            return pd.DataFrame()
        
        return pd.DataFrame(splits)
    
    def get_stock_dividends(self, ticker: str) -> pd.DataFrame:
        """
        Get dividend information for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            DataFrame with dividend information
        """
        endpoint = f"/v3/reference/dividends"
        params = {
            "ticker": ticker
        }
        
        logger.info(f"Fetching dividends for {ticker}")
        response = self._make_request(endpoint, params)
        
        if response.get('status') == 'error' or 'results' not in response:
            logger.error(f"Error fetching dividends for {ticker}: {response.get('error', 'Unknown error')}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        dividends = response['results']
        if not dividends:
            logger.warning(f"No dividends found for {ticker}")
            return pd.DataFrame()
        
        return pd.DataFrame(dividends)
    
    def get_market_holidays(self) -> pd.DataFrame:
        """
        Get market holidays.
        
        Returns:
            DataFrame with market holidays
        """
        endpoint = "/v1/marketstatus/upcoming"
        
        logger.info("Fetching market holidays")
        response = self._make_request(endpoint)
        
        if response.get('status') == 'error':
            logger.error(f"Error fetching market holidays: {response.get('error', 'Unknown error')}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        holidays = response
        if not holidays:
            logger.warning("No market holidays found")
            return pd.DataFrame()
        
        return pd.DataFrame(holidays)
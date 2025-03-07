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
        
        # Initialize params if None
        if params is None:
            params = {}
            
        # Add API key to query parameters
        params['apiKey'] = self.api_key
        
        # Set up headers
        headers = {
            'Accept': 'application/json'
        }
        
        try:
            logger.debug(f"Making request to {url} with params: {params}")
            response = self.session.get(url, params=params, headers=headers)
            
            # Log response headers for debugging
            logger.debug(f"Response headers: {dict(response.headers)}")
            
            # Update rate limit information
            if 'X-RateLimit-Remaining' in response.headers:
                self.rate_limit_remaining = int(response.headers['X-RateLimit-Remaining'])
                logger.debug(f"Rate limit remaining: {self.rate_limit_remaining}")
            if 'X-RateLimit-Reset' in response.headers:
                self.rate_limit_reset = int(response.headers['X-RateLimit-Reset'])
            
            # Handle HTTP errors
            if response.status_code != 200:
                logger.error(f"HTTP Error {response.status_code}: {response.text}")
                return {'status': 'error', 'error': f"HTTP Error {response.status_code}: {response.text}"}
            
            # Log raw response for debugging
            logger.debug(f"Raw response: {response.text[:1000]}")  # First 1000 chars
            
            data = response.json()
            
            # Check for API errors
            if data.get('status') == 'ERROR':
                logger.error(f"API Error: {data}")
                return {'status': 'error', 'error': str(data)}
            elif data.get('status') != 'OK':
                logger.warning(f"Unexpected response status: {data.get('status')}")
            
            return data
        
        except requests.RequestException as e:
            logger.error(f"Request error: {e}")
            return {'status': 'error', 'error': str(e)}
        except ValueError as e:
            logger.error(f"JSON decode error: {e}")
            logger.error(f"Response content: {response.text[:1000]}")  # First 1000 chars
            return {'status': 'error', 'error': 'Invalid JSON response'}
    
    def get_stock_bars(self, 
                      ticker: str, 
                      timespan: str = 'hour', 
                      from_date: str = None, 
                      to_date: str = None, 
                      limit: int = 5000) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        获取股票历史价格数据。
        
        Args:
            ticker: 股票代码
            timespan: 时间间隔 ('minute', 'hour', 'day', 'week', 'month', 'quarter', 'year')
            from_date: 开始日期，格式 YYYY-MM-DD
            to_date: 结束日期，格式 YYYY-MM-DD
            limit: 返回的最大数据条数（默认 5000，最大 50000）
            
        Returns:
            Tuple 包含:
            - DataFrame 包含 OHLCV 数据
            - Dictionary 包含元数据和可序列化数据
        """
        # 处理日期
        current_date = datetime.now().date()
        
        # 如果没有提供结束日期，使用当前日期的前一天
        if not to_date:
            to_date = (current_date - timedelta(days=1)).strftime('%Y-%m-%d')
        else:
            # 确保结束日期不超过当前日期
            to_date_obj = datetime.strptime(to_date, '%Y-%m-%d').date()
            if to_date_obj > current_date:
                to_date = (current_date - timedelta(days=1)).strftime('%Y-%m-%d')
                logger.warning(f"Adjusted to_date to yesterday as it was in the future: {to_date}")
        
        # 如果没有提供开始日期，使用结束日期前7天（对于小时数据来说更合理）
        if not from_date:
            from_date = (datetime.strptime(to_date, '%Y-%m-%d') - timedelta(days=7)).strftime('%Y-%m-%d')
        
        # 构建 endpoint
        endpoint = f"/v2/aggs/ticker/{ticker}/range/1/{timespan}/{from_date}/{to_date}"
        
        # 设置参数
        params = {
            "adjusted": "true",
            "sort": "asc",
            "limit": limit
        }
        
        logger.info(f"Fetching {timespan} data for {ticker} from {from_date} to {to_date}")
        response = self._make_request(endpoint, params)
        
        # 准备元数据字典
        metadata = {
            'ticker': ticker,
            'timespan': timespan,
            'from_date': from_date,
            'to_date': to_date,
            'status': 'success',
            'error': None,
            'count': 0,
            'request_params': params,
            'timestamp': datetime.now().isoformat(),
            'query_count': response.get('queryCount', 0),
            'results_count': response.get('resultsCount', 0)
        }
        
        # 处理错误响应
        if response.get('status') != 'OK':
            error_msg = response.get('error', 'Unknown error')
            logger.error(f"Error fetching bars for {ticker}: {error_msg}")
            metadata.update({
                'status': 'error',
                'error': error_msg
            })
            return pd.DataFrame(), metadata
        
        # 检查响应中是否有结果
        results = response.get('results', [])
        if not results:
            logger.warning(f"No data retrieved for {ticker}")
            metadata.update({
                'status': 'success',
                'count': 0
            })
            return pd.DataFrame(), metadata
        
        try:
            # 转换为 DataFrame
            df = pd.DataFrame(results)
            
            # 重命名列
            column_map = {
                't': 'timestamp',  # Unix 毫秒时间戳
                'o': 'open',      # 开盘价
                'h': 'high',      # 最高价
                'l': 'low',       # 最低价
                'c': 'close',     # 收盘价
                'v': 'volume',    # 成交量
                'vw': 'vwap',     # 成交量加权平均价
                'n': 'trades'     # 交易次数
            }
            df.rename(columns=column_map, inplace=True)
            
            # 转换时间戳
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['date'] = df['timestamp'].dt.date
            df['time'] = df['timestamp'].dt.time
            
            # 按时间戳排序
            df.sort_values('timestamp', inplace=True)
            
            # 准备可序列化数据
            serializable_data = df.to_dict(orient='records')
            
            # 更新元数据
            metadata.update({
                'status': 'success',
                'count': len(df),
                'first_timestamp': df['timestamp'].min().isoformat() if not df.empty else None,
                'last_timestamp': df['timestamp'].max().isoformat() if not df.empty else None,
                'columns': df.columns.tolist(),
                'shape': df.shape,
                'data': serializable_data
            })
            
            logger.info(f"Successfully retrieved {len(df)} {timespan} bars for {ticker}")
            return df, metadata
            
        except Exception as e:
            error_msg = f"Error processing data for {ticker}: {str(e)}"
            logger.error(error_msg)
            metadata.update({
                'status': 'error',
                'error': error_msg
            })
            return pd.DataFrame(), metadata
    
    def get_multiple_stocks_bars(self, 
                               tickers: List[str], 
                               timespan: str = 'day', 
                               from_date: str = None, 
                               to_date: str = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Get historical price bars for multiple stocks.
        
        Args:
            tickers: List of stock ticker symbols
            timespan: Time span of the bars ('minute', 'hour', 'day', 'week', 'month')
            from_date: Start date in YYYY-MM-DD format
            to_date: End date in YYYY-MM-DD format
            
        Returns:
            Tuple containing:
            - DataFrame with OHLCV data for all tickers
            - Dictionary with metadata and results for all tickers
        """
        all_dfs = []
        all_metadata = {
            'tickers': tickers,
            'timespan': timespan,
            'from_date': from_date,
            'to_date': to_date,
            'timestamp': datetime.now().isoformat(),
            'results': {},
            'status': 'success',
            'error': None,
            'total_count': 0
        }
        
        for ticker in tickers:
            df, metadata = self.get_stock_bars(ticker, timespan, from_date, to_date)
            all_metadata['results'][ticker] = metadata
            
            if not df.empty:
                df['ticker'] = ticker
                all_dfs.append(df)
                all_metadata['total_count'] += len(df)
            
            # Update overall status if any ticker fails
            if metadata['status'] == 'error':
                all_metadata['status'] = 'partial_error'
                if all_metadata['error'] is None:
                    all_metadata['error'] = {}
                all_metadata['error'][ticker] = metadata['error']
            
            # Slight delay to avoid rate limiting
            time.sleep(0.1)
        
        if not all_dfs:
            all_metadata['status'] = 'error' if all_metadata['status'] != 'partial_error' else all_metadata['status']
            return pd.DataFrame(), all_metadata
        
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        # Add summary statistics to metadata
        all_metadata.update({
            'total_tickers': len(tickers),
            'successful_tickers': len(all_dfs),
            'failed_tickers': len(tickers) - len(all_dfs),
            'total_records': len(combined_df),
            'date_range': {
                'start': combined_df['timestamp'].min().isoformat() if not combined_df.empty else None,
                'end': combined_df['timestamp'].max().isoformat() if not combined_df.empty else None
            }
        })
        
        return combined_df, all_metadata
    
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
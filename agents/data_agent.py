import logging
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import os
import json
from typing import Dict, List, Optional, Any, Union
import threading

# Add parent directory to path for imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import custom modules
from utils.data.polygon_api import PolygonAPI
from utils.communication.message import Message, MessageType, create_data_message, create_broadcast_message
from agents.base_agent import BaseAgent

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataAgent(BaseAgent):
    """
    Agent responsible for fetching and preprocessing market data.
    Uses Polygon API for real-time and historical data.
    """
    
    def __init__(self, 
                agent_id: str,
                communicator,
                api_key: Optional[str] = None,
                config_path: Optional[str] = None):
        """
        Initialize the DataAgent.
        
        Args:
            agent_id: Unique identifier for this agent
            communicator: Communication manager for inter-agent messaging
            api_key: Polygon.io API key (optional, will check environment variable if None)
            config_path: Path to configuration file
        """
        super().__init__(agent_id, communicator)
        self.config = self._load_config(config_path)
        
        # Initialize Polygon API client
        self.api = PolygonAPI(api_key)
        
        # Data storage
        self.stock_data = {}
        self.options_data = {}
        self.news_data = pd.DataFrame()
        self.market_status = {}
        
        # Threading control
        self.running = False
        self.update_thread = None
        self.data_lock = threading.RLock()  # Reentrant lock for thread safety
        
        # Register request handlers
        self._register_handlers()
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """
        Load configuration from file or use defaults.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        default_config = {
            'default_tickers': ['SPY', 'QQQ', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META'],
            'update_interval': 60,  # seconds
            'data_history_days': 90,
            'news_limit': 100,
            'options_enabled': True,
            'default_timespan': 'day',
            'data_storage': './data/market_data'
        }
        
        if not config_path:
            return default_config
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                return {**default_config, **config}  # Merge with defaults
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return default_config
    
    def _register_handlers(self):
        """
        Register handlers for incoming requests.
        """
        if self.communicator:
            # Register handlers for specific request types
            self.communicator.register_request_handler(
                self.agent_id, "get_stock_data", self._handle_stock_data_request)
            self.communicator.register_request_handler(
                self.agent_id, "get_options_data", self._handle_options_data_request)
            self.communicator.register_request_handler(
                self.agent_id, "get_market_news", self._handle_news_request)
    
    def start(self):
        """
        Start the data agent's background data updating thread.
        """
        if self.running:
            logger.warning("Data agent is already running")
            return
        
        self.running = True
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
        logger.info(f"{self.agent_id} started successfully")
    
    def stop(self):
        """
        Stop the data agent's background thread.
        """
        self.running = False
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=5)
        logger.info(f"{self.agent_id} stopped")
    
    def fetch_stock_data(self, 
                       tickers: Optional[List[str]] = None, 
                       timespan: Optional[str] = None, 
                       days_back: Optional[int] = None):
        """
        Fetch historical stock data for the given tickers.
        
        Args:
            tickers: List of ticker symbols (defaults to config value)
            timespan: Time interval ('minute', 'hour', 'day', etc.)
            days_back: Number of days of history to fetch
            
        Returns:
            Dictionary with ticker symbols as keys and DataFrames as values
        """
        tickers = tickers or self.config['default_tickers']
        timespan = timespan or self.config['default_timespan']
        days_back = days_back or self.config['data_history_days']
        
        to_date = datetime.now().strftime('%Y-%m-%d')
        from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        stock_data = {}
        
        try:
            for ticker in tickers:
                logger.info(f"Fetching {timespan} data for {ticker} from {from_date} to {to_date}")
                df = self.api.get_stock_bars(
                    ticker=ticker, 
                    timespan=timespan, 
                    from_date=from_date, 
                    to_date=to_date
                )
                
                if not df.empty:
                    stock_data[ticker] = df
                    logger.info(f"Fetched {len(df)} bars for {ticker}")
                else:
                    logger.warning(f"No data retrieved for {ticker}")
                
                # Small delay to avoid rate limiting
                time.sleep(0.2)
            
            # Update the stored data
            with self.data_lock:
                for ticker, data in stock_data.items():
                    self.stock_data[ticker] = data
            
            # Notify other agents about new data
            self._notify_data_update('stock_data_updated')
            
            return stock_data
            
        except Exception as e:
            logger.error(f"Error fetching stock data: {e}")
            return {}
    
    def fetch_options_data(self, 
                         tickers: Optional[List[str]] = None, 
                         expiration_days: Optional[int] = None):
        """
        Fetch options data for the given tickers.
        
        Args:
            tickers: List of underlying ticker symbols
            expiration_days: Max days to expiration to include
            
        Returns:
            Dictionary with ticker symbols as keys and options DataFrames as values
        """
        if not self.config['options_enabled']:
            logger.info("Options data fetching is disabled in config")
            return {}
        
        tickers = tickers or self.config['default_tickers']
        
        # Calculate expiration cutoff if specified
        expiration_cutoff = None
        if expiration_days:
            expiration_cutoff = (datetime.now() + timedelta(days=expiration_days)).strftime('%Y-%m-%d')
        
        options_data = {}
        
        try:
            for ticker in tickers:
                logger.info(f"Fetching options data for {ticker}")
                
                # Get available options contracts
                contracts_df = self.api.get_options_contracts(
                    underlying_ticker=ticker, 
                    limit=100  # Adjust based on needs
                )
                
                if not contracts_df.empty:
                    # Filter by expiration if needed
                    if expiration_cutoff and 'expiration_date' in contracts_df.columns:
                        contracts_df = contracts_df[contracts_df['expiration_date'] <= expiration_cutoff]
                    
                    options_data[ticker] = contracts_df
                    logger.info(f"Fetched {len(contracts_df)} options contracts for {ticker}")
                else:
                    logger.warning(f"No options contracts found for {ticker}")
                
                # Small delay to avoid rate limiting
                time.sleep(0.2)
            
            # Update the stored data
            with self.data_lock:
                for ticker, data in options_data.items():
                    self.options_data[ticker] = data
            
            # Notify other agents about new data
            self._notify_data_update('options_data_updated')
            
            return options_data
            
        except Exception as e:
            logger.error(f"Error fetching options data: {e}")
            return {}
    
    def fetch_market_news(self, tickers: Optional[List[str]] = None, days_back: int = 7, limit: int = 100):
        """
        Fetch market news for the given tickers.
        
        Args:
            tickers: List of ticker symbols (None for general market news)
            days_back: Number of days of news to fetch
            limit: Maximum number of news items to fetch per ticker
            
        Returns:
            DataFrame with news articles
        """
        from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        all_news = []
        
        try:
            # Fetch general market news if no tickers specified
            if not tickers:
                logger.info("Fetching general market news")
                news_df = self.api.get_market_news(
                    limit=limit,
                    from_date=from_date
                )
                
                if not news_df.empty:
                    all_news.append(news_df)
                
            else:  # Fetch news for each ticker
                for ticker in tickers:
                    logger.info(f"Fetching news for {ticker}")
                    news_df = self.api.get_market_news(
                        ticker=ticker,
                        limit=limit // len(tickers),  # Distribute limit across tickers
                        from_date=from_date
                    )
                    
                    if not news_df.empty:
                        all_news.append(news_df)
                    
                    # Small delay to avoid rate limiting
                    time.sleep(0.2)
            
            # Combine all news
            if all_news:
                combined_news = pd.concat(all_news, ignore_index=True)
                
                # Remove duplicates
                combined_news.drop_duplicates(subset=['id'], keep='first', inplace=True)
                
                # Sort by published date (newest first)
                if 'published_datetime' in combined_news.columns:
                    combined_news.sort_values('published_datetime', ascending=False, inplace=True)
                
                # Update the stored news data
                with self.data_lock:
                    self.news_data = combined_news
                
                # Notify other agents about new data
                self._notify_data_update('news_data_updated')
                
                return combined_news
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error fetching market news: {e}")
            return pd.DataFrame()
    
    def fetch_related_companies(self, ticker: str):
        """
        Fetch companies related to the given ticker and their correlations.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with related companies data
        """
        try:
            logger.info(f"Fetching related companies for {ticker}")
            related_data = self.api.get_related_companies(ticker)
            
            if related_data and related_data.get('status') != 'error':
                # Save related companies data
                with self.data_lock:
                    if 'related_companies' not in self.__dict__:
                        self.related_companies = {}
                    self.related_companies[ticker] = related_data
                    
                # Fetch price data for correlation analysis
                if 'similar' in related_data and related_data['similar']:
                    similar_tickers = [ticker] + related_data['similar'][:5]  # Original + top 5 similar
                    self.fetch_stock_data(tickers=similar_tickers, days_back=90)  # 3 months of daily data
                
                return related_data
            
            return {}
            
        except Exception as e:
            logger.error(f"Error fetching related companies: {e}")
            return {}
    
    def fetch_market_status(self):
        """
        Fetch current market status (open/closed).
        
        Returns:
            Dictionary with market status information
        """
        try:
            logger.info("Fetching market status")
            status = self.api.get_market_status()
            
            if status and status.get('status') != 'error':
                # Update stored market status
                with self.data_lock:
                    self.market_status = status
                
                # Notify other agents
                self._notify_data_update('market_status_updated')
                
                return status
            
            return {}
            
        except Exception as e:
            logger.error(f"Error fetching market status: {e}")
            return {}
    
    def _update_loop(self):
        """
        Background thread for periodic data updates.
        """
        logger.info(f"{self.agent_id} update loop started")
        
        # Initial data fetch
        self.fetch_stock_data()
        self.fetch_market_news()
        self.fetch_market_status()
        
        if self.config['options_enabled']:
            self.fetch_options_data()
        
        last_news_update = datetime.now()
        last_options_update = datetime.now()
        
        while self.running:
            try:
                # Fetch stock data on each cycle
                self.fetch_stock_data()
                
                # Fetch market status
                self.fetch_market_status()
                
                # Fetch news less frequently
                time_since_news = (datetime.now() - last_news_update).total_seconds()
                if time_since_news > self.config['update_interval'] * 5:  # 5x less frequent
                    self.fetch_market_news()
                    last_news_update = datetime.now()
                
                # Fetch options less frequently
                if self.config['options_enabled']:
                    time_since_options = (datetime.now() - last_options_update).total_seconds()
                    if time_since_options > self.config['update_interval'] * 10:  # 10x less frequent
                        self.fetch_options_data()
                        last_options_update = datetime.now()
                
                # Process incoming messages
                self._process_messages()
                
                # Sleep until next update cycle
                time.sleep(self.config['update_interval'])
                
            except Exception as e:
                logger.error(f"Error in update loop: {e}")
                time.sleep(10)  # Wait before retrying after error
        
        logger.info(f"{self.agent_id} update loop stopped")
    
    def _process_messages(self):
        """
        Process incoming messages from other agents.
        """
        if not self.communicator:
            return
        
        messages = self.communicator.get_messages(self.agent_id)
        
        for message in messages:
            try:
                # Handle message based on type
                if message.message_type == MessageType.REQUEST:
                    self._handle_request(message)
                
            except Exception as e:
                logger.error(f"Error processing message: {e}")
    
    def _handle_request(self, message: Message):
        """
        Handle request messages from other agents.
        """
        # Request handling is done via registered handlers
        pass
    
    def _handle_stock_data_request(self, message: Message):
        """
        Handle a request for stock data.
        
        Args:
            message: Request message
            
        Returns:
            Stock data to be sent in response
        """
        content = message.content
        tickers = content.get('tickers', self.config['default_tickers'])
        refresh = content.get('refresh', False)
        timespan = content.get('timespan', self.config['default_timespan'])
        days_back = content.get('days_back', self.config['data_history_days'])
        
        # Refresh data if requested
        if refresh:
            self.fetch_stock_data(tickers=tickers, timespan=timespan, days_back=days_back)
        
        # Get the requested data
        with self.data_lock:
            result = {}
            for ticker in tickers:
                if ticker in self.stock_data:
                    result[ticker] = self.stock_data[ticker].copy()
        
        return result
    
    def _handle_options_data_request(self, message: Message):
        """
        Handle a request for options data.
        
        Args:
            message: Request message
            
        Returns:
            Options data to be sent in response
        """
        if not self.config['options_enabled']:
            return {'error': 'Options data is disabled'}
        
        content = message.content
        tickers = content.get('tickers', self.config['default_tickers'])
        refresh = content.get('refresh', False)
        expiration_days = content.get('expiration_days')
        
        # Refresh data if requested
        if refresh:
            self.fetch_options_data(tickers=tickers, expiration_days=expiration_days)
        
        # Get the requested data
        with self.data_lock:
            result = {}
            for ticker in tickers:
                if ticker in self.options_data:
                    result[ticker] = self.options_data[ticker].copy()
        
        return result
    
    def _handle_news_request(self, message: Message):
        """
        Handle a request for market news.
        
        Args:
            message: Request message
            
        Returns:
            News data to be sent in response
        """
        content = message.content
        tickers = content.get('tickers')
        refresh = content.get('refresh', False)
        days_back = content.get('days_back', 7)
        limit = content.get('limit', self.config['news_limit'])
        
        # Refresh data if requested
        if refresh:
            self.fetch_market_news(tickers=tickers, days_back=days_back, limit=limit)
        
        # Get the news data
        with self.data_lock:
            # Filter by tickers if specified
            if tickers and 'tickers' in self.news_data.columns:
                result = self.news_data[self.news_data['tickers'].apply(
                    lambda x: any(ticker in x for ticker in tickers) if isinstance(x, list) else False
                )].copy()
            else:
                result = self.news_data.copy()
        
        return result.to_dict(orient='records') if not result.empty else []
    
    def _notify_data_update(self, update_type: str):
        """
        Notify other agents about a data update.
        
        Args:
            update_type: Type of update (e.g., 'stock_data_updated')
        """
        if not self.communicator:
            return
        
        # Update central data store
        if update_type == 'stock_data_updated':
            with self.data_lock:
                # Combine all stock data into a single DataFrame for convenience
                if self.stock_data:
                    combined_df = pd.DataFrame()
                    for ticker, df in self.stock_data.items():
                        if not df.empty:
                            ticker_df = df.copy()
                            ticker_df['ticker'] = ticker
                            combined_df = pd.concat([combined_df, ticker_df], ignore_index=True)
                    
                    self.communicator.update_data('stock_data', combined_df)
        
        # Send update notification
        broadcast_msg = create_broadcast_message(
            sender_id=self.agent_id,
            content={'update_type': update_type},
            metadata={'timestamp': datetime.now().isoformat()}
        )
        
        self.communicator.send_message(broadcast_msg)
import json
import logging
import os
import sys
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import yaml

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now import local modules
from utils.data.polygon_api import PolygonAPI
from utils.communication import MessageType
from utils.communication.message import Message
from agents.base_agent import BaseAgent

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _load_config(config_path: Optional[str]) -> Dict[str, Any]:
    """
    Load configuration from YAML file or use defaults.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    default_config = {
        'default_tickers': ['SPY', 'QQQ', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META'],
        'update_interval': 60,  # seconds
        'data_history_days': 7,  # 减少历史数据天数，因为我们使用小时数据
        'news_limit': 100,
        'options_enabled': False,  # Disable options data
        'default_timespan': 'hour',  # 修改默认时间间隔为小时
        'data_storage': './data/market_data',
        'output': {
            'results_dir': 'results'
        }
    }

    if not config_path:
        return default_config

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            return {**default_config, **config}  # Merge with defaults
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return default_config


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
            api_key: Polygon.io API key (required)
            config_path: Path to configuration file
        """
        super().__init__(agent_id, communicator)
        self.config = _load_config(config_path)
        
        # 设置交易模式
        self.mode = self.config.get('trading_mode', 'paper')  # 默认为 paper 模式
        logger.info(f"Trading mode: {self.mode}")

        # Initialize Polygon API client
        if not api_key:
            raise ValueError("Polygon API key is required for DataAgent")
        
        self.api = PolygonAPI(api_key)
        logger.info("Successfully initialized Polygon API client")

        # Data storage
        self.stock_data = {}
        self.options_data = {}
        self.news_data = pd.DataFrame()
        
        # Results storage
        self.results_dir = self.config.get('output', {}).get('results_dir', 'results')
        self.results = {
            'signals': {},
            'sentiment': {},
            'risk': {},
            'predictions': {},
            'market_data': {},
            'metadata': {
                'start_time': datetime.now().isoformat(),
                'last_update': datetime.now().isoformat(),
                'tickers': self.config.get('default_tickers', []),
                'config': self.config,
                'mode': self.mode
            }
        }

        # Threading control
        self.running = False
        self.update_thread = None
        self.data_lock = threading.RLock()  # Reentrant lock for thread safety

        # Register request handlers
        self._register_handlers()

    def _register_handlers(self):
        """
        Register handlers for incoming requests.
        """
        if self.communicator:
            # Register handlers for specific request types
            self.communicator.register_request_handler(
                self.agent_id, "get_stock_data", self._handle_stock_data_request)
            self.communicator.register_request_handler(
                self.agent_id, "get_market_news", self._handle_news_request)
            self.communicator.register_request_handler(
                self.agent_id, "store_agent_result", self._handle_store_result_request)

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

    def handle_message(self, message: Message) -> None:
        """
        Handle incoming messages.
        
        Args:
            message: Message object containing the message data
        """
        try:
            logger.info(f"DataAgent {self.agent_id} received message: {message.message_type} from {message.sender_id}")
            
            # 处理系统状态消息
            if message.message_type == MessageType.STATUS:
                if message.content.get('status') == 'shutdown_requested':
                    logger.info("Received shutdown request, stopping update loop")
                    self._stop_update_loop()
                return
            
            # 处理数据请求消息
            if message.message_type == MessageType.REQUEST:
                if message.content.get('action') == 'fetch_data':
                    tickers = message.content.get('tickers', self.config.get('data', {}).get('default_tickers', []))
                    timespan = message.content.get('timespan', self.config.get('data', {}).get('default_timespan', 'day'))
                    days_back = message.content.get('days_back', 90)
                    
                    logger.info(f"Processing data request for tickers: {tickers}")
                    self.fetch_stock_data(tickers, timespan, days_back)
                return
            
            # 处理结果存储请求
            if message.message_type == MessageType.RESULT:
                content = message.content
                agent_id = message.sender_id
                result_type = content.get('result_type')
                data = content.get('data')
                
                if result_type and data:
                    logger.info(f"Storing {result_type} results from {agent_id}")
                    self.store_agent_result(agent_id, result_type, data)
                return
            
            # 处理其他类型的消息
            logger.debug(f"Received message of type {message.message_type} from {message.sender_id}")
            
        except Exception as e:
            logger.error(f"Error handling message: {str(e)}", exc_info=True)
            # 使用 BaseAgent 的 send_message 方法发送错误响应
            self.send_message(
                receiver_id=message.sender_id,
                message_type=MessageType.ERROR,
                content={
                    'error': str(e),
                    'original_message': message.content
                }
            )

    def fetch_stock_data(self,
                         tickers: Optional[List[str]] = None,
                         timespan: Optional[str] = None,
                         days_back: Optional[int] = None):
        """
        获取股票数据并存储。
        
        Args:
            tickers: 股票代码列表，如果为 None 则使用配置中的默认股票
            timespan: 时间间隔 ('minute', 'hour', 'day', 'week', 'month')
            days_back: 获取多少天前的数据
        """
        try:
            # 使用默认值
            if tickers is None:
                tickers = self.config.get('default_tickers', [])
            if timespan is None:
                timespan = self.config.get('default_timespan', 'hour')  # 默认使用小时数据
            if days_back is None:
                days_back = self.config.get('data_history_days', 7)  # 默认获取7天数据
            
            # 计算日期范围
            end_date = datetime.now().date() - timedelta(days=1)  # 使用昨天作为结束日期
            start_date = end_date - timedelta(days=days_back)
            
            # 格式化日期
            start_date_str = start_date.strftime('%Y-%m-%d')
            end_date_str = end_date.strftime('%Y-%m-%d')
            
            logger.info(f"Fetching {timespan} data for {len(tickers)} tickers from {start_date_str} to {end_date_str}")
            
            # 存储所有数据和元数据
            all_data = {}
            all_metadata = {}
            failed_tickers = []
            
            for ticker in tickers:
                try:
                    with self.data_lock:
                        df, metadata = self.api.get_stock_bars(
                            ticker=ticker,
                            timespan=timespan,
                            from_date=start_date_str,
                            to_date=end_date_str
                        )
                        
                        if df.empty:
                            logger.warning(f"No data available for {ticker}")
                            failed_tickers.append(ticker)
                            continue
                        
                        # 存储数据
                        all_data[ticker] = df
                        all_metadata[ticker] = metadata
                        
                        # 更新内部存储
                        self.stock_data[ticker] = {
                            'data': df,
                            'metadata': metadata,
                            'last_update': datetime.now().isoformat()
                        }
                        
                        # 添加到结果存储
                        self.results['market_data'][ticker] = {
                            'data': metadata['data'],  # 使用已经序列化的数据
                            'metadata': {
                                'status': metadata['status'],
                                'count': metadata['count'],
                                'first_timestamp': metadata['first_timestamp'],
                                'last_timestamp': metadata['last_timestamp'],
                                'query_count': metadata.get('query_count', 0),
                                'results_count': metadata.get('results_count', 0),
                                'last_update': datetime.now().isoformat()
                            }
                        }
                        
                        # 短暂暂停以避免触发速率限制
                        time.sleep(0.1)
                        
                except Exception as e:
                    logger.error(f"Error fetching data for {ticker}: {str(e)}")
                    failed_tickers.append(ticker)
                    continue
            
            # 更新结果存储的元数据
            self.results['metadata'].update({
                'last_update': datetime.now().isoformat(),
                'tickers': list(all_data.keys()),
                'failed_tickers': failed_tickers,
                'timespan': timespan,
                'start_date': start_date_str,
                'end_date': end_date_str,
                'total_records': sum(len(df) for df in all_data.values()),
                'successful_tickers': len(all_data),
                'failed_tickers_count': len(failed_tickers)
            })
            
            # 保存结果到文件
            self._save_results()
            
            # 通知其他代理数据已更新
            if all_data:
                self._notify_data_update('stock_data_updated')
            
            return len(all_data), len(failed_tickers)
            
        except Exception as e:
            logger.error(f"Error in fetch_stock_data: {str(e)}", exc_info=True)
            raise

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

    def _update_loop(self):
        """
        后台线程用于定期更新数据。
        """
        logger.info(f"{self.agent_id} update loop started in {self.mode} mode")

        # 跟踪上次更新时间
        last_stock_update = datetime.now()
        last_news_update = datetime.now()

        # 初始数据获取
        self.fetch_stock_data()
        self.fetch_market_news()

        while self.running:
            try:
                current_time = datetime.now()
                
                # 根据模式设置更新间隔
                stock_update_interval = (
                    self.config['update_interval'] 
                    if self.mode == 'live' 
                    else self.config.get('paper_update_interval', 300)  # 纸面交易默认5分钟
                )
                
                # 新闻更新间隔（股票数据更新间隔的5倍）
                news_update_interval = stock_update_interval * 5

                # 检查是否需要更新股票数据
                if (current_time - last_stock_update).total_seconds() >= stock_update_interval:
                    self.fetch_stock_data()
                    last_stock_update = current_time

                # 检查是否需要更新新闻
                if (current_time - last_news_update).total_seconds() >= news_update_interval:
                    self.fetch_market_news()
                    last_news_update = current_time

                # 处理消息
                self._process_messages()

                # 短暂休眠以避免过度CPU使用
                time.sleep(1)

            except Exception as e:
                logger.error(f"Error in update loop: {e}")
                time.sleep(10)  # 错误后等待10秒再重试

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
                # 直接处理消息，因为 get_messages 已经返回了 Message 对象
                self.handle_message(message)

            except Exception as e:
                logger.error(f"Error processing message: {e}")
                import traceback
                logger.error(traceback.format_exc())

    def _handle_request(self, message):
        """
        Handle request messages from other agents.
        """
        # Request handling is done via registered handlers
        pass

    def _handle_stock_data_request(self, message):
        """
        Handle a request for stock data.

        Args:
            message: Request message

        Returns:
            Stock data to be sent in response
        """
        if not isinstance(message, Message):
            logger.error(f"Invalid message type in stock data request: {type(message)}")
            return {"error": "Invalid message format"}

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

    def _handle_news_request(self, message):
        """
        Handle a request for market news.

        Args:
            message: Request message

        Returns:
            News data to be sent in response
        """
        if not isinstance(message, Message):
            logger.error(f"Invalid message type in news request: {type(message)}")
            return {"error": "Invalid message format"}

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

    def _handle_store_result_request(self, message):
        """
        Handle a request to store results from other agents.

        Args:
            message: Request message

        Returns:
            Response message
        """
        if not isinstance(message, Message):
            logger.error(f"Invalid message type in store result request: {type(message)}")
            return {"error": "Invalid message format"}

        content = message.content
        agent_id = message.sender_id
        result_type = content.get('result_type')
        data = content.get('data')
        
        if result_type and data:
            logger.info(f"Storing {result_type} results from {agent_id}")
            self.store_agent_result(agent_id, result_type, data)
            return {"status": "success", "message": f"Results from {agent_id} stored successfully"}
        else:
            logger.error("Invalid result type or data in store result request")
            return {"error": "Invalid result type or data"}

    def _notify_data_update(self, update_type: str):
        """
        通知其他代理数据已更新。
        
        Args:
            update_type: 更新类型 ('stock_data_updated', 'market_status_updated', etc.)
        """
        try:
            if not self.communicator:
                logger.warning("No communicator available for notifications")
                return
            
            # 对于股票数据更新，发布到不同的 topics
            if update_type == 'stock_data_updated':
                # 合并所有股票数据到一个 DataFrame
                all_data = []
                for ticker_data in self.stock_data.values():
                    if isinstance(ticker_data, dict) and 'data' in ticker_data:
                        df = ticker_data['data']
                        if isinstance(df, pd.DataFrame):
                            all_data.append(df)
                
                if all_data:
                    combined_data = pd.concat(all_data, axis=0)
                    # 将 DataFrame 转换为可序列化的格式
                    serialized_data = combined_data.to_dict('records')
                    
                    # 创建基础消息内容
                    base_content = {
                        'data': serialized_data,
                        'tickers': list(self.stock_data.keys()),
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    # 发送到不同的 topics 以触发不同的分析
                    topics = {
                        'market_data': 'market_data_agent',
                        'sentiment_analysis': 'sentiment_agent',
                        'risk_analysis': 'risk_agent',
                        'signal_generation': 'signal_agent',
                        'prediction': 'prediction_agent'
                    }
                    
                    for topic, receiver in topics.items():
                        message = Message(
                            sender_id=self.agent_id,
                            receiver_id=receiver,
                            message_type=MessageType.DATA,
                            content={
                                **base_content,
                                'analysis_type': topic
                            }
                        )
                        
                        # 发布到特定 topic
                        self.communicator.publish(topic, message)
                        logger.info(f"Published market data update to {topic}")
            
            # 创建通知消息
            notification_message = Message(
                sender_id=self.agent_id,
                receiver_id="all",
                message_type=MessageType.NOTIFICATION,
                content={
                    'event_type': 'data_update',
                    'update_type': update_type,
                    'agent_id': self.agent_id,
                    'timestamp': datetime.now().isoformat()
                }
            )
            
            # 发送通知
            self.communicator.send_message(notification_message)
            
        except Exception as e:
            logger.error(f"Error in _notify_data_update: {str(e)}")
            logger.debug(f"Error details:", exc_info=True)

    def store_agent_result(self, agent_id: str, result_type: str, data: Dict[str, Any]) -> None:
        """
        存储并整合来自其他代理的分析结果。
        
        Args:
            agent_id: 产生结果的代理ID
            result_type: 结果类型 (signals, sentiment, risk, predictions)
            data: 结果数据
        """
        try:
            with self.data_lock:
                # 确保结果类型存在
                if result_type not in self.results:
                    self.results[result_type] = {
                        'latest': {},  # 最新的结果
                        'history': {},  # 历史结果
                        'metadata': {
                            'last_update': datetime.now().isoformat(),
                            'total_updates': 0,
                            'contributing_agents': set()
                        }
                    }
                
                # 更新结果
                timestamp = datetime.now().isoformat()
                
                # 存储最新结果
                self.results[result_type]['latest'][agent_id] = {
                    'data': data,
                    'timestamp': timestamp
                }
                
                # 添加到历史记录
                if agent_id not in self.results[result_type]['history']:
                    self.results[result_type]['history'][agent_id] = []
                
                self.results[result_type]['history'][agent_id].append({
                    'data': data,
                    'timestamp': timestamp
                })
                
                # 限制历史记录大小
                max_history = 100  # 可以从配置中读取
                if len(self.results[result_type]['history'][agent_id]) > max_history:
                    self.results[result_type]['history'][agent_id] = (
                        self.results[result_type]['history'][agent_id][-max_history:]
                    )
                
                # 更新元数据
                self.results[result_type]['metadata'].update({
                    'last_update': timestamp,
                    'total_updates': self.results[result_type]['metadata']['total_updates'] + 1
                })
                self.results[result_type]['metadata']['contributing_agents'].add(agent_id)
                
                # 更新总体元数据
                self.results['metadata'].update({
                    'last_update': timestamp,
                    'active_agents': list(set(
                        agent_id for r in self.results.values() 
                        if isinstance(r, dict) and 'metadata' in r 
                        for agent_id in r.get('metadata', {}).get('contributing_agents', set())
                    ))
                })
                
                # 保存结果到文件
                self._save_results()
                
                logger.info(f"Stored {result_type} results from {agent_id}")
                
                # 检查是否所有预期的代理都已提供结果
                self._check_analysis_completion(result_type)
                
        except Exception as e:
            logger.error(f"Error storing results from {agent_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _check_analysis_completion(self, result_type: str) -> None:
        """
        检查特定类型的分析是否已收到所有预期代理的结果。
        
        Args:
            result_type: 结果类型
        """
        try:
            # 预期的代理列表（可以从配置中读取）
            expected_agents = {
                'sentiment': ['sentiment_agent'],
                'risk': ['risk_agent'],
                'signals': ['signal_agent'],
                'predictions': ['prediction_agent']
            }
            
            if result_type in expected_agents:
                current_agents = set(self.results[result_type]['latest'].keys())
                expected = set(expected_agents[result_type])
                
                if current_agents >= expected:  # 所有预期的代理都已提供结果
                    logger.info(f"Received all expected results for {result_type}")
                    
                    # 可以在这里触发进一步的处理或通知
                    self._notify_analysis_completion(result_type)
        
        except Exception as e:
            logger.error(f"Error checking analysis completion: {e}")
    
    def _notify_analysis_completion(self, result_type: str) -> None:
        """
        通知分析完成。
        
        Args:
            result_type: 完成的分析类型
        """
        try:
            if self.communicator:
                message = Message(
                    sender_id=self.agent_id,
                    receiver_id="all",
                    message_type=MessageType.NOTIFICATION,
                    content={
                        'event_type': 'analysis_complete',
                        'analysis_type': result_type,
                        'timestamp': datetime.now().isoformat(),
                        'results_available': True
                    }
                )
                
                self.communicator.send_message(message)
                logger.info(f"Notified analysis completion for {result_type}")
        
        except Exception as e:
            logger.error(f"Error notifying analysis completion: {e}")

    def _save_results(self) -> None:
        """
        保存结果到文件。使用固定文件名，并保留最近的备份。
        """
        try:
            # 确保目录存在
            os.makedirs(self.results_dir, exist_ok=True)
            
            # 使用固定的文件名
            main_file = os.path.join(self.results_dir, "market_data_latest.json")
            backup_file = os.path.join(self.results_dir, "market_data_backup.json")
            
            # 如果主文件存在，先将其备份
            if os.path.exists(main_file):
                try:
                    if os.path.exists(backup_file):
                        os.remove(backup_file)
                    os.rename(main_file, backup_file)
                except Exception as e:
                    logger.warning(f"Error backing up results file: {e}")
            
            # 转换 DataFrame 和 Timestamp 为可序列化的格式
            serializable_results = self.results.copy()
            for ticker, data in serializable_results.get('market_data', {}).items():
                if isinstance(data, dict) and 'data' in data and isinstance(data['data'], pd.DataFrame):
                    serializable_results['market_data'][ticker]['data'] = data['data'].to_dict('records')
            
            # 保存新的结果
            with open(main_file, 'w') as f:
                json.dump(serializable_results, f, indent=2, default=str)
            
            logger.info(f"Results saved to {main_file}")
            
            # 清理旧的时间戳文件
            self._cleanup_old_results()
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
    
    def _cleanup_old_results(self):
        """
        清理旧的结果文件，只保留最新的文件。
        """
        try:
            # 获取目录中的所有结果文件
            result_files = [f for f in os.listdir(self.results_dir) 
                          if f.startswith("market_data_results_")]
            
            # 如果存在旧的时间戳文件，删除它们
            for old_file in result_files:
                try:
                    os.remove(os.path.join(self.results_dir, old_file))
                    logger.debug(f"Cleaned up old result file: {old_file}")
                except Exception as e:
                    logger.warning(f"Error deleting old result file {old_file}: {e}")
                    
        except Exception as e:
            logger.warning(f"Error during cleanup of old results: {e}")

    def _create_dummy_data_provider(self):
        """创建一个用于测试的虚拟数据提供者。"""

        class DummyDataProvider:
            def get_stock_bars(self, ticker, timespan='hour', from_date=None, to_date=None, 
                             limit=5000, **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
                """
                返回虚拟的股票数据。
                
                Args:
                    ticker: 股票代码
                    timespan: 时间间隔
                    from_date: 开始日期
                    to_date: 结束日期
                    limit: 数据条数限制
                """
                # 创建空的 DataFrame 和元数据
                df = pd.DataFrame(columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 
                    'volume', 'vwap', 'trades'
                ])
                
                metadata = {
                    'ticker': ticker,
                    'timespan': timespan,
                    'from_date': from_date,
                    'to_date': to_date,
                    'status': 'success',
                    'error': None,
                    'count': 0,
                    'request_params': {
                        'limit': limit,
                        'adjusted': True,
                        'sort': 'asc'
                    },
                    'timestamp': datetime.now().isoformat(),
                    'data': []
                }
                
                return df, metadata
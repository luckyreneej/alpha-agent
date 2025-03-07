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
                            timespan: str = 'hour') -> Dict[str, pd.DataFrame]:
        """
        获取历史股票数据。
        
        Args:
            tickers: 股票代码列表
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            timespan: 时间间隔 ('hour' 或 'day')
            
        Returns:
            Dict[str, pd.DataFrame]: 股票数据字典
        """
        # 验证日期
        try:
            start = datetime.strptime(start_date, '%Y-%m-%d').date()
            end = datetime.strptime(end_date, '%Y-%m-%d').date()
            current = datetime.now().date()
            
            # 确保日期不超过当前日期
            if end > current:
                logger.warning(f"End date {end_date} is in the future, using yesterday instead")
                end = current - timedelta(days=1)
                end_date = end.strftime('%Y-%m-%d')
            
            if start > end:
                logger.warning(f"Start date {start_date} is after end date {end_date}, adjusting start date")
                start = end - timedelta(days=7)  # 默认获取7天数据
                start_date = start.strftime('%Y-%m-%d')
            
        except ValueError as e:
            logger.error(f"Invalid date format: {e}")
            raise ValueError(f"Invalid date format: {e}")

        logger.info(f"Fetching {timespan} data for {len(tickers)} tickers from {start_date} to {end_date}")
        
        stock_data = {}
        for ticker in tickers:
            try:
                logger.info(f"Fetching {timespan} data for {ticker} from {start_date} to {end_date}")
                df, metadata = self.api.get_stock_bars(
                    ticker=ticker,
                    timespan=timespan,
                    from_date=start_date,
                    to_date=end_date,
                    multiplier=1
                )
                
                if not df.empty:
                    stock_data[ticker] = df
                    logger.info(f"Successfully retrieved {len(df)} bars for {ticker}")
                else:
                    logger.warning(f"No data retrieved for {ticker}")
                
                # 添加短暂延迟以避免触发速率限制
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error fetching data for {ticker}: {str(e)}")
                continue
        
        if not stock_data:
            logger.error("No data retrieved for any ticker")
            raise ValueError("No stock data available for backtesting")
        
        return stock_data

    def prepare_backtest_data(self,
                              dataset: Dict[str, Any],
                              format_type: str = 'panel') -> pd.DataFrame:
        """
        准备回测数据。
        
        Args:
            dataset: 包含股票数据的字典
            format_type: 数据格式类型 ('panel' 或 'long')
            
        Returns:
            pd.DataFrame: 格式化的回测数据
        """
        if not dataset.get('stocks'):
            raise ValueError("No stock data available for backtesting")
        
        stock_data = dataset['stocks']
        
        if format_type == 'panel':
            # 创建多索引 DataFrame
            panel_data = []
            for ticker, df in stock_data.items():
                if not df.empty:
                    df = df.copy()
                    df['ticker'] = ticker
                    panel_data.append(df)
            
            if not panel_data:
                raise ValueError("No valid data found in any ticker")
            
            combined_df = pd.concat(panel_data, axis=0)
            combined_df.set_index(['timestamp', 'ticker'], inplace=True)
            combined_df.sort_index(inplace=True)
            
            return combined_df
        
        elif format_type == 'long':
            # 创建长格式 DataFrame
            long_data = []
            for ticker, df in stock_data.items():
                if not df.empty:
                    df = df.copy()
                    df['ticker'] = ticker
                    long_data.append(df)
            
            if not long_data:
                raise ValueError("No valid data found in any ticker")
            
            combined_df = pd.concat(long_data, axis=0)
            combined_df.sort_values(['timestamp', 'ticker'], inplace=True)
            combined_df.reset_index(drop=True, inplace=True)
            
            return combined_df
        
        else:
            raise ValueError(f"Unsupported format type: {format_type}")

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

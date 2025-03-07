import pandas as pd
from datetime import datetime
import logging
import time
import numpy as np
from typing import Dict, Any

from agents.base_agent import BaseAgent
from utils.communication.message import Message, MessageType
from models.signals.alpha_factors import AlphaFactors
from backtest.backtest_engine import BacktestEngine
from backtest.strategy import MovingAverageCrossStrategy, RSIStrategy, MACDStrategy
from backtest.strategy import create_combined_strategy
from backtest.performance_metrics import PerformanceAnalyzer


class SignalAgent(BaseAgent):
    """Signal generation agent that creates trading signals based on alpha factors.

    This agent evaluates alpha factors to generate trading signals and manages
    signal-related aspects of the trading system.

    Attributes:
        alpha_factors (AlphaFactors): Component for calculating and evaluating alpha factors
        signals (dict): Recently generated signals
        signal_history (list): History of generated signals
        signal_performance (dict): Performance tracking for signals
    """

    def __init__(self, agent_id: str, communicator, config=None):
        """Initialize the SignalAgent.

        Args:
            agent_id: Unique agent identifier
            communicator: Communication manager instance
            config (dict, optional): Configuration settings
        """
        super().__init__(agent_id, communicator)

        # Default configuration
        self.config = config or {
            'alpha_weights': {
                'alpha1': 0.2,
                'alpha12': 0.3,
                'alpha101': 0.5
            },
            'factor_evaluation': {
                'min_ic_threshold': 0.05,
                'max_correlation': 0.7,
                'min_half_life_days': 5
            },
            'signal_thresholds': {
                'buy': 0.3,  # Signal must be above 0.3 to generate buy
                'sell': -0.3,  # Signal must be below -0.3 to generate sell
                'strong_threshold': 0.7  # Threshold for strong signal classification
            }
        }

        # Initialize alpha factor engine with optional config path
        alpha_config_path = config.get('alpha_config_path') if config else None
        self.alpha_factors = AlphaFactors(config_path=alpha_config_path)

        # Initialize signal tracking
        self.signals = {}  # Current active signals {ticker: {direction, strength, timestamp, factors}}
        self.signal_history = []  # History of generated signals
        self.signal_performance = {  # Performance tracking
            'hit_rate': 0.0,
            'total_signals': 0,
            'successful_signals': 0,
            'false_signals': 0
        }

        # Register for relevant data updates
        self._register_data_handlers()

        # Initialize logger
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize backtest components
        self.backtest_engine = BacktestEngine(
            initial_capital=self.config.get('trading', {}).get('initial_capital', 100000),
            commission=self.config.get('trading', {}).get('commission', 0.001),
            slippage=self.config.get('trading', {}).get('slippage', 0.001)
        )
        
        # Initialize strategies
        self.strategies = {
            'ma_cross': MovingAverageCrossStrategy(
                fast_period=self.config.get('strategy', {}).get('ma_fast', 50),
                slow_period=self.config.get('strategy', {}).get('ma_slow', 200)
            ),
            'rsi': RSIStrategy(
                period=self.config.get('strategy', {}).get('rsi_period', 14),
                oversold=self.config.get('strategy', {}).get('rsi_oversold', 30),
                overbought=self.config.get('strategy', {}).get('rsi_overbought', 70)
            ),
            'macd': MACDStrategy(
                fast_period=self.config.get('strategy', {}).get('macd_fast', 12),
                slow_period=self.config.get('strategy', {}).get('macd_slow', 26),
                signal_period=self.config.get('strategy', {}).get('macd_signal', 9)
            )
        }
        
        # Initialize performance analyzer
        self.performance_analyzer = PerformanceAnalyzer()

    def _register_data_handlers(self):
        """Register handlers for data updates from other agents."""
        if self.communicator:
            # Register for signal generation requests
            self.communicator.register_request_handler(
                self.agent_id, "generate_signal", self._handle_signal_request)

            # Subscribe to market data updates
            self.subscribe_to_topic("market_data")

            # Register for topic-based notifications
            self.communicator.register_event_handler("stock_data_updated", self._on_stock_data_updated)

    def _on_stock_data_updated(self, key, value):
        """Handler for stock data updates."""
        self.logger.info(f"Received stock data update. Generating new signals.")
        self._generate_signals_from_data(value)

    def process_message(self, message: Message) -> None:
        """Process a single message.

        Args:
            message: Message to process
        """
        try:
            # Handle different message types
            if message.message_type == MessageType.REQUEST:
                # Process request
                if message.metadata and 'request_type' in message.metadata:
                    request_type = message.metadata['request_type']
                    if request_type == 'generate_signal':
                        response_content = self._handle_signal_request(message)

                        # Send response
                        self.send_message(
                            receiver_id=message.sender_id,
                            message_type=MessageType.RESPONSE,
                            content=response_content,
                            reply_to=message.id,
                            correlation_id=message.correlation_id
                        )

            elif message.message_type == MessageType.DATA:
                # Process data message
                if message.metadata and 'topic' in message.metadata:
                    topic = message.metadata['topic']
                    if 'market_data' in topic:
                        # Handle market data update
                        self._generate_signals_from_data(message.content)
                    elif 'signal_feedback' in topic:
                        # Handle feedback on signal performance
                        self._update_signal_performance(message.content)

            else:
                # Log message
                self.logger.debug(f"Received message: {message.message_type} from {message.sender_id}")

        except Exception as e:
            self.logger.error(f"Error processing message: {e}")

    def _handle_signal_request(self, message: Message):
        """Handle requests for generating trading signals.

        Args:
            message (Message): Request message

        Returns:
            dict: Signal information
        """
        try:
            ticker = message.content.get('ticker')
            data = message.content.get('data')

            if not ticker or data is None:
                return {'error': 'Missing ticker or data'}

            # Convert to dataframe if it's a dict
            if isinstance(data, dict):
                data = pd.DataFrame(data)

            # Generate signal
            signal = self._generate_signal(ticker, data)

            return {
                'signal': signal,
                'ticker': ticker,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Error handling signal request: {e}")
            return {'error': str(e)}

    def _generate_signals_from_data(self, market_data):
        """Generate signals from market data update.

        Args:
            market_data (DataFrame): Market data for multiple tickers
        """
        try:
            # Group data by ticker
            if not isinstance(market_data, pd.DataFrame):
                self.logger.warning("Market data is not a DataFrame")
                return

            if 'ticker' not in market_data.columns:
                self.logger.warning("Market data missing ticker column")
                return

            # Get unique tickers
            tickers = market_data['ticker'].unique()

            # Generate signals for each ticker
            signals = []
            for ticker in tickers:
                ticker_data = market_data[market_data['ticker'] == ticker].copy()

                # Only process if we have enough data
                if len(ticker_data) < 20:  # Need at least 20 bars for most factors
                    continue

                # Reset index to ensure time-series is properly aligned
                ticker_data = ticker_data.sort_values('timestamp').reset_index(drop=True)

                # Generate signal
                signal = self._generate_signal(ticker, ticker_data)
                signals.append(signal)

                # Store if it exceeds thresholds
                self._store_signal_if_significant(signal)

            # Publish signals to topic
            if signals:
                self.publish_to_topic(
                    topic="trading_signals",
                    content={
                        'signals': signals,
                        'timestamp': datetime.now().isoformat(),
                        'source': self.agent_id
                    }
                )

                self.logger.info(f"Published {len(signals)} trading signals")

        except Exception as e:
            self.logger.error(f"Error generating signals from market data: {e}")

    def _generate_signal(self, ticker, data):
        """Generate signal for a single ticker.

        Args:
            ticker (str): Stock ticker symbol
            data (DataFrame): Historical price data

        Returns:
            dict: Signal information
        """
        try:
            # Get list of factors to use from config
            alpha_weights = self.config.get('alpha_weights', {})
            selected_factors = list(alpha_weights.keys())

            # Calculate factors
            data_with_factors = self.alpha_factors.calculate_alpha_factors(
                data, selected_factors=selected_factors)

            # Generate combined signal
            signal_value = self._combine_factor_signals(data_with_factors, alpha_weights)

            # Determine signal direction and strength
            direction = "neutral"
            if signal_value > self.config['signal_thresholds']['buy']:
                direction = "buy"
            elif signal_value < self.config['signal_thresholds']['sell']:
                direction = "sell"

            strength = "normal"
            if abs(signal_value) > self.config['signal_thresholds']['strong_threshold']:
                strength = "strong"

            # Create signal object
            signal = {
                'ticker': ticker,
                'value': float(signal_value),
                'direction': direction,
                'strength': strength,
                'timestamp': datetime.now().isoformat(),
                'factors': {factor: float(data_with_factors[factor].iloc[-1])
                            for factor in selected_factors
                            if factor in data_with_factors.columns}
            }

            return signal

        except Exception as e:
            self.logger.error(f"Error generating signal for {ticker}: {e}")
            return {
                'error': str(e),
                'ticker': ticker
            }

    def _store_signal_if_significant(self, signal):
        """Store signal if it exceeds significance thresholds.

        Args:
            signal (dict): Signal information
        """
        # Check if it's a valid signal dict
        if not isinstance(signal, dict) or 'ticker' not in signal or 'direction' not in signal:
            return

        # Only store buy/sell signals, not neutral
        if signal['direction'] == 'neutral':
            return

        # Store in active signals
        ticker = signal['ticker']
        self.signals[ticker] = signal

        # Add to history
        self.signal_history.append(signal)

        # Limit history size
        if len(self.signal_history) > 1000:
            self.signal_history = self.signal_history[-1000:]

        # Log significant signals
        if signal['strength'] == 'strong':
            self.logger.info(
                f"Strong {signal['direction']} signal generated for {ticker} with value {signal['value']:.4f}")

    def _combine_factor_signals(self, data_with_factors, alpha_weights):
        """Combine multiple alpha factor signals into a single signal.

        Args:
            data_with_factors (DataFrame): Data with calculated factors
            alpha_weights (dict): Factor weights for combining signals

        Returns:
            float: Combined signal value (-1 to 1 scale)
        """
        # Get the most recent values for each factor
        latest_values = {}
        for factor, weight in alpha_weights.items():
            if factor in data_with_factors.columns:
                latest_values[factor] = data_with_factors[factor].iloc[-1]

        if not latest_values:
            return 0.0

        # Normalize factor values to -1 to 1 scale if not already
        normalized_values = {}
        for factor, value in latest_values.items():
            # Simple normalization - use percentile rank within the factor's history
            if len(data_with_factors) > 1:
                # Use percentile to normalize
                percentile = (data_with_factors[factor].rank(pct=True).iloc[-1] - 0.5) * 2
                normalized_values[factor] = percentile
            else:
                # Can't normalize with just one value, use raw value clamped to [-1, 1]
                normalized_values[factor] = max(min(value, 1.0), -1.0)

        # Apply weights and combine
        weighted_sum = sum(normalized_values[factor] * weight
                           for factor, weight in alpha_weights.items()
                           if factor in normalized_values)

        # Normalize by sum of weights
        total_weight = sum(weight for factor, weight in alpha_weights.items()
                           if factor in normalized_values)

        if total_weight == 0:
            return 0.0

        combined_signal = weighted_sum / total_weight

        # Ensure signal is in [-1, 1] range
        return max(min(combined_signal, 1.0), -1.0)

    def _update_signal_performance(self, feedback):
        """Update signal performance based on feedback.

        Args:
            feedback (dict): Feedback information about signal performance
        """
        try:
            # Check if feedback contains required fields
            if not isinstance(feedback, dict) or 'ticker' not in feedback or 'success' not in feedback:
                return

            # Update performance metrics
            self.signal_performance['total_signals'] += 1

            if feedback['success']:
                self.signal_performance['successful_signals'] += 1
            else:
                self.signal_performance['false_signals'] += 1

            # Calculate hit rate
            total = self.signal_performance['total_signals']
            if total > 0:
                hit_rate = self.signal_performance['successful_signals'] / total
                self.signal_performance['hit_rate'] = hit_rate

            # Log performance update
            self.logger.info(
                f"Signal performance updated: {self.signal_performance['hit_rate']:.2f} hit rate from {total} signals")

        except Exception as e:
            self.logger.error(f"Error updating signal performance: {e}")

    def get_signals_for_ticker(self, ticker):
        """Get current signal for a ticker.

        Args:
            ticker (str): Ticker symbol

        Returns:
            dict: Signal information or None if no active signal
        """
        return self.signals.get(ticker)

    def get_all_active_signals(self):
        """Get all currently active signals.

        Returns:
            dict: Dictionary of ticker -> signal
        """
        return self.signals.copy()

    def clear_expired_signals(self, expiry_hours=24):
        """Clear signals older than the specified time.

        Args:
            expiry_hours (int): Hours after which signals are considered expired
        """
        now = datetime.now()
        expired_tickers = []

        for ticker, signal in self.signals.items():
            try:
                signal_time = datetime.fromisoformat(signal['timestamp'])
                if (now - signal_time).total_seconds() > expiry_hours * 3600:
                    expired_tickers.append(ticker)
            except (ValueError, KeyError):
                # If timestamp is invalid, consider it expired
                expired_tickers.append(ticker)

        # Remove expired signals
        for ticker in expired_tickers:
            del self.signals[ticker]

        if expired_tickers:
            self.logger.info(f"Cleared {len(expired_tickers)} expired signals")

    def run(self):
        """Run the signal agent's main loop."""
        self.logger.info(f"Signal agent {self.agent_id} starting")

        # Start the agent
        self.start()

        try:
            while self.running:
                # Process incoming messages
                self.process_messages()

                # Clean up expired signals hourly
                now = datetime.now()
                if now.minute == 0 and now.second < 10:  # Run at the top of each hour
                    self.clear_expired_signals()

                # Sleep to avoid high CPU usage
                time.sleep(1)

        except KeyboardInterrupt:
            self.logger.info("Signal agent received shutdown signal")
        except Exception as e:
            self.logger.error(f"Error in signal agent main loop: {e}")
        finally:
            self.stop()
            self.logger.info("Signal agent stopped")

    def run_trading_loop(self):
        """
        Main trading loop for the signal agent.
        Continuously monitors for signals and manages trading operations.
        """
        self.logger.info("Starting signal agent trading loop")
        self.running = True
        
        try:
            while self.running:
                # Process any pending messages
                self.process_messages()
                
                # Clear expired signals
                self.clear_expired_signals()
                
                # Sleep to prevent busy waiting
                time.sleep(1)
                
        except Exception as e:
            self.logger.error(f"Error in trading loop: {e}")
            raise
        finally:
            self.logger.info("Signal agent trading loop stopped")

    def __str__(self):
        return f"SignalAgent(id={self.agent_id})"

    def run_backtest(self) -> Dict[str, Any]:
        """
        Run backtest using historical data and configured strategies.

        Returns:
            Dict containing backtest results
        """
        try:
            logger.info("Starting backtest")
            
            # Get backtest date from config
            backtest_date = self.config.get('backtest', {}).get('date')
            if not backtest_date:
                logger.error("No backtest date specified in config")
                return {'error': 'No backtest date specified'}

            # Wait for market data to be available
            max_retries = 10
            retry_count = 0
            while not self.market_data and retry_count < max_retries:
                logger.info(f"Waiting for market data... (attempt {retry_count + 1}/{max_retries})")
                import time
                time.sleep(2)
                retry_count += 1

            if not self.market_data:
                logger.error("No market data available for backtest")
                return {'error': 'No market data available'}

            # Prepare data for backtest
            backtest_data = self._prepare_backtest_data()
            
            # Create combined strategy
            strategy = create_combined_strategy([
                self.strategies['ma_cross'],
                self.strategies['rsi'],
                self.strategies['macd']
            ])

            # Run backtest
            results = self.backtest_engine.run_backtest(
                data=backtest_data,
                strategy=strategy.generate_signals
            )

            # Calculate performance metrics
            portfolio_history = results['portfolio_history']
            portfolio_history['date'] = pd.to_datetime(portfolio_history['date'])
            portfolio_history.set_index('date', inplace=True)
            returns = portfolio_history['portfolio_value'].pct_change().dropna()

            # Get benchmark data if available
            benchmark_returns = None
            if 'SPY' in self.market_data:
                spy_data = pd.DataFrame(self.market_data['SPY'])
                spy_data['date'] = pd.to_datetime(spy_data['date'])
                spy_data.set_index('date', inplace=True)
                benchmark_returns = spy_data['close'].pct_change().dropna()

            # Calculate performance metrics
            performance_metrics = self.performance_analyzer.calculate_metrics(
                returns=returns,
                benchmark_returns=benchmark_returns,
                risk_free_rate=self.config.get('backtest', {}).get('risk_free_rate', 0.02)
            )

            # Combine all results
            final_results = {
                'portfolio_history': results['portfolio_history'].to_dict('records'),
                'trades': results['trades'],
                'metrics': {
                    **results['metrics'],
                    **performance_metrics
                },
                'backtest_date': backtest_date,
                'strategy_config': {
                    name: strategy.get_parameters()
                    for name, strategy in self.strategies.items()
                }
            }

            logger.info("Backtest completed successfully")
            return final_results

        except Exception as e:
            logger.error(f"Error in backtest: {e}", exc_info=True)
            return {'error': str(e)}

    def _prepare_backtest_data(self) -> Dict[str, pd.DataFrame]:
        """
        Prepare market data for backtesting.

        Returns:
            Dict mapping tickers to their historical data DataFrames
        """
        backtest_data = {}
        
        for ticker, data in self.market_data.items():
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, pd.DataFrame):
                df = data.copy()
            else:
                logger.warning(f"Unexpected data type for {ticker}: {type(data)}")
                continue

            # Ensure required columns exist
            required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_columns):
                logger.warning(f"Missing required columns for {ticker}")
                continue

            # Convert date column to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(df['date']):
                df['date'] = pd.to_datetime(df['date'])

            # Sort by date
            df = df.sort_values('date')

            backtest_data[ticker] = df

        return backtest_data

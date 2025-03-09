import argparse
import logging
import os
import yaml
from datetime import datetime, timedelta
from dotenv import load_dotenv
import json
import asyncio
from typing import Optional, Dict, Any
import sys

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler('alpha_agent.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import local modules
from utils.communication.unified_communication import UnifiedCommunicationManager
from utils.communication.conversation_manager import ConversationManager
from agents.data_agent import DataAgent
from agents.prediction_agent import PredictionAgent
from agents.risk_agent import RiskAgent
from agents.sentiment_agent import SentimentAgent
from agents.signal_agent import SignalAgent
from utils.communication.message import Message, MessageType
from backtest.historical_data_fetcher import HistoricalDataFetcher
from backtest.backtest_engine import BacktestEngine
from backtest.performance_metrics import PerformanceAnalyzer
from backtest.strategy import MovingAverageCrossStrategy, RSIStrategy, MACDStrategy, create_combined_strategy

class TradingSystem:
    def __init__(self, config_path: str = 'configs/config.yaml'):
        self.config_path = config_path
        self.config = None
        self.unified_comm = None
        self.conversation_manager = None
        self.agents = {}
        self._setup_directories()

    def _setup_directories(self):
        """Set up required directories"""
        required_dirs = [
            'data/cache',
            'data/market_data',
            'results',
            'plots',
            'models/trained',
            'backtest',
            'logs'
        ]
        for directory in required_dirs:
            os.makedirs(directory, exist_ok=True)
            logger.debug(f"Directory created/verified: {directory}")

    def load_config(self) -> Dict[str, Any]:
        """Load and validate configuration"""
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)

            # Set API keys from environment variables
            self.config.setdefault('api_keys', {})
            self.config['api_keys'].update({
                'openai': os.getenv('OPENAI_API_KEY'),
                'polygon': os.getenv('POLYGON_API_KEY')
            })

            # Validate required API keys
            missing_keys = [
                key for key in ['openai', 'polygon']
                if not self.config['api_keys'].get(key)
            ]
            if missing_keys:
                raise ValueError(f"Missing required API keys: {', '.join(missing_keys)}")

            return self.config

        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise

    async def initialize_system(self):
        """Initialize the trading system components."""
        try:
            # Create directories if they don't exist
            self._setup_directories()

            # Load configuration
            self.load_config()

            # Initialize communication manager
            self.unified_comm = UnifiedCommunicationManager()
            await self.unified_comm.start()

            # Initialize conversation manager
            self.conversation_manager = ConversationManager(self.unified_comm)

            # Define agent configurations
            agent_configs = {
                'data': {
                    'data_config': self.config.get('data', {}),
                    'base_config': self.config,
                    'api_key': self.config.get('api_keys', {}).get('polygon')
                },
                'prediction': {
                    'prediction_config': self.config.get('prediction', {}),
                    'base_config': self.config
                },
                'risk': {
                    'risk_config': self.config.get('risk', {}),
                    'base_config': self.config
                },
                'sentiment': {
                    'sentiment_config': self.config.get('sentiment', {}),
                    'base_config': self.config
                },
                'signal': {
                    'trading_config': self.config.get('trading', {}),
                    'base_config': self.config
                }
            }

            # Initialize agents
            agent_classes = {
                'data': DataAgent,
                'prediction': PredictionAgent,
                'risk': RiskAgent,
                'sentiment': SentimentAgent,
                'signal': SignalAgent
            }

            self.agents = {}
            for agent_type, agent_class in agent_classes.items():
                try:
                    # Create agent instance with configuration
                    agent = agent_class(
                        agent_id_or_config=f"{agent_type}_agent",
                        communicator=self.unified_comm,
                        **agent_configs[agent_type]
                    )
                    
                    # Register agent with communication manager
                    await self.unified_comm.register_agent(agent)
                    
                    # Initialize agent if it has async initialization
                    if hasattr(agent, 'initialize'):
                        await agent.initialize()
                    
                    # Store agent reference
                    self.agents[agent_type] = agent
                    
                    logger.info(f"Successfully initialized {agent_type} agent")
                    
                except Exception as e:
                    logger.error(f"Failed to initialize {agent_type} agent: {e}")
                    raise

            # Setup communication channels and topics
            await self._setup_communication()

            # Start all agents
            for agent in self.agents.values():
                if hasattr(agent, 'start'):
                    await agent.start()
                logger.info(f"Started agent: {agent.agent_id}")

            logger.info("System initialization completed successfully")
            return True

        except Exception as e:
            logger.error(f"Error during system initialization: {e}")
            await self.cleanup()
            return False

    async def _setup_communication(self):
        """Setup communication channels and topics between agents."""
        try:
            # Define topics
            topics = [
                'market_data',          # Market data updates
                'trading_signals',      # Trading signals
                'risk_updates',         # Risk assessment updates
                'sentiment_updates',    # Market sentiment updates
                'prediction_updates',   # Price prediction updates
                'system_status',        # System status updates
                'agent_status',         # Individual agent status
                'trade_execution',      # Trade execution updates
                'portfolio_updates',    # Portfolio status updates
                'performance_metrics'   # Trading performance metrics
            ]

            # Register topics with communication manager
            for topic in topics:
                await self.unified_comm.register_topic(topic)
                logger.info(f"Registered topic: {topic}")

            # Setup agent subscriptions
            subscriptions = {
                'data': ['system_status', 'trade_execution'],
                'prediction': ['market_data', 'system_status'],
                'risk': ['market_data', 'trading_signals', 'portfolio_updates', 'system_status'],
                'sentiment': ['market_data', 'system_status'],
                'signal': ['market_data', 'sentiment_updates', 'prediction_updates', 'risk_updates', 'system_status']
            }

            # Subscribe agents to their topics
            for agent_type, topics in subscriptions.items():
                agent = self.agents.get(agent_type)
                if agent:
                    for topic in topics:
                        success = await self.unified_comm.subscribe_to_topic(agent.agent_id, topic)
                        if success:
                            logger.info(f"Subscribed {agent.agent_id} to {topic}")
                        else:
                            logger.warning(f"Failed to subscribe {agent.agent_id} to {topic}")

            # Setup request handlers
            request_handlers = {
                'data': {
                    'get_market_data': self.agents['data'].handle_market_data_request,
                    'get_news_data': self.agents['data'].handle_news_request
                },
                'prediction': {
                    'get_prediction': self.agents['prediction'].handle_prediction_request
                },
                'risk': {
                    'assess_risk': self.agents['risk'].handle_risk_assessment_request,
                    'validate_trade': self.agents['risk'].handle_trade_validation_request
                },
                'sentiment': {
                    'analyze_sentiment': self.agents['sentiment'].handle_sentiment_request
                },
                'signal': {
                    'generate_signal': self.agents['signal'].handle_signal_request,
                    'get_active_signals': self.agents['signal'].get_all_active_signals
                }
            }

            # Register request handlers
            for agent_type, handlers in request_handlers.items():
                agent = self.agents.get(agent_type)
                if agent:
                    for request_type, handler in handlers.items():
                        await agent.register_request_handler(request_type, handler)
                        logger.info(f"Registered {request_type} handler for {agent.agent_id}")

            logger.info("Communication setup completed successfully")
            return True

        except Exception as e:
            logger.error(f"Error setting up communication: {e}")
            return False

    async def run_live_trading(self):
        """Run live trading session"""
        try:
            trading_params = self.config.get('trading', {})
            symbols = trading_params.get('symbols', ['AAPL', 'GOOGL', 'MSFT'])
            update_interval = trading_params.get('update_interval', 60)
            risk_threshold = trading_params.get('risk_threshold', 0.7)

            while True:
                try:
                    for symbol in symbols:
                        await self._process_symbol(symbol, risk_threshold)
                    await asyncio.sleep(update_interval)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in trading loop: {e}")
                    await asyncio.sleep(update_interval)

        except Exception as e:
            logger.error(f"Fatal error in live trading: {e}")
            raise
        finally:
            await self.cleanup()

    async def _process_symbol(self, symbol: str, risk_threshold: float):
        """Process a single symbol"""
        try:
            # Create conversation context
            conv_id = self.conversation_manager.create_conversation(
                initiator_id='data_agent',
                participants=list(self.agents.keys()),
                topic=f'analysis_{symbol}',
                timeout=300
            )

            # Get market data
            market_data = await self.agents['data_agent'].fetch_stock_data(
                tickers=[symbol],
                timespan=self.config.get('data', {}).get('default_timespan', 'day'),
                days_back=self.config.get('data', {}).get('data_history_days', 90)
            )

            if not market_data or symbol not in market_data:
                logger.warning(f"No market data available for {symbol}")
                return

            # Process data through agents
            sentiment = await self._get_sentiment(symbol, conv_id)
            predictions = await self._get_predictions(symbol, market_data[symbol], conv_id)
            risk_assessment = await self._get_risk_assessment(
                symbol, market_data[symbol], predictions, sentiment, conv_id
            )
            trading_signal = await self._get_trading_signal(
                symbol, market_data[symbol], predictions, risk_assessment, sentiment, conv_id
            )

            # Execute trading decision if risk is acceptable
            risk_score = risk_assessment.get('market_risk_assessment', {}).get('overall_risk_score', 1.0)
            if risk_score <= risk_threshold:
                await self._execute_trade(symbol, trading_signal, risk_score)

        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
        finally:
            if conv_id:
                self.conversation_manager.close_conversation(conv_id)

    async def _get_sentiment(self, symbol: str, conv_id: str) -> Dict[str, Any]:
        """Get sentiment analysis for a symbol"""
        try:
            sentiment_agent = self.agents.get('sentiment_agent')
            if not sentiment_agent:
                logger.error("Sentiment agent not initialized")
                return {}

            sentiment = await sentiment_agent.analyze_sentiment(symbol)
            logger.debug(f"Sentiment analysis for {symbol}: {sentiment}")
            return sentiment
        except Exception as e:
            logger.error(f"Error getting sentiment for {symbol}: {e}")
            return {}

    async def _get_predictions(self, symbol: str, market_data: Dict, conv_id: str) -> Dict[str, Any]:
        """Get predictions for a symbol"""
        try:
            prediction_agent = self.agents.get('prediction_agent')
            if not prediction_agent:
                logger.error("Prediction agent not initialized")
                return {}

            predictions = await prediction_agent.generate_predictions(symbol, market_data)
            logger.debug(f"Predictions for {symbol}: {predictions}")
            return predictions
        except Exception as e:
            logger.error(f"Error getting predictions for {symbol}: {e}")
            return {}

    async def _get_risk_assessment(self, symbol: str, market_data: Dict,
                                 predictions: Dict, sentiment: Dict, conv_id: str) -> Dict[str, Any]:
        """Get risk assessment for a symbol"""
        try:
            risk_agent = self.agents.get('risk_agent')
            if not risk_agent:
                logger.error("Risk agent not initialized")
                return {}

            risk_assessment = await risk_agent.assess_market_risk(
                symbol, market_data, predictions, sentiment
            )
            logger.debug(f"Risk assessment for {symbol}: {risk_assessment}")
            return risk_assessment
        except Exception as e:
            logger.error(f"Error getting risk assessment for {symbol}: {e}")
            return {}

    async def _get_trading_signal(self, symbol: str, market_data: Dict,
                                predictions: Dict, risk_assessment: Dict,
                                sentiment: Dict, conv_id: str) -> Dict[str, Any]:
        """Get trading signal for a symbol"""
        try:
            signal_agent = self.agents.get('signal_agent')
            if not signal_agent:
                logger.error("Signal agent not initialized")
                return {}

            signal = await signal_agent.generate_signal(
                symbol=symbol,
                market_data=market_data,
                predictions=predictions,
                risk_assessment=risk_assessment,
                sentiment=sentiment
            )
            logger.debug(f"Trading signal for {symbol}: {signal}")
            return signal
        except Exception as e:
            logger.error(f"Error getting trading signal for {symbol}: {e}")
            return {}

    async def _execute_trade(self, symbol: str, trading_signal: Dict, risk_score: float) -> bool:
        """Execute a trade based on the signal"""
        try:
            if not trading_signal:
                logger.warning(f"No trading signal available for {symbol}")
                return False

            signal_type = trading_signal.get('signal_type')
            confidence = trading_signal.get('confidence', 0)
            
            # Get trading parameters from config
            trading_config = self.config.get('trading', {})
            min_confidence = trading_config.get('signal_threshold', 0.7)
            position_size = trading_config.get('position_size', 0.1)
            
            if confidence < min_confidence:
                logger.info(f"Signal confidence {confidence} below threshold {min_confidence} for {symbol}")
                return False

            # Log trading decision
            logger.info(f"Executing {signal_type} trade for {symbol} "
                       f"(confidence: {confidence}, risk_score: {risk_score})")

            # Here you would implement actual trade execution
            # For now, we just log the decision
            trade_info = {
                'symbol': symbol,
                'signal_type': signal_type,
                'confidence': confidence,
                'risk_score': risk_score,
                'position_size': position_size,
                'timestamp': datetime.now().isoformat()
            }

            # Save trade information
            self._save_trade_info(trade_info)
            return True

        except Exception as e:
            logger.error(f"Error executing trade for {symbol}: {e}")
            return False

    def _save_trade_info(self, trade_info: Dict[str, Any]) -> None:
        """Save trade information to file"""
        try:
            trades_file = os.path.join('results', 'trades.json')
            trades = []
            
            # Load existing trades if file exists
            if os.path.exists(trades_file):
                with open(trades_file, 'r') as f:
                    trades = json.load(f)
            
            # Add new trade
            trades.append(trade_info)
            
            # Save updated trades
            with open(trades_file, 'w') as f:
                json.dump(trades, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving trade info: {e}")

    async def run_backtest(self):
        """Run backtest simulation"""
        try:
            # Load backtest configuration
            backtest_config = self.config.get('backtest', {})
            start_date = datetime.strptime(backtest_config.get('start_date'), '%Y-%m-%d')
            end_date = datetime.strptime(backtest_config.get('end_date'), '%Y-%m-%d')
            symbols = backtest_config.get('symbols', ['SPY'])

            logger.info(f"Starting backtest from {start_date} to {end_date} for {symbols}")

            # Initialize backtest components
            data_fetcher = HistoricalDataFetcher(
                api_key=self.config['api_keys']['polygon'],
                cache_dir='data/cache'
            )

            # Create strategy
            strategy_config = backtest_config.get('strategy', {})
            strategy = create_combined_strategy(strategy_config)

            # Initialize backtest engine
            engine = BacktestEngine(
                data_fetcher=data_fetcher,
                strategy=strategy,
                initial_capital=backtest_config.get('initial_capital', 100000)
            )

            # Run backtest
            results = await engine.run(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                risk_manager=self.agents['risk_agent'],
                sentiment_analyzer=self.agents['sentiment_agent']
            )

            # Analyze results
            analyzer = PerformanceAnalyzer()
            metrics = analyzer.calculate_metrics(results)
            
            # Save results
            self._save_backtest_results(results, metrics)
            
            logger.info("Backtest completed successfully")
            return results, metrics

        except Exception as e:
            logger.error(f"Error in backtest: {e}")
            raise
        finally:
            await self.cleanup()

    def _save_backtest_results(self, results: Dict[str, Any], metrics: Dict[str, Any]) -> None:
        """Save backtest results and metrics"""
        try:
            timestamp = datetime.now().strftime(self.config['output']['timestamp_format'])
            results_dir = self.config['output']['results_dir']
            
            # Save results
            results_file = os.path.join(
                results_dir,
                f"backtest_results_{timestamp}.json"
            )
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            # Save metrics
            metrics_file = os.path.join(
                results_dir,
                f"backtest_metrics_{timestamp}.json"
            )
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
                
            logger.info(f"Backtest results saved to {results_file}")
            logger.info(f"Backtest metrics saved to {metrics_file}")
            
        except Exception as e:
            logger.error(f"Error saving backtest results: {e}")

    async def cleanup(self):
        """Clean up system resources"""
        try:
            # Stop all agents
            if hasattr(self, 'agents'):
                for agent_id, agent in self.agents.items():
                    try:
                        if agent is not None:
                            if hasattr(agent, 'cleanup'):
                                await agent.cleanup()
                            if hasattr(agent, 'stop'):
                                await agent.stop()
                    except Exception as e:
                        logger.error(f"Error cleaning up {agent_id}: {e}")

            # Shutdown communication
            if hasattr(self, 'unified_comm') and self.unified_comm is not None:
                try:
                    await self.unified_comm.shutdown()
                except Exception as e:
                    logger.error(f"Error shutting down communication manager: {e}")

            if hasattr(self, 'conversation_manager') and self.conversation_manager is not None:
                try:
                    await self.conversation_manager.shutdown()
                except Exception as e:
                    logger.error(f"Error shutting down conversation manager: {e}")

            logger.info("System cleanup completed")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        finally:
            # Clear references
            self.agents = {}
            self.unified_comm = None
            self.conversation_manager = None

async def main():
    """Main entry point"""
    try:
        # Parse arguments
        parser = argparse.ArgumentParser(description='Alpha Agent Trading System')
        parser.add_argument('--config', type=str, default='configs/config.yaml',
                          help='Path to configuration file')
        parser.add_argument('--mode', type=str, choices=['live', 'backtest'],
                          default='live', help='Trading mode')
        args = parser.parse_args()

        # Initialize and run system
        system = TradingSystem(args.config)
        system.load_config()

        # Set up event loop
        if sys.platform == 'win32':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

        try:
            # Initialize system
            init_success = await system.initialize_system()
            if not init_success:
                logger.error("System initialization failed")
                return

            # Run trading mode
            if args.mode == 'live':
                await system.run_live_trading()
            else:
                await system.run_backtest()
        except KeyboardInterrupt:
            logger.info("System shutdown requested by user")
        finally:
            await system.cleanup()

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    # Run the async main function
    asyncio.run(main())

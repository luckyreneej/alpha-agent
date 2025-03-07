import argparse
import logging
import os
import yaml
from datetime import datetime, timedelta
from dotenv import load_dotenv
import time
import json
import pandas as pd
import traceback

# Load environment variables from .env file
load_dotenv()

# Set up logging
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
from utils.data.polygon_api import PolygonAPI
from models.signals.alpha_factors import AlphaFactors
from utils.communication.message import Message, MessageType, MessagePriority
from backtest.historical_data_fetcher import HistoricalDataFetcher
from backtest.backtest_engine import BacktestEngine
from backtest.performance_metrics import PerformanceAnalyzer
from backtest.portfolio_optimizer import PortfolioOptimizer
from backtest.strategy import MovingAverageCrossStrategy, RSIStrategy, MACDStrategy, create_combined_strategy


def check_environment():
    """
    Check if required environment variables are set.
    Only warns if variables are missing, doesn't raise error.
    """
    required_vars = ['OPENAI_API_KEY', 'POLYGON_API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.warning(f"Missing env vars: {', '.join(missing_vars)}")
        logger.warning("Using dummy values for missing variables")


def load_config(config_path='configs/config.yaml'):
    """
    Load configuration from a YAML file and override with environment variables.

    Args:
        config_path: Path to the config YAML file, defaults to configs/config.yaml

    Returns:
        Dict containing configuration data
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Initialize api_keys if not present
        if 'api_keys' not in config:
            config['api_keys'] = {}
            
        # Handle API keys with environment variables taking precedence
        config['api_keys']['openai'] = os.getenv('OPENAI_API_KEY') or config['api_keys'].get('openai', 'dummy_openai_key')
        config['api_keys']['polygon'] = os.getenv('POLYGON_API_KEY') or config['api_keys'].get('polygon', 'dummy_polygon_key')
        
        # Validate API keys but don't raise error
        missing_keys = []
        for key_name, key_value in config['api_keys'].items():
            if not key_value or key_value.startswith('dummy_'):
                missing_keys.append(key_name)
        
        if missing_keys:
            logger.warning(f"Using dummy keys: {', '.join(missing_keys)}")
        
        logger.info(f"Config loaded from {config_path}")
        return config
    except FileNotFoundError:
        logger.warning(f"Config not found at {config_path}, using defaults")
        # Return default configuration
        return {
            'api_keys': {
                'openai': os.getenv('OPENAI_API_KEY', 'dummy_openai_key'),
                'polygon': os.getenv('POLYGON_API_KEY', 'dummy_polygon_key')
            },
            'system': {
                'update_interval': 60,
                'log_level': 'INFO'
            }
        }
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        raise


def setup_communication_framework():
    """
    Set up the multi-agent communication framework.

    Returns:
        tuple: (UnifiedCommunicationManager, ConversationManager) instances
    """
    # Create the unified communication manager
    unified_comm = UnifiedCommunicationManager()

    # Create the conversation manager for more complex interactions
    conversation_manager = ConversationManager(unified_comm)

    logger.info("Communication framework initialized")
    return unified_comm, conversation_manager


def initialize_agents(config, unified_comm, conversation_manager):
    """
    Initialize all agents based on the configuration.

    Args:
        config: Configuration dictionary
        unified_comm: UnifiedCommunicationManager instance
        conversation_manager: ConversationManager instance

    Returns:
        Dict mapping agent names to agent objects
    """
    # Initialize all agents with the same communication manager
    data_agent = DataAgent(
        agent_id="data_agent",
        communicator=unified_comm,
        api_key=config['api_keys']['polygon'],
        config_path="configs/config.yaml"
    )
    prediction_agent = PredictionAgent(config, unified_comm)
    risk_agent = RiskAgent(config, unified_comm)
    sentiment_agent = SentimentAgent(config, unified_comm)
    signal_agent = SignalAgent(config, unified_comm)

    # Put all agents into a dictionary
    agents = {
        'data_agent': data_agent,
        'prediction_agent': prediction_agent,
        'risk_agent': risk_agent,
        'sentiment_agent': sentiment_agent,
        'signal_agent': signal_agent,
    }

    # Register all agents with the communication manager
    for agent_id, agent in agents.items():
        unified_comm.register_agent(agent_id, agent)

    # Set up communication channels/topics
    setup_agent_communication(unified_comm, agents, conversation_manager)

    # Configure logging level from config
    if 'system' in config and 'log_level' in config['system']:
        log_level = getattr(logging, config['system']['log_level'])
        logging.getLogger().setLevel(log_level)
        logger.info(f"Set logging level to {config['system']['log_level']}")

    logger.info("All agents initialized and registered with the communication framework.")
    return agents


def setup_agent_communication(unified_comm, agents, conversation_manager):
    """
    Set up communication channels between agents.

    Args:
        unified_comm: UnifiedCommunicationManager instance
        agents: Dict of agent objects
        conversation_manager: ConversationManager instance
    """
    # First ensure all agents are registered
    for agent_id, agent in agents.items():
        if not unified_comm.register_agent(agent_id, agent):
            logger.error(f"Failed to register agent: {agent_id}")
            continue
        logger.info(f"Successfully registered agent: {agent_id}")

    # Define the communication topics
    topics = [
        "market_data",
        "predictions",
        "risk_metrics",
        "sentiment_analysis",
        "trading_signals",
        "system_status"
    ]

    # Set up topics in the communication manager
    for topic in topics:
        unified_comm.create_topic(topic)

    # Subscribe agents to relevant topics
    subscriptions = {
        "data_agent": ["market_data", "system_status"],
        "prediction_agent": ["market_data", "predictions"],
        "risk_agent": ["market_data", "predictions", "risk_metrics"],
        "sentiment_agent": ["sentiment_analysis"],
        "signal_agent": ["market_data", "predictions", "risk_metrics", 
                        "sentiment_analysis", "trading_signals", "system_status"]
    }

    # Subscribe each agent to their topics
    for agent_id, topic_list in subscriptions.items():
        for topic in topic_list:
            if not unified_comm.subscribe_to_topic(agent_id, topic):
                logger.error(f"Failed to subscribe {agent_id} to topic: {topic}")
            else:
                logger.info(f"Successfully subscribed {agent_id} to topic: {topic}")

    # Set up conversation flows in the conversation manager
    market_analysis_id = conversation_manager.create_conversation(
        initiator_id="data_agent",
        participants=["data_agent", "prediction_agent", "risk_agent", "sentiment_agent"],
        topic="market_analysis"
    )
    if not market_analysis_id:
        logger.error("Failed to create market analysis conversation")

    trading_decision_id = conversation_manager.create_conversation(
        initiator_id="prediction_agent",
        participants=["prediction_agent", "risk_agent", "sentiment_agent", "signal_agent"],
        topic="trading_decision"
    )
    if not trading_decision_id:
        logger.error("Failed to create trading decision conversation")

    logger.info("Inter-agent communication channels established")


def parse_args():
    """
    Parse command line arguments.

    Returns:
        Namespace containing the arguments
    """
    parser = argparse.ArgumentParser(description='Alpha Agent Trading System')

    # Add arguments
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to the configuration file')
    parser.add_argument('--mode', type=str, default='live',
                        choices=['live', 'backtest', 'paper'],
                        help='Trading mode: live, backtest, or paper')
    parser.add_argument('--date', type=str,
                        help='Date for backtest mode (YYYY-MM-DD)')

    return parser.parse_args()


def setup_backtest_params(args, config):
    """
    Set up parameters for backtest mode.

    Args:
        args: Command line arguments
        config: Configuration dictionary

    Returns:
        Updated configuration dictionary with backtest parameters
    """
    if args.mode != 'backtest':
        return config

    # Get current date
    current_date = datetime.now().date()

    # Use provided date or default to yesterday
    if args.date:
        try:
            # Parse the provided date
            backtest_date = datetime.strptime(args.date, '%Y-%m-%d').date()
            
            # Check if the date is in the future
            if backtest_date > current_date:
                logger.warning(f"Provided date {args.date} is in the future. Using yesterday's date instead.")
                backtest_date = current_date - timedelta(days=1)
        except ValueError:
            logger.warning(f"Invalid date format: {args.date}. Using yesterday's date instead.")
            backtest_date = current_date - timedelta(days=1)
    else:
        # Default to yesterday
        backtest_date = current_date - timedelta(days=1)

    # Format the date as string
    backtest_date_str = backtest_date.strftime('%Y-%m-%d')

    # Update the config with backtest parameters
    if 'backtest' not in config:
        config['backtest'] = {}

    config['backtest'].update({
        'date': backtest_date_str,
        'timespan': 'hour',  # 使用小时级数据
        'days_back': 7,      # 回测使用7天数据
        'data_format': 'panel'
    })

    logger.info(f"Set up backtest for date: {backtest_date_str}")
    return config


def setup_directories(config):
    """
    Set up required directories from configuration.

    Args:
        config: Configuration dictionary
    """
    required_dirs = [
        config.get('system', {}).get('cache_dir', 'data/cache'),
        config.get('data', {}).get('data_storage', 'data/market_data'),
        config.get('output', {}).get('results_dir', 'results'),
        config.get('output', {}).get('plots_dir', 'plots'),
        'models/trained',
        'backtest',
        'logs'
    ]

    for directory in required_dirs:
        if directory:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Directory created/verified: {directory}")


def main():
    """
    Main function to run the Alpha Agent Trading System.
    """
    try:
        # Check environment variables
        check_environment()

        # Parse command line arguments
        args = parse_args()

        # Load configuration
        config = load_config(args.config)

        # Set the trading mode
        if 'mode' not in config:
            config['mode'] = args.mode

        # Set up backtest parameters if needed
        if args.mode == 'backtest':
            config = setup_backtest_params(args, config)

        # Set up required directories
        setup_directories(config)

        # Set up communication framework
        unified_comm, conversation_manager = setup_communication_framework()

        # Initialize the agents
        agents = initialize_agents(config, unified_comm, conversation_manager)

        # Run the appropriate mode
        if config['mode'] == 'live' or config['mode'] == 'paper':
            run_live_trading(agents, config, unified_comm, conversation_manager)
        elif config['mode'] == 'backtest':
            run_backtest(agents, config, unified_comm, conversation_manager)
        else:
            logger.error(f"Invalid mode: {config['mode']}")

    except Exception as e:
        logger.error(f"Critical error in main: {e}", exc_info=True)
        raise


def run_live_trading(agents, config, unified_comm, conversation_manager):
    """
    Run the system in live or paper trading mode.

    Args:
        agents: Dictionary of agents
        config: Configuration dictionary
        unified_comm: UnifiedCommunicationManager instance
        conversation_manager: ConversationManager instance
    """
    # Extract the signal agent which coordinates the trading process
    signal_agent = agents['signal_agent']

    try:
        # Start all agent processes
        for agent_name, agent in agents.items():
            if hasattr(agent, 'start'):
                agent.start()

        # Publish system status message
        status_message = Message(
            sender_id="system",
            receiver_id="system_status",
            message_type=MessageType.STATUS,
            content={
                "status": "starting",
                "mode": config['mode'],
                "timestamp": time.time()
            }
        )
        unified_comm.publish("system_status", status_message)

        # Update config with current date
        current_date = datetime.now().date()
        if 'data' not in config:
            config['data'] = {}
        
        # Calculate dates correctly - from 90 days ago to yesterday
        end_date = current_date - timedelta(days=1)  # Use yesterday as end date
        start_date = end_date - timedelta(days=90)   # Go back 90 days from end date
        
        config['data']['start_date'] = start_date.strftime('%Y-%m-%d')
        config['data']['end_date'] = end_date.strftime('%Y-%m-%d')

        # Create and start the market analysis conversation
        market_analysis_id = conversation_manager.create_conversation(
            initiator_id="data_agent",
            participants=["data_agent", "prediction_agent", "risk_agent", "sentiment_agent"],
            topic="market_analysis"
        )
        if market_analysis_id:
            conversation_manager.start_conversation(
                market_analysis_id,
                {
                    "action": "initialize",
                    "config": {
                        "mode": config['mode'],
                        "tickers": config.get('data', {}).get('default_tickers', ['SPY', 'QQQ', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META']),
                        "timespan": config.get('data', {}).get('default_timespan', 'day'),
                        "start_date": config['data']['start_date'],
                        "end_date": config['data']['end_date']
                    }
                }
            )

        # Create and start the trading decision conversation
        trading_decision_id = conversation_manager.create_conversation(
            initiator_id="prediction_agent",
            participants=["prediction_agent", "risk_agent", "sentiment_agent", "signal_agent"],
            topic="trading_decision"
        )
        if trading_decision_id:
            conversation_manager.start_conversation(
                trading_decision_id,
                {
                    "action": "initialize_trading",
                    "config": {
                        "mode": config['mode'],
                        "risk_params": config['risk'],
                        "trading_params": config['trading'],
                        "start_date": config['data']['start_date'],
                        "end_date": config['data']['end_date']
                    }
                }
            )

        # Start the signal agent's trading loop
        signal_agent.run_trading_loop()

    except KeyboardInterrupt:
        logger.info("Trading system interrupted by user")
        shutdown_message = Message(
            sender_id="system",
            receiver_id="system_status",
            message_type=MessageType.STATUS,
            content={"status": "shutdown_requested"}
        )
        unified_comm.publish("system_status", shutdown_message)
    except Exception as e:
        logger.error(f"Error in live trading: {e}")
        error_message = Message(
            sender_id="system",
            receiver_id="system_status",
            message_type=MessageType.ERROR,
            content={"status": "error", "message": str(e)}
        )
        unified_comm.publish("system_status", error_message)
    finally:
        # Publish shutdown message
        final_message = Message(
            sender_id="system",
            receiver_id="system_status",
            message_type=MessageType.STATUS,
            content={"status": "shutting_down"}
        )
        unified_comm.publish("system_status", final_message)

        # Stop all agent processes
        for agent_name, agent in agents.items():
            if hasattr(agent, 'stop'):
                agent.stop()

        # Close communication channels
        unified_comm.shutdown()
        conversation_manager.shutdown()

        logger.info("Trading system shutdown complete")


def run_backtest(agents, config, unified_comm, conversation_manager):
    """
    Run the system in backtest mode.

    Args:
        agents: Dictionary of agents
        config: Configuration dictionary
        unified_comm: UnifiedCommunicationManager instance
        conversation_manager: ConversationManager instance
    """
    try:
        # Start the backtest
        logger.info(f"Starting backtest for date: {config['backtest']['date']}")

        # Initialize results storage
        backtest_results = {
            'metadata': {
                'start_time': datetime.now().isoformat(),
                'backtest_date': config['backtest']['date'],
                'config': config,
                'status': 'running'
            },
            'market_data': {},
            'agent_results': {
                'data_agent': {'data': {}, 'scores': {}, 'messages': []},
                'prediction_agent': {'predictions': {}, 'scores': {}, 'messages': []},
                'risk_agent': {'risk_metrics': {}, 'scores': {}, 'messages': []},
                'sentiment_agent': {'sentiment_data': {}, 'scores': {}, 'messages': []},
                'signal_agent': {'signals': {}, 'scores': {}, 'messages': []}
            },
            'conversations': {},
            'performance': {},
            'trades': [],
            'errors': []
        }

        # Publish system status message
        status_message = Message(
            sender_id="system",
            receiver_id="system_status",
            message_type=MessageType.STATUS,
            content={
                "status": "starting_backtest",
                "date": config['backtest']['date'],
                "timestamp": datetime.now().isoformat()
            }
        )
        unified_comm.publish("system_status", status_message)

        # Initialize backtest components
        data_fetcher = HistoricalDataFetcher(api_client=agents['data_agent'].api)
        backtest_engine = BacktestEngine(
            initial_capital=config.get('trading', {}).get('initial_capital', 100000),
            commission=config.get('trading', {}).get('commission', 0.001),
            slippage=config.get('trading', {}).get('slippage', 0.001),
            lot_size=config.get('trading', {}).get('lot_size', 100),
            leverage=config.get('trading', {}).get('leverage', 1.0),
            risk_free_rate=config.get('backtest', {}).get('risk_free_rate', 0.02)
        )

        # Get backtest date range
        backtest_date = datetime.strptime(config['backtest']['date'], '%Y-%m-%d').date()
        current_date = datetime.now().date()

        # Ensure we're not trying to backtest with future dates
        if backtest_date > current_date:
            logger.warning(f"Backtest date {backtest_date} is in the future. Using yesterday's date instead.")
            backtest_date = current_date - timedelta(days=1)
            config['backtest']['date'] = backtest_date.strftime('%Y-%m-%d')

        # Calculate date range
        end_date = min(backtest_date, current_date)  # Ensure end date is not in the future
        start_date = end_date - timedelta(days=90)   # Go back 90 days from end date
        
        # Format dates as strings
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')

        # Get tickers from config
        tickers = config.get('data', {}).get('default_tickers', ['SPY', 'QQQ', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META'])

        # Fetch historical data
        logger.info(f"Fetching historical data for {tickers} from {start_date_str} to {end_date_str}")
        stock_data = data_fetcher.fetch_stock_history(
            tickers=tickers,
            start_date=start_date_str,
            end_date=end_date_str,
            timespan='day'
        )

        # Store market data
        backtest_results['market_data'] = {
            'raw_data': {ticker: data.to_dict('records') for ticker, data in stock_data.items()},
            'metadata': {
                'tickers': tickers,
                'start_date': start_date_str,
                'end_date': end_date_str,
                'timespan': 'day'
            }
        }

        # Create dataset and prepare for backtesting
        dataset = {'stocks': stock_data}
        backtest_data = data_fetcher.prepare_backtest_data(dataset, format_type='panel')

        # Start the conversation flow for backtest initialization
        market_analysis_id = conversation_manager.create_conversation(
            initiator_id="data_agent",
            participants=["data_agent", "prediction_agent", "risk_agent", "sentiment_agent"],
            topic="market_analysis"
        )
        
        if market_analysis_id:
            # Store conversation ID
            backtest_results['conversations']['market_analysis'] = market_analysis_id
            
            # Start conversation and store initial message
            init_message = {
                "action": "initialize_backtest",
                "backtest_date": config['backtest']['date'],
                "config": config
            }
            conversation_manager.start_conversation(market_analysis_id, init_message)
            backtest_results['agent_results']['data_agent']['messages'].append({
                'conversation_id': market_analysis_id,
                'content': init_message,
                'timestamp': datetime.now().isoformat()
            })

        # Initialize strategies
        strategies = {
            'ma_cross': MovingAverageCrossStrategy(
                fast_period=config.get('strategy', {}).get('ma_fast', 50),
                slow_period=config.get('strategy', {}).get('ma_slow', 200)
            ),
            'rsi': RSIStrategy(
                period=config.get('strategy', {}).get('rsi_period', 14),
                oversold=config.get('strategy', {}).get('rsi_oversold', 30),
                overbought=config.get('strategy', {}).get('rsi_overbought', 70)
            ),
            'macd': MACDStrategy(
                fast_period=config.get('strategy', {}).get('macd_fast', 12),
                slow_period=config.get('strategy', {}).get('macd_slow', 26),
                signal_period=config.get('strategy', {}).get('macd_signal', 9)
            )
        }

        # Store strategy configurations
        backtest_results['metadata']['strategies'] = {
            name: strategy.get_parameters() for name, strategy in strategies.items()
        }

        # Create combined strategy
        strategy = create_combined_strategy(list(strategies.values()))

        # Run backtest
        engine_results = backtest_engine.run_backtest(
            data=backtest_data,
            strategy=strategy.generate_signals
        )

        # Store trading results
        backtest_results['trades'] = engine_results['trades'].to_dict('records') if not engine_results['trades'].empty else []

        # Initialize performance analyzer
        analyzer = PerformanceAnalyzer()

        # Calculate benchmark returns if available
        benchmark_returns = None
        if 'SPY' in stock_data:
            spy_data = stock_data['SPY']
            spy_data['date'] = pd.to_datetime(spy_data['date'])
            spy_data.set_index('date', inplace=True)
            benchmark_returns = spy_data['close'].pct_change().dropna()

        # Calculate performance metrics
        portfolio_history = engine_results['portfolio_history']
        portfolio_history['date'] = pd.to_datetime(portfolio_history['date'])
        portfolio_history.set_index('date', inplace=True)
        returns = portfolio_history['portfolio_value'].pct_change().dropna()

        performance_metrics = analyzer.calculate_metrics(
            returns=returns,
            benchmark_returns=benchmark_returns,
            risk_free_rate=config.get('backtest', {}).get('risk_free_rate', 0.02)
        )

        # Store performance results
        backtest_results['performance'] = {
            'portfolio_history': engine_results['portfolio_history'].to_dict('records'),
            'metrics': {
                **engine_results['metrics'],
                **performance_metrics
            },
            'returns': returns.to_dict(),
            'benchmark_returns': benchmark_returns.to_dict() if benchmark_returns is not None else None
        }

        # Update metadata
        backtest_results['metadata']['end_time'] = datetime.now().isoformat()
        backtest_results['metadata']['status'] = 'completed'

        # Save results
        output_dir = config.get('output', {}).get('results_dir', 'results')
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = os.path.join(output_dir, f"backtest_results_{timestamp}.json")
        
        with open(results_file, 'w') as f:
            json.dump(backtest_results, f, indent=2)
        
        logger.info(f"Backtest results saved to {results_file}")

        # Generate performance plots
        plots_dir = config.get('output', {}).get('plots_dir', 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        plot_file = os.path.join(plots_dir, f"backtest_performance_{timestamp}.png")
        
        analyzer.plot_returns_analysis(
            returns=returns,
            benchmark_returns=benchmark_returns,
            save_path=plot_file
        )
        
        logger.info(f"Performance plot saved to {plot_file}")

        # Output key metrics
        logger.info("\nPerformance Metrics:")
        logger.info(f"Total Return: {backtest_results['performance']['metrics']['total_return']:.2%}")
        logger.info(f"Annualized Return: {backtest_results['performance']['metrics']['annualized_return']:.2%}")
        logger.info(f"Sharpe Ratio: {backtest_results['performance']['metrics']['sharpe_ratio']:.2f}")
        logger.info(f"Max Drawdown: {backtest_results['performance']['metrics']['max_drawdown']:.2%}")
        logger.info(f"Sortino Ratio: {backtest_results['performance']['metrics'].get('sortino_ratio', 'N/A')}")

        if benchmark_returns is not None:
            logger.info(f"Alpha: {backtest_results['performance']['metrics'].get('alpha', 'N/A'):.4f}")
            logger.info(f"Beta: {backtest_results['performance']['metrics'].get('beta', 'N/A'):.4f}")

        return backtest_results

    except Exception as e:
        logger.error(f"Error in backtest: {e}", exc_info=True)
        error_message = Message(
            sender_id="system",
            receiver_id="system_status",
            message_type=MessageType.ERROR,
            content={
                "status": "error",
                "message": str(e)
            }
        )
        unified_comm.publish("system_status", error_message)
        
        # Store error in results
        if 'backtest_results' in locals():
            backtest_results['metadata']['status'] = 'error'
            backtest_results['metadata']['end_time'] = datetime.now().isoformat()
            backtest_results['errors'].append({
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'traceback': traceback.format_exc()
            })
            
            # Try to save partial results
            try:
                output_dir = config.get('output', {}).get('results_dir', 'results')
                os.makedirs(output_dir, exist_ok=True)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                results_file = os.path.join(output_dir, f"backtest_results_error_{timestamp}.json")
                
                with open(results_file, 'w') as f:
                    json.dump(backtest_results, f, indent=2)
                
                logger.info(f"Partial results saved to {results_file}")
            except Exception as save_error:
                logger.error(f"Failed to save partial results: {save_error}")
        
        return {'error': str(e)}
    finally:
        # Close communication channels
        unified_comm.shutdown()
        conversation_manager.shutdown()
        logger.info("Backtest system shutdown complete")


if __name__ == '__main__':
    main()

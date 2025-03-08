import argparse
import logging
import os
import yaml
from datetime import datetime, timedelta, date
from dotenv import load_dotenv
import json
import pandas as pd
import traceback
import asyncio

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
from utils.communication.message import Message, MessageType
from backtest.historical_data_fetcher import HistoricalDataFetcher
from backtest.backtest_engine import BacktestEngine
from backtest.performance_metrics import PerformanceAnalyzer
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
        
    Raises:
        FileNotFoundError: If config file does not exist
        Exception: If there is an error loading the config file
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Initialize api_keys if not present
        if 'api_keys' not in config:
            config['api_keys'] = {}
            
        # Handle API keys with environment variables taking precedence
        config['api_keys']['openai'] = os.getenv('OPENAI_API_KEY') or config['api_keys'].get('openai')
        config['api_keys']['polygon'] = os.getenv('POLYGON_API_KEY') or config['api_keys'].get('polygon')
        
        # Validate API keys
        missing_keys = []
        for key_name, key_value in config['api_keys'].items():
            if not key_value:
                missing_keys.append(key_name)
        
        if missing_keys:
            raise ValueError(f"Missing required API keys: {', '.join(missing_keys)}")
        
        logger.info(f"Config loaded from {config_path}")
        return config
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found at {config_path}")
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


async def initialize_agents(config, unified_comm, conversation_manager):
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
        config={
            'api_key': config['api_keys']['polygon'],
            'config_path': "configs/config.yaml"
        },
        communicator=unified_comm
    )
    
    prediction_agent = PredictionAgent(
        config=config,
        communicator=unified_comm
    )
    
    risk_agent = RiskAgent(
        config={
            'polygon_api_key': config['api_keys']['polygon']
        },
        communicator=unified_comm
    )
    
    sentiment_agent = SentimentAgent(
        config={
            'api_key': config['api_keys']['openai'],
            'polygon_api_key': config['api_keys']['polygon']
        },
        communicator=unified_comm
    )
    
    signal_agent = SignalAgent(
        agent_id="signal_agent",
        config=config,
        communicator=unified_comm
    )

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

    # Start all agents
    for agent_id, agent in agents.items():
        if hasattr(agent, 'start') and callable(agent.start):
            await agent.start()

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
    # Define the communication topics
    topics = [
        "market_data",          # Raw market data
        "predictions",          # Price predictions
        "risk_metrics",         # Risk assessments
        "sentiment_analysis",   # Market sentiment
        "trading_signals",      # Trading decisions
        "system_status",        # System health and monitoring
        "portfolio_updates"     # Portfolio changes and performance
    ]

    # Set up topics in the communication manager
    for topic in topics:
        unified_comm.create_topic(topic)

    # Define agent subscriptions
    subscriptions = {
        "data_agent": [
            "market_data",
            "system_status"
        ],
        "prediction_agent": [
            "market_data",
            "predictions",
            "sentiment_analysis"
        ],
        "risk_agent": [
            "market_data",
            "predictions",
            "portfolio_updates",
            "trading_signals"
        ],
        "sentiment_agent": [
            "market_data",
            "sentiment_analysis"
        ],
        "signal_agent": [
            "market_data",
            "predictions",
            "risk_metrics",
            "sentiment_analysis",
            "trading_signals",
            "portfolio_updates"
        ]
    }

    # Subscribe each agent to their topics
    for agent_id, topic_list in subscriptions.items():
        for topic in topic_list:
            if not unified_comm.subscribe_to_topic(agent_id, topic):
                logger.error(f"Failed to subscribe {agent_id} to topic: {topic}")
            else:
                logger.info(f"Successfully subscribed {agent_id} to topic: {topic}")

    # Set up conversation flows
    conversations = [
        {
            "name": "market_analysis",
            "initiator": "data_agent",
            "participants": ["data_agent", "prediction_agent", "risk_agent", "sentiment_agent"],
            "description": "Analyze market conditions and generate predictions"
        },
        {
            "name": "signal_analysis",
            "initiator": "signal_agent",
            "participants": ["prediction_agent", "risk_agent", "sentiment_agent", "signal_agent"],
            "description": "Analyze signals and generate trading decisions based on all available information"
        },
        {
            "name": "risk_assessment",
            "initiator": "risk_agent",
            "participants": ["risk_agent", "data_agent", "prediction_agent"],
            "description": "Continuous risk monitoring and assessment"
        }
    ]

    # Create conversations in the conversation manager
    for conv in conversations:
        conv_id = conversation_manager.create_conversation(
            initiator_id=conv["initiator"],
            participants=conv["participants"],
            topic=conv["name"]
        )
        if not conv_id:
            logger.error(f"Failed to create conversation: {conv['name']}")
        else:
            logger.info(f"Created conversation: {conv['name']}")

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
        agents = asyncio.run(initialize_agents(config, unified_comm, conversation_manager))

        # Run the appropriate mode
        if config['mode'] == 'live' or config['mode'] == 'paper':
            asyncio.run(run_live_trading(agents, config, unified_comm, conversation_manager))
        elif config['mode'] == 'backtest':
            asyncio.run(run_backtest(agents, config, unified_comm, conversation_manager))
        else:
            logger.error(f"Invalid mode: {config['mode']}")

    except Exception as e:
        logger.error(f"Critical error in main: {e}", exc_info=True)
        raise


async def run_live_trading(agents, config, unified_comm, conversation_manager):
    """
    Run live trading with multi-agent coordination.

    Args:
        agents: Dictionary of initialized agents
        config: Configuration dictionary
        unified_comm: UnifiedCommunicationManager instance
        conversation_manager: ConversationManager instance
    """
    logger.info("Starting live trading session")
    
    # Extract trading parameters from config
    trading_params = config.get('trading', {})
    symbols = trading_params.get('symbols', ['AAPL', 'GOOGL', 'MSFT'])  # Default symbols if not specified
    update_interval = trading_params.get('update_interval', 60)  # Default 60 seconds
    risk_threshold = trading_params.get('risk_threshold', 0.7)  # Default risk threshold
    
    try:
        while True:
            for symbol in symbols:
                try:
                    # 1. Fetch market data
                    market_data = await agents['data_agent'].fetch_stock_data(
                        tickers=[symbol],
                        timespan='hour',  # 使用小时级数据
                        days_back=7  # 获取7天数据
                    )
                    await unified_comm.publish_to_topic('market_data', {
                        'symbol': symbol,
                        'data': market_data,
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    # 2. Get sentiment analysis
                    sentiment_msg = Message(
                        sender_id='system',
                        receiver_id='sentiment_agent',
                        content={
                            'action': 'analyze_sentiment',
                            'ticker': symbol,
                            'force_update': True
                        }
                    )
                    sentiment_result = await agents['sentiment_agent'].process_message(sentiment_msg)
                    await unified_comm.publish_to_topic('sentiment_analysis', sentiment_result.content)
                    
                    # 3. Get price predictions
                    prediction_msg = Message(
                        sender_id='system',
                        receiver_id='prediction_agent',
                        content={
                            'action': 'predict_price',
                            'ticker': symbol,
                            'market_data': market_data
                        }
                    )
                    prediction_result = await agents['prediction_agent'].process_message(prediction_msg)
                    await unified_comm.publish_to_topic('predictions', prediction_result.content)
                    
                    # 4. Assess risk
                    risk_msg = Message(
                        sender_id='system',
                        receiver_id='risk_agent',
                        content={
                            'action': 'assess_market_risk',
                            'ticker': symbol,
                            'market_data': market_data,
                            'predictions': prediction_result.content,
                            'sentiment': sentiment_result.content
                        }
                    )
                    risk_result = await agents['risk_agent'].process_message(risk_msg)
                    await unified_comm.publish_to_topic('risk_metrics', risk_result.content)
                    
                    # 5. Generate trading signal
                    signal_msg = Message(
                        sender_id='system',
                        receiver_id='signal_agent',
                        content={
                            'action': 'generate_signal',
                            'ticker': symbol,
                            'market_data': market_data,
                            'predictions': prediction_result.content,
                            'risk_metrics': risk_result.content,
                            'sentiment': sentiment_result.content
                        }
                    )
                    signal_result = await agents['signal_agent'].process_message(signal_msg)
                    
                    # 6. Execute trading decision if risk is acceptable
                    risk_score = risk_result.content.get('market_risk_assessment', {}).get('overall_risk_score', 1.0)
                    if risk_score < risk_threshold:
                        await unified_comm.publish_to_topic('trading_signals', signal_result.content)
                        # Execute trade based on signal
                        if signal_result.content.get('signal') in ['buy', 'sell']:
                            logger.info(f"Executing trade for {symbol}: {signal_result.content}")
                            # Here you would implement actual trade execution
                            # For now, we just log the decision
                    else:
                        logger.warning(f"Risk too high for {symbol}, skipping trade execution")
                    
                    # 7. Update portfolio status
                    await unified_comm.publish_to_topic('portfolio_updates', {
                        'symbol': symbol,
                        'timestamp': datetime.now().isoformat(),
                        'status': 'updated',
                        'risk_score': risk_score,
                        'signal': signal_result.content.get('signal')
                    })
                    
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {str(e)}")
                    logger.error(traceback.format_exc())
                    continue
            
            # Wait for next update interval
            await asyncio.sleep(update_interval)
            
    except KeyboardInterrupt:
        logger.info("Live trading session terminated by user")
    except Exception as e:
        logger.error(f"Live trading session terminated due to error: {str(e)}")
        logger.error(traceback.format_exc())
    finally:
        # Clean up and save final status
        try:
            await unified_comm.publish_to_topic('system_status', {
                'status': 'shutdown',
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")


async def run_backtest(agents, config, unified_comm, conversation_manager):
    """
    Run backtest with multi-agent coordination.

    Args:
        agents: Dictionary of initialized agents
        config: Configuration dictionary
        unified_comm: UnifiedCommunicationManager instance
        conversation_manager: ConversationManager instance
    """
    logger.info("Starting backtest session")
    
    # Extract backtest parameters from config
    backtest_params = config.get('backtest', {})
    symbols = backtest_params.get('symbols', ['AAPL', 'GOOGL', 'MSFT'])
    start_date = backtest_params.get('start_date', (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'))
    end_date = backtest_params.get('end_date', (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'))
    risk_threshold = backtest_params.get('risk_threshold', 0.7)
    initial_capital = backtest_params.get('initial_capital', 100000)
    
    try:
        # Initialize backtest components
        data_fetcher = HistoricalDataFetcher(config['api_keys']['polygon'])
        backtest_engine = BacktestEngine(initial_capital=initial_capital)
        performance_analyzer = PerformanceAnalyzer()
        
        # Store results for each symbol
        results = {}
        
        for symbol in symbols:
            try:
                logger.info(f"Starting backtest for {symbol}")
                
                # 1. Fetch historical data
                historical_data = await data_fetcher.fetch_historical_data(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date
                )
                
                # Process each day in the historical data
                for date, data in historical_data.iterrows():
                    # 2. Get sentiment analysis for the date
                    sentiment_msg = Message(
                        sender_id='system',
                        receiver_id='sentiment_agent',
                        content={
                            'action': 'analyze_sentiment',
                            'ticker': symbol,
                            'date': date.strftime('%Y-%m-%d')
                        }
                    )
                    sentiment_result = await agents['sentiment_agent'].process_message(sentiment_msg)
                    
                    # 3. Get price predictions
                    prediction_msg = Message(
                        sender_id='system',
                        receiver_id='prediction_agent',
                        content={
                            'action': 'predict_price',
                            'ticker': symbol,
                            'market_data': data.to_dict()
                        }
                    )
                    prediction_result = await agents['prediction_agent'].process_message(prediction_msg)
                    
                    # 4. Assess risk
                    risk_msg = Message(
                        sender_id='system',
                        receiver_id='risk_agent',
                        content={
                            'action': 'assess_market_risk',
                            'ticker': symbol,
                            'market_data': data.to_dict(),
                            'predictions': prediction_result.content,
                            'sentiment': sentiment_result.content
                        }
                    )
                    risk_result = await agents['risk_agent'].process_message(risk_msg)
                    
                    # 5. Generate trading signal
                    signal_msg = Message(
                        sender_id='system',
                        receiver_id='signal_agent',
                        content={
                            'action': 'generate_signal',
                            'ticker': symbol,
                            'market_data': data.to_dict(),
                            'predictions': prediction_result.content,
                            'risk_metrics': risk_result.content,
                            'sentiment': sentiment_result.content
                        }
                    )
                    signal_result = await agents['signal_agent'].process_message(signal_msg)
                    
                    # 6. Execute backtest trade if risk is acceptable
                    risk_score = risk_result.content.get('market_risk_assessment', {}).get('overall_risk_score', 1.0)
                    if risk_score < risk_threshold:
                        signal = signal_result.content.get('signal')
                        if signal in ['buy', 'sell']:
                            # Execute trade in backtest engine
                            backtest_engine.execute_trade(
                                symbol=symbol,
                                action=signal,
                                price=data['close'],
                                date=date,
                                quantity=backtest_engine.calculate_position_size(
                                    price=data['close'],
                                    risk_score=risk_score
                                )
                            )
                    
                    # Update portfolio status for the day
                    portfolio_status = backtest_engine.get_portfolio_status()
                    await unified_comm.publish_to_topic('portfolio_updates', {
                        'symbol': symbol,
                        'date': date.strftime('%Y-%m-%d'),
                        'status': 'updated',
                        'portfolio': portfolio_status
                    })
                
                # Calculate performance metrics for this symbol
                symbol_performance = performance_analyzer.calculate_metrics(
                    backtest_engine.get_trade_history(symbol),
                    historical_data
                )
                results[symbol] = symbol_performance
                
                logger.info(f"Completed backtest for {symbol}")
                
            except Exception as e:
                logger.error(f"Error in backtest for {symbol}: {str(e)}")
                logger.error(traceback.format_exc())
                continue
        
        # Generate and save final backtest report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = os.path.join(
            config.get('output', {}).get('results_dir', 'results'),
            f'backtest_report_{timestamp}.json'
        )
        
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump({
                'parameters': {
                    'symbols': symbols,
                    'start_date': start_date,
                    'end_date': end_date,
                    'initial_capital': initial_capital,
                    'risk_threshold': risk_threshold
                },
                'results': results,
                'portfolio_summary': backtest_engine.get_portfolio_summary(),
                'timestamp': timestamp
            }, f, indent=4)
        
        logger.info(f"Backtest report saved to {report_path}")
        
    except Exception as e:
        logger.error(f"Backtest session terminated due to error: {str(e)}")
        logger.error(traceback.format_exc())
    finally:
        # Clean up and save final status
        try:
            await unified_comm.publish_to_topic('system_status', {
                'status': 'backtest_completed',
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")


if __name__ == '__main__':
    main()

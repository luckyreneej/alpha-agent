import os
import logging
import time
import argparse
from typing import Dict, Any, Optional
import yaml
import json

# Import agent components
from agents.data_agent import DataAgent
from agents.prediction_agent import PredictionAgent
from agents.trading_agent import TradingAgent
from agents.risk_agent import RiskAgent
from agents.sentiment_agent import SentimentAgent
from utils.communication.unified_communication import UnifiedCommunicationManager
from evaluation.agent_evaluator import AgentEvaluator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('alpha_agent.log')
    ]
)
logger = logging.getLogger(__name__)


class AlphaAgentSystem:
    """
    Main class for the Alpha-Agent multi-agent trading system.
    Orchestrates all agents and manages the overall system.
    """

    def __init__(self, config_path: str = 'configs/config.yaml'):
        """
        Initialize the Alpha-Agent system.

        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.communicator = UnifiedCommunicationManager()
        self.agents = {}
        self.evaluator = AgentEvaluator()
        self.running = False

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to configuration file

        Returns:
            Configuration dictionary
        """
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            # Return default configuration
            return {
                'api_keys': {
                    'polygon': os.environ.get('POLYGON_API_KEY', ''),
                    'openai': os.environ.get('OPENAI_API_KEY', '')
                },
                'system': {
                    'update_interval': 60,
                    'evaluation_interval': 3600
                },
                'data': {
                    'default_tickers': ['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL']
                },
                'prediction': {
                    'default_model': 'ensemble',
                    'use_alpha_factors': True
                },
                'trading': {
                    'initial_capital': 100000,
                    'max_position_size': 0.1,
                    'risk_per_trade': 0.02
                },
                'risk': {
                    'max_drawdown': 0.2,
                    'var_confidence': 0.95
                }
            }

    def initialize_agents(self):
        """
        Initialize all agent components.
        """
        try:
            # Start communication manager
            self.communicator.start()

            # Initialize data agent
            self.agents['data'] = DataAgent(
                agent_id="data_agent",
                communicator=self.communicator,
                api_key=self.config['api_keys'].get('polygon'),
                config_path=None  # Use default config
            )

            # Initialize prediction agent
            self.agents['prediction'] = PredictionAgent(
                agent_id="prediction_agent",
                communicator=self.communicator,
                config=self.config.get('prediction')
            )

            # Initialize trading agent
            self.agents['trading'] = TradingAgent(
                agent_id="trading_agent",
                communicator=self.communicator,
                config=self.config.get('trading')
            )

            # Initialize risk agent
            self.agents['risk'] = RiskAgent(
                agent_id="risk_agent",
                communicator=self.communicator,
                config=self.config.get('risk')
            )

            # Initialize sentiment agent if OpenAI API key is available
            if self.config['api_keys'].get('openai'):
                self.agents['sentiment'] = SentimentAgent(
                    agent_id="sentiment_agent",
                    communicator=self.communicator,
                    api_key=self.config['api_keys'].get('openai')
                )

            logger.info("All agents initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing agents: {e}")
            raise

    def start(self):
        """
        Start all agents and the overall system.
        """
        if self.running:
            logger.warning("System is already running")
            return

        try:
            logger.info("Starting Alpha-Agent system")

            # Initialize agents if not already done
            if not self.agents:
                self.initialize_agents()

            # Start each agent
            for name, agent in self.agents.items():
                logger.info(f"Starting {name} agent")
                agent.start()

            self.running = True
            logger.info("Alpha-Agent system started successfully")

            # Start evaluation loop in the background
            self._schedule_evaluations()

        except Exception as e:
            logger.error(f"Error starting system: {e}")
            self.stop()
            raise

    def stop(self):
        """
        Stop all agents and the overall system.
        """
        logger.info("Stopping Alpha-Agent system")

        # Stop all agents
        for name, agent in self.agents.items():
            try:
                logger.info(f"Stopping {name} agent")
                agent.stop()
            except Exception as e:
                logger.error(f"Error stopping {name} agent: {e}")

        # Stop communication manager
        try:
            self.communicator.stop()
        except Exception as e:
            logger.error(f"Error stopping communication manager: {e}")

        self.running = False
        logger.info("Alpha-Agent system stopped")

    def get_recommendations(self) -> Dict[str, Any]:
        """
        Get current trading recommendations from the system.

        Returns:
            Dictionary of trading recommendations
        """
        if not self.running:
            logger.warning("System is not running")
            return {"error": "System not running"}

        # Get latest predictions
        predictions = self.communicator.get_data('stock_price_prediction')

        # Get current positions
        positions = self.communicator.get_data('current_positions') or {}

        # Get trade suggestions from trading agent
        trading_agent = self.agents.get('trading')
        if trading_agent:
            recommendations = trading_agent.get_trading_signals()
        else:
            recommendations = {}

        return {
            "timestamp": time.time(),
            "predictions": predictions,
            "current_positions": positions,
            "recommendations": recommendations
        }

    def _schedule_evaluations(self):
        """
        Schedule periodic system evaluations.
        """
        import threading

        def evaluation_loop():
            while self.running:
                try:
                    self._run_evaluation()
                except Exception as e:
                    logger.error(f"Error in evaluation loop: {e}")

                # Sleep until next evaluation
                evaluation_interval = self.config['system'].get('evaluation_interval', 3600)
                time.sleep(evaluation_interval)

        # Start evaluation thread
        evaluation_thread = threading.Thread(target=evaluation_loop, daemon=True)
        evaluation_thread.start()

    def _run_evaluation(self):
        """
        Run a full system evaluation.
        """
        logger.info("Running system evaluation")

        try:
            # Get prediction performance
            prediction_agent = self.agents.get('prediction')
            if prediction_agent:
                predictions = self.communicator.get_data('model_predictions') or {}
                actual_values = self.communicator.get_data('actual_values') or {}

                if predictions and actual_values:
                    self.evaluator.evaluate_prediction_agent(predictions, actual_values)

            # Get trading performance
            trading_agent = self.agents.get('trading')
            if trading_agent:
                trades = self.communicator.get_data('executed_trades') or []
                portfolio_history = self.communicator.get_data('portfolio_history') or []

                if trades and portfolio_history:
                    self.evaluator.evaluate_trading_agent(trades, portfolio_history)

            # Get communication metrics
            message_logs = self.communicator.get_message_history(None)  # Get all messages
            if message_logs:
                self.evaluator.evaluate_communication(message_logs, time_period='day')

            # Get runtime metrics (simple example)
            import psutil
            process = psutil.Process(os.getpid())
            runtime_metrics = {
                'memory_usage_mb': process.memory_info().rss / (1024 * 1024),
                'cpu_usage_percent': process.cpu_percent(interval=1)
            }

            self.evaluator.evaluate_system_performance(runtime_metrics)

            # Save evaluation results periodically
            self.evaluator.save_metrics(save_plots=True)

            logger.info("System evaluation completed")

        except Exception as e:
            logger.error(f"Error during system evaluation: {e}")

    def run_backtest(self, backtest_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a backtest using the Alpha-Agent strategies.

        Args:
            backtest_config: Backtest configuration parameters

        Returns:
            Dictionary of backtest results
        """
        from backtest.backtest_engine import BacktestEngine
        from backtest.historical_data_fetcher import HistoricalDataFetcher

        logger.info("Setting up backtesting environment")

        try:
            # Create data fetcher
            data_fetcher = HistoricalDataFetcher(
                api_key=self.config['api_keys'].get('polygon')
            )

            # Fetch historical data
            dataset = data_fetcher.fetch_complete_dataset(
                tickers=backtest_config.get('tickers', self.config['data']['default_tickers']),
                start_date=backtest_config.get('start_date', '2022-01-01'),
                end_date=backtest_config.get('end_date', '2022-12-31'),
                include_options=backtest_config.get('include_options', False),
                include_news=backtest_config.get('include_news', True)
            )

            # Prepare data for backtesting
            backtest_data = data_fetcher.prepare_backtest_data(
                dataset,
                format_type=backtest_config.get('format_type', 'panel'),
                resample_freq=backtest_config.get('resample_freq')
            )

            # Create backtest engine
            backtester = BacktestEngine(
                initial_capital=backtest_config.get('initial_capital',
                                                    self.config['trading'].get('initial_capital', 100000)),
                commission=backtest_config.get('commission', 0.001),
                slippage=backtest_config.get('slippage', 0.001)
            )

            # Get strategy function
            strategy_function = self._get_strategy_function(backtest_config.get('strategy'))

            # Run backtest
            results = backtester.run_backtest(
                data=backtest_data,
                strategy=strategy_function,
                strategy_params=backtest_config.get('strategy_params', {})
            )

            # Generate plots
            if backtest_config.get('generate_plots', True):
                plots_dir = os.path.join('evaluation', 'backtest_plots')
                os.makedirs(plots_dir, exist_ok=True)

                backtester.plot_portfolio_performance(
                    save_path=os.path.join(plots_dir, 'portfolio_performance.png')
                )

                backtester.plot_trade_analysis(
                    save_path=os.path.join(plots_dir, 'trade_analysis.png')
                )

            logger.info("Backtest completed successfully")
            return results

        except Exception as e:
            logger.error(f"Error during backtesting: {e}")
            return {"error": str(e)}

    def _get_strategy_function(self, strategy_name: Optional[str] = None):
        """
        Get the appropriate strategy function for backtesting.

        Args:
            strategy_name: Name of the strategy to use

        Returns:
            Strategy function for backtesting
        """

        # Default strategy using alpha factors
        def alpha_factor_strategy(historical_data, current_day, **params):
            """
            Strategy based on alpha factors.

            Args:
                historical_data: DataFrame with historical price data
                current_day: Current day's data
                **params: Additional strategy parameters

            Returns:
                Dictionary of trading signals
            """
            from models.signals.alpha_factors import AlphaFactors

            # Get parameters
            lookback = params.get('lookback', 60)  # Days of history to use
            threshold = params.get('threshold', 0.8)  # Signal threshold
            top_n = params.get('top_n', 3)  # Number of top stocks to select
            position_size = params.get('position_size', 0.3)  # Position size per stock

            # Calculate alpha factors
            alpha_calculator = AlphaFactors()
            relevant_history = historical_data.iloc[-lookback:].copy() if len(
                historical_data) > lookback else historical_data.copy()

            # Get unique tickers
            if 'ticker' in relevant_history.columns:
                tickers = relevant_history['ticker'].unique()
            elif isinstance(relevant_history.index, pd.MultiIndex):
                tickers = relevant_history.index.get_level_values('ticker').unique()
            else:
                # Assume it's a dictionary of ticker -> DataFrame
                tickers = list(relevant_history.keys())

            signals = {}

            # Calculate alpha values for each ticker
            for ticker in tickers:
                try:
                    # Extract ticker data
                    if isinstance(relevant_history, dict):
                        ticker_data = relevant_history[ticker]
                    elif 'ticker' in relevant_history.columns:
                        ticker_data = relevant_history[relevant_history['ticker'] == ticker]
                    elif isinstance(relevant_history.index, pd.MultiIndex):
                        ticker_data = relevant_history.xs(ticker, level='ticker')

                    # Calculate combined alpha signal
                    if not ticker_data.empty:
                        with_alphas = alpha_calculator.calculate_alpha_factors(
                            ticker_data,
                            selected_factors=params.get('factors', ['alpha1', 'alpha12', 'alpha101'])
                        )

                        # Use last row for current signals
                        last_row = with_alphas.iloc[-1] if len(with_alphas) > 0 else None

                        if last_row is not None:
                            # Get alpha values
                            alpha_cols = [col for col in with_alphas.columns if col.startswith('alpha')]
                            if alpha_cols:
                                # Normalize alpha values
                                alpha_values = [last_row[col] for col in alpha_cols]
                                normalized = [(val - np.mean(alpha_values)) / np.std(alpha_values) if np.std(
                                    alpha_values) > 0 else 0
                                              for val in alpha_values]

                                # Calculate combined signal (-1 to 1)
                                combined_signal = np.mean(normalized)

                                # Apply threshold for signal strength
                                if abs(combined_signal) > threshold:
                                    signals[ticker] = combined_signal
                except Exception as e:
                    logger.error(f"Error calculating alphas for {ticker}: {e}")

            # Select top N stocks based on absolute signal strength
            if signals:
                sorted_signals = sorted(signals.items(), key=lambda x: abs(x[1]), reverse=True)
                top_signals = sorted_signals[:top_n]

                # Generate final signals (-1 to 1 scale)
                return {ticker: np.sign(signal) * position_size for ticker, signal in top_signals}

            return {}  # No signals generated

        # Map of available strategies
        strategies = {
            'alpha_factor': alpha_factor_strategy,
            # Add more strategies here
        }

        # Return the requested strategy or the default
        return strategies.get(strategy_name, alpha_factor_strategy)


def start_system(config_path: str = 'configs/config.yaml'):
    """
    Convenience function to start the Alpha-Agent system.

    Args:
        config_path: Path to configuration file

    Returns:
        AlphaAgentSystem instance
    """
    system = AlphaAgentSystem(config_path=config_path)
    system.start()
    return system


# Command-line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Alpha-Agent: Multi-Agent Trading System')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--backtest', action='store_true',
                        help='Run in backtest mode')
    parser.add_argument('--backtest-config', type=str, default=None,
                        help='Path to backtest configuration JSON file')

    args = parser.parse_args()

    # Create system
    system = AlphaAgentSystem(config_path=args.config)

    # Run in backtest mode if specified
    if args.backtest:
        # Load backtest config
        backtest_config = {}
        if args.backtest_config and os.path.exists(args.backtest_config):
            with open(args.backtest_config, 'r') as f:
                backtest_config = json.load(f)

        # Run backtest
        results = system.run_backtest(backtest_config)
        print(json.dumps(results['metrics'], indent=2))
    else:
        # Start the system in regular mode
        system.start()

        try:
            # Keep the main thread running
            while True:
                time.sleep(60)

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received, stopping system")
            system.stop()
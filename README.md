# Alpha-Agent

## Overview
Alpha-Agent is a sophisticated multi-agent trading system that leverages machine learning, alpha factor selection, and real-time market data to generate trading signals for stocks and options.

## Features

- **Multi-Agent Architecture**: Specialized agents for data, prediction, trading, risk management, and sentiment analysis
- **Alpha Factor Selection**: Implementation of 101 alphas and custom factors with dynamic evaluation
- **Machine Learning Models**: LSTM, XGBoost, and Prophet models for price prediction
- **Options Strategies**: Black-Scholes pricing and advanced options strategy generation
- **Polygon.io Integration**: Real-time and historical data from Polygon.io API
- **Backtesting Framework**: Comprehensive backtesting with detailed performance metrics
- **Agent Communication**: Robust communication protocol inspired by metaGPT

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/alpha-agent.git
cd alpha-agent

# Install dependencies
pip install -r requirements.txt

# Set environment variables for API keys
export POLYGON_API_KEY="your_polygon_api_key"
export OPENAI_API_KEY="your_openai_api_key"  # If using sentiment analysis
```

## Configuration

Edit the configuration file at `configs/config.yaml` to customize:

- Default tickers to monitor
- Data update intervals
- Model hyperparameters
- Trading parameters
- Risk management settings

## Usage

### Basic Usage

```python
# Start the Alpha-Agent system
from alpha_agent.main import start_system

# Start the system with default configuration
start_system()

# Or specify a custom configuration file
start_system(config_path="path/to/custom_config.yaml")
```

### Running Individual Agents

```python
# Run the prediction agent in standalone mode
from agents.prediction_agent import PredictionAgent
from utils.communication.unified_communication import UnifiedCommunicationManager

# Initialize communication manager
communicator = UnifiedCommunicationManager()
communicator.start()

# Create and start prediction agent
prediction_agent = PredictionAgent("prediction_agent", communicator)
prediction_agent.start()
```

### Running Backtests

```python
# Run a backtest
from backtest.backtest_engine import BacktestEngine
from backtest.historical_data_fetcher import HistoricalDataFetcher

# Fetch historical data
data_fetcher = HistoricalDataFetcher(api_key="your_polygon_api_key")
dataset = data_fetcher.fetch_complete_dataset(
    tickers=["AAPL", "MSFT", "GOOGL"],
    start_date="2022-01-01",
    end_date="2022-12-31"
)

# Run backtest
backtester = BacktestEngine(initial_capital=100000)
results = backtester.run_backtest(dataset["stocks"], your_strategy_function)

# Plot results
backtester.plot_portfolio_performance()
backtester.plot_trade_analysis()
```

### Evaluating Alpha Factors

```python
# Evaluate alpha factors
from models.signals.alpha_factors import AlphaFactors
from utils.factor_evaluation.factor_analyzer import FactorAnalyzer

# Calculate alpha factors
alpha_factors = AlphaFactors()
df_with_alphas = alpha_factors.calculate_alpha_factors(historical_data)

# Evaluate factor performance
analyzer = FactorAnalyzer()
metrics = analyzer.calculate_factor_metrics(
    data=df_with_alphas,
    factors=["alpha1", "alpha12", "alpha101"],
    forward_returns_periods=[1, 5, 10]
)

# Get best factors
best_factors = analyzer.get_best_factors(metric="ic", n=5)
print(f"Best factors: {best_factors}")
```

### Adding Custom Alpha Factors

```python
from models.signals.alpha_factors import AlphaFactors


class ExtendedAlphaFactors(AlphaFactors):
    def custom_momentum(self, df):
        """Custom momentum alpha factor."""
        return df['close'].pct_change(20)  # 20-day momentum


# Use your custom implementation
my_factors = ExtendedAlphaFactors()
df_with_alphas = my_factors.calculate_alpha_factors(data)
```

## System Architecture

```
+------------------+    +------------------+    +-------------------+
|                  |    |                  |    |                   |
|    Data Agent    |<-->| Prediction Agent |<-->|   Trading Agent   |
|                  |    |                  |    |                   |
+------------------+    +------------------+    +-------------------+
        ^                       ^                       ^
        |                       |                       |
        v                       v                       v
+------------------+    +------------------+    +-------------------+
|                  |    |                  |    |                   |
|  Sentiment Agent |<-->| Communication    |<-->|    Risk Agent     |
|                  |    |    Manager       |    |                   |
+------------------+    +------------------+    +-------------------+
```

## Evaluation Metrics

The system provides comprehensive evaluation metrics for both individual agents and the overall system:

- **Prediction Accuracy**: RMSE, MAE, Directional Accuracy
- **Trading Performance**: Returns, Sharpe, Sortino, Maximum Drawdown
- **Factor Performance**: IC, Turnover, Half-life, Factor correlation
- **System Performance**: Agent communication efficiency, runtime metrics

## Documentation

Detailed documentation is available in the `docs/` directory:

- System architecture and design: `docs/README.md`
- API reference documentation: `docs/api_reference.md`
- Factor analysis guide: `docs/factor_analysis.md`
- Backtesting guide: `docs/backtesting.md`

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

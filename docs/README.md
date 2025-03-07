# Alpha-Agent: Enhanced Multi-Agent Trading System

## Overview
Alpha-Agent is a sophisticated multi-agent system designed for algorithmic trading with a focus on stock selection, options strategies, and risk management. The system leverages machine learning models, alpha factors, and real-time market data to generate trading signals and execute strategies.

## System Architecture

### Core Components

#### Agent Framework
The Alpha-Agent system is built on a robust multi-agent framework where specialized agents collaborate to analyze markets and make trading decisions:

1. **Data Agent** - Fetches and processes market data from Polygon.io API
2. **Prediction Agent** - Forecasts price movements using ML models and alpha factors
3. **Trading Agent** - Evaluates trading opportunities and generates orders
4. **Risk Agent** - Monitors and manages portfolio risk
5. **Sentiment Agent** - Analyzes news and social media for market sentiment

#### Communication System
Agents communicate through a sophisticated message-passing system inspired by metaGPT:

- **Structured Messages** - Standardized format for consistent inter-agent communication
- **Message Routing** - Centralized management of message delivery
- **Pub/Sub Pattern** - Topic-based subscriptions for efficient updates
- **State Management** - Shared memory for critical data

#### Alpha Factors
The system implements quantitative alpha factors derived from academic research and industry practice:

- **101 Alphas** - Implementation of the well-known 101 alpha factors
- **Custom Factors** - Additional proprietary factors for enhanced performance
- **Factor Evaluation** - Dynamic evaluation and selection of highest-performing factors

#### Machine Learning Models
Machine learning models power the prediction capabilities:

- **LSTM Networks** - Deep learning for time series prediction
- **XGBoost** - Gradient boosting for feature-rich prediction
- **Prophet** - Time series decomposition for trend analysis
- **Ensemble Methods** - Combining multiple models for robust predictions

#### Backtesting Engine
Robust backtesting capabilities to evaluate strategies:

- **Historical Data Analysis** - Testing across different market conditions
- **Performance Metrics** - Comprehensive evaluation of strategy performance
- **Risk Assessment** - Analysis of drawdowns and risk-adjusted returns

## Key Features

### Data Integration
- **Real-time Market Data** - Integration with Polygon.io API for live data
- **Historical Data** - Comprehensive database of historical prices for backtesting
- **News and Events** - Incorporation of news sentiment and market events
- **Options Data** - Full options chain data for advanced strategies

### Alpha Selection
- **Dynamic Factor Evaluation** - Continuous assessment of alpha factor performance
- **Correlation Analysis** - Selection of uncorrelated factors for diversification
- **Regime Detection** - Adaptation to changing market conditions
- **Factor Blending** - Optimal combination of factors for enhanced signals

### Prediction Models
- **Multi-timeframe Analysis** - Predictions across different time horizons
- **Confidence Intervals** - Probability distributions for forecasts
- **Anomaly Detection** - Identification of unusual market conditions
- **Feature Importance** - Understanding of key drivers for predictions

### Trading Strategies
- **Stock Selection** - Identification of promising equities
- **Options Strategies** - Advanced options positions based on volatility forecasts
- **Risk Management** - Position sizing and stop-loss mechanisms
- **Dynamic Allocation** - Adaptive capital allocation based on market conditions

### Evaluation Framework
- **Agent Performance Metrics** - Tracking of individual agent effectiveness
- **System-wide Evaluation** - Holistic assessment of the multi-agent system
- **Visualization Tools** - Interactive dashboards for performance analysis

## Getting Started

### Prerequisites
- Python 3.8+
- Polygon.io API key
- Required Python packages (see requirements.txt)

### Installation
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up your configuration in `configs/config.yaml`
4. Set your API keys in environment variables

### Basic Usage
```python
# Start the Alpha-Agent system
from alpha_agent.main import AlphaAgentSystem

system = AlphaAgentSystem(config_path="configs/config.yaml")
system.start()

# Get trading recommendations
recommendations = system.get_recommendations()
print(recommendations)

# Perform backtesting
from alpha_agent.backtest import BacktestEngine

backtester = BacktestEngine(initial_capital=100000)
results = backtester.run_backtest(data="historical_data", strategy=system.strategy)
backtester.plot_portfolio_performance()
```

## Advanced Usage

### Custom Alpha Factors
Create your own alpha factors by extending the AlphaFactors class:

```python
from models.signals.alpha_factors import AlphaFactors


class CustomAlphaFactors(AlphaFactors):
    def my_custom_alpha(self, df):
        # Custom alpha logic here
        return result
```

### Custom Trading Strategies
Implement custom trading strategies:

```python
from agents.trading_agent import BaseTradingStrategy

class MyStrategy(BaseTradingStrategy):
    def generate_signals(self, data, predictions):
        # Strategy logic here
        return signals
```

### Integrating External Data
```python
from utils.data.external_data import ExternalDataSource

data_source = ExternalDataSource(api_key="your_api_key")
system.add_data_source(data_source)
```

## Technical Documentation

### Agent Communication Protocol
The communication protocol defines how agents exchange information:

- **Message Structure**: Each message contains metadata, content, and routing information
- **Message Types**: Various types including DATA, REQUEST, RESPONSE, COMMAND, etc.
- **Subscription Model**: Agents subscribe to relevant topics
- **Error Handling**: Standardized error reporting and recovery

### Alpha Factor Evaluation
Factors are evaluated based on:

- **Information Coefficient (IC)** - Correlation with future returns
- **Turnover** - Trading activity required to capture the signal
- **Decay Rate** - How quickly the signal loses predictive power
- **Stability** - Consistency across different market regimes

### Risk Management Framework
Comprehensive risk controls including:

- **Position Sizing** - Dynamic allocation based on confidence and volatility
- **Correlation Management** - Diversification across uncorrelated bets
- **Stop Loss Mechanisms** - Automated exit rules for adverse movements
- **VaR Calculations** - Value at Risk modeling for portfolio protection

## Performance Evaluation

### Metrics
- **Returns**: Total return, annualized return
- **Risk-adjusted**: Sharpe ratio, Sortino ratio, Calmar ratio
- **Drawdown Analysis**: Maximum drawdown, drawdown duration
- **Factor Performance**: IC decay, factor attribution

### Visualization
- **Equity Curves**: Portfolio value over time
- **Drawdown Charts**: Visualization of downside risk
- **Factor Contribution**: Attribution of performance to factors
- **Agent Metrics**: Performance of individual agents

## Contributing
We welcome contributions to Alpha-Agent! Please see our contributing guidelines for more information on how to get involved.

## License
Alpha-Agent is released under the MIT License. See the LICENSE file for more details.
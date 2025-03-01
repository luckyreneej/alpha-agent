# Alpha-Agent System User Guide - Part 3: Advanced Topics

## Customizing Strategies

### Creating a Custom Strategy

```python
from backtest.strategy import Strategy
import pandas as pd
from typing import Dict

class BollingerBandsStrategy(Strategy):
    """
    Strategy based on Bollinger Bands.
    Buy when price touches lower band, sell when price touches upper band.
    """
    
    def __init__(self, window: int = 20, num_std: float = 2.0):
        """
        Initialize the Bollinger Bands strategy.
        
        Args:
            window: Window size for moving average
            num_std: Number of standard deviations for bands
        """
        super().__init__()
        self.set_parameters(window=window, num_std=num_std)
        self.set_description(f"Bollinger Bands Strategy (window={window}, std={num_std})")
    
    def generate_signals(self, data: pd.DataFrame, current_date: pd.Timestamp, 
                          positions: Dict[str, int]) -> Dict[str, float]:
        """
        Generate trading signals using Bollinger Bands.
        
        Args:
            data: Historical data up to current_date
            current_date: Current date in the backtest
            positions: Current positions {symbol: quantity}
            
        Returns:
            Dictionary mapping symbols to signal values
        """
        signals = {}
        
        # Get unique tickers in the data
        tickers = data['ticker'].unique() if 'ticker' in data.columns else [None]
        
        for ticker in tickers:
            # Filter data for this ticker
            if ticker is not None:
                ticker_data = data[data['ticker'] == ticker]
            else:
                ticker_data = data
            
            if len(ticker_data) < self.parameters['window']:
                # Not enough data for calculation
                signals[ticker or 'default'] = 0
                continue
            
            # Calculate Bollinger Bands
            close_prices = ticker_data['close']
            ma = close_prices.rolling(window=self.parameters['window']).mean()
            std = close_prices.rolling(window=self.parameters['window']).std()
            upper_band = ma + self.parameters['num_std'] * std
            lower_band = ma - self.parameters['num_std'] * std
            
            # Get the latest values
            current_price = close_prices.iloc[-1]
            current_upper = upper_band.iloc[-1]
            current_lower = lower_band.iloc[-1]
            
            # Generate signal
            if current_price <= current_lower:
                # Price at or below lower band - buy signal
                signals[ticker or 'default'] = 1.0
            elif current_price >= current_upper:
                # Price at or above upper band - sell signal
                signals[ticker or 'default'] = -1.0
            else:
                # Price between bands - maintain position
                current_position = positions.get(ticker or 'default', 0)
                signals[ticker or 'default'] = 1.0 if current_position > 0 else (-1.0 if current_position < 0 else 0)
        
        return signals
```

### Creating a Function-Based Strategy

```python
from backtest.strategy import create_strategy_from_function
import pandas as pd
from typing import Dict

def simple_momentum_strategy(data: pd.DataFrame, current_date: pd.Timestamp, 
                           positions: Dict[str, int]) -> Dict[str, float]:
    """
    Simple momentum strategy - buy if price increasing, sell if decreasing.
    
    Args:
        data: Historical data up to current_date
        current_date: Current date in the backtest
        positions: Current positions {symbol: quantity}
        
    Returns:
        Dictionary mapping symbols to signal values
    """
    signals = {}
    
    # Get unique tickers in the data
    tickers = data['ticker'].unique() if 'ticker' in data.columns else [None]
    
    for ticker in tickers:
        # Filter data for this ticker
        if ticker is not None:
            ticker_data = data[data['ticker'] == ticker]
        else:
            ticker_data = data
        
        if len(ticker_data) < 5:  # Need at least 5 periods
            signals[ticker or 'default'] = 0
            continue
        
        # Calculate 5-day return
        returns_5d = ticker_data['close'].pct_change(5).iloc[-1]
        
        # Generate signal based on momentum
        if returns_5d > 0.02:  # 2% threshold for positive momentum
            signals[ticker or 'default'] = 1.0
        elif returns_5d < -0.02:  # -2% threshold for negative momentum
            signals[ticker or 'default'] = -1.0
        else:
            # Neutral momentum
            current_position = positions.get(ticker or 'default', 0)
            signals[ticker or 'default'] = 1.0 if current_position > 0 else (-1.0 if current_position < 0 else 0)
    
    return signals

# Create a strategy from this function
momentum_strategy = create_strategy_from_function(simple_momentum_strategy, name="SimpleMomentum")
```

## Troubleshooting

### Common Issues

#### API Rate Limiting

**Issue**: You receive errors related to API rate limits when fetching data from Polygon.io.

**Solution**: The system has built-in rate limiting handling, but you may need to adjust the delay between API calls:

```python
from utils.data.polygon_api import PolygonAPI
import time

# Initialize API with custom settings
api = PolygonAPI()

# Save original method
api._original_make_request = api._make_request

# Modify the delay between API calls
def delayed_request(*args, **kwargs):
    time.sleep(0.5)  # 500ms delay
    return api._original_make_request(*args, **kwargs)

api._make_request = delayed_request
```

#### Missing Data

**Issue**: Some historical data points are missing or incomplete.

**Solution**: Use the data filling and cleaning utilities:

```python
from utils.data.data_cleaner import fill_missing_data

# Assume stock_data is a dictionary of DataFrames from fetch_stock_history
for ticker, df in stock_data.items():
    # Fill missing values
    stock_data[ticker] = fill_missing_data(
        df, 
        method='ffill',  # Forward fill
        limit=5          # Maximum consecutive missing values to fill
    )
```

#### Memory Issues

**Issue**: Running out of memory when processing large datasets.

**Solution**: Use chunking to process data in smaller batches:

```python
# Define date ranges to fetch data in chunks
from datetime import datetime, timedelta

start_date = datetime.strptime("2018-01-01", "%Y-%m-%d")
end_date = datetime.strptime("2022-12-31", "%Y-%m-%d")
chunk_size = timedelta(days=90)

chunks = []
current_date = start_date

while current_date < end_date:
    chunk_end = min(current_date + chunk_size, end_date)
    
    chunk_data = data_fetcher.fetch_stock_history(
        tickers=["AAPL", "MSFT", "GOOGL"],
        start_date=current_date.strftime("%Y-%m-%d"),
        end_date=chunk_end.strftime("%Y-%m-%d")
    )
    
    # Process the chunk
    chunks.append(chunk_data)
    current_date = chunk_end + timedelta(days=1)
```

### FAQ

#### How many tickers can I backtest simultaneously?

The system can handle dozens of tickers simultaneously, but performance depends on your hardware. For large scale backtests (100+ tickers), consider using chunking and parallel processing.

#### How do I add technical indicators that aren't built into the system?

You can add custom indicators by creating helper functions and using them in your strategies:

```python
def calculate_keltner_channels(data, ema_period=20, atr_period=10, multiplier=2.0):
    """Calculate Keltner Channels."""
    from ta.volatility import AverageTrueRange
    
    # Calculate EMA of close prices
    ema = data['close'].ewm(span=ema_period, adjust=False).mean()
    
    # Calculate ATR
    atr = AverageTrueRange(data['high'], data['low'], data['close'], atr_period).average_true_range()
    
    # Calculate upper and lower bands
    upper_band = ema + (multiplier * atr)
    lower_band = ema - (multiplier * atr)
    
    return ema, upper_band, lower_band
```

#### How do I save and load backtest results?

Use Python's pickle module or pandas' built-in saving functions:

```python
# Save backtest results
import pickle

with open('results/backtest_results.pkl', 'wb') as f:
    pickle.dump(results, f)

# Save equity curve as CSV
results['equity_curve'].to_csv('results/equity_curve.csv')

# Load backtest results later
with open('results/backtest_results.pkl', 'rb') as f:
    loaded_results = pickle.load(f)
```

### Performance Optimization

#### Using Vectorized Operations

For best performance, use vectorized operations instead of loops:

```python
# Slow approach with loops
for i in range(len(data)):
    data.loc[i, 'new_column'] = data.loc[i, 'close'] * 2

# Fast vectorized approach
data['new_column'] = data['close'] * 2
```

#### Parallelize Backtests

For testing multiple strategies or parameter combinations:

```python
from concurrent.futures import ProcessPoolExecutor

# Define a function to run a single backtest
def run_single_backtest(strategy_params):
    strategy = MovingAverageCrossStrategy(
        fast_period=strategy_params['fast'], 
        slow_period=strategy_params['slow']
    )
    
    backtest = BacktestEngine(
        initial_capital=100000,
        data_fetcher=data_fetcher,
        strategy=strategy
    )
    
    return backtest.run(
        tickers=["SPY"],
        start_date="2018-01-01",
        end_date="2022-12-31"
    )

# Define parameter combinations to test
params_to_test = [
    {'fast': 10, 'slow': 50},
    {'fast': 20, 'slow': 100},
    {'fast': 50, 'slow': 200}
]

# Run backtests in parallel
with ProcessPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(run_single_backtest, params_to_test))
```

## Extension Guide

### Adding Custom Alpha Factors

Alpha factors represent signals that may predict future returns. To add a custom alpha factor:

1. Create a new module in the `utils/alpha_factors` directory:

```python
# utils/alpha_factors/momentum_factors.py
import pandas as pd

def calculate_momentum(data: pd.DataFrame, period: int = 20) -> pd.Series:
    """Calculate price momentum over a specific period."""
    return data['close'].pct_change(period)

def calculate_pmo(data: pd.DataFrame, short_period: int = 35, long_period: int = 20, 
                signal_period: int = 10) -> pd.DataFrame:
    """Calculate Price Momentum Oscillator (PMO)."""
    # Calculate Rate of Change (ROC)
    roc = data['close'].pct_change() * 100
    
    # Calculate EMAs
    short_ema = roc.ewm(span=short_period, adjust=False).mean()
    long_ema = short_ema.ewm(span=long_period, adjust=False).mean() * 10
    
    # Calculate PMO and signal line
    pmo = long_ema
    pmo_signal = pmo.ewm(span=signal_period, adjust=False).mean()
    
    result = pd.DataFrame({
        'pmo': pmo,
        'pmo_signal': pmo_signal
    })
    
    return result
```

2. Use the alpha factor in a strategy:

```python
from backtest.strategy import Strategy
from utils.alpha_factors.momentum_factors import calculate_pmo

class PMOStrategy(Strategy):
    def __init__(self, short_period: int = 35, long_period: int = 20, signal_period: int = 10):
        super().__init__()
        self.set_parameters(
            short_period=short_period,
            long_period=long_period,
            signal_period=signal_period
        )
        self.set_description(f"PMO Strategy ({short_period}/{long_period}/{signal_period})")
    
    def generate_signals(self, data, current_date, positions):
        # Calculate PMO
        pmo_data = calculate_pmo(
            data, 
            self.parameters['short_period'],
            self.parameters['long_period'],
            self.parameters['signal_period']
        )
        
        # Generate signals
        signals = {}
        tickers = data['ticker'].unique() if 'ticker' in data.columns else [None]
        
        for ticker in tickers:
            # Generate buy/sell signals based on PMO crossovers
            if ticker is not None:
                ticker_data = data[data['ticker'] == ticker]
                ticker_pmo = pmo_data.loc[ticker_data.index]
            else:
                ticker_pmo = pmo_data
            
            if len(ticker_pmo) < 2:
                signals[ticker or 'default'] = 0
                continue
            
            # Generate signals based on PMO and signal line crossovers
            latest_pmo = ticker_pmo['pmo'].iloc[-1]
            latest_signal = ticker_pmo['pmo_signal'].iloc[-1]
            prev_pmo = ticker_pmo['pmo'].iloc[-2]
            prev_signal = ticker_pmo['pmo_signal'].iloc[-2]
            
            if prev_pmo <= prev_signal and latest_pmo > latest_signal:
                signals[ticker or 'default'] = 1.0  # Bullish crossover
            elif prev_pmo >= prev_signal and latest_pmo < latest_signal:
                signals[ticker or 'default'] = -1.0  # Bearish crossover
            else:
                current_position = positions.get(ticker or 'default', 0)
                signals[ticker or 'default'] = 1.0 if current_position > 0 else \
                                              (-1.0 if current_position < 0 else 0)
        
        return signals
```

### Adding New Agents

To extend the Alpha-Agent system with custom agents, follow these steps:

1. Define a new agent class in the `agents` directory:

```python
# agents/sentiment_agent.py
from agents.base_agent import BaseAgent
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

class SentimentAgent(BaseAgent):
    """Agent for analyzing sentiment from news and social media."""
    
    def __init__(self, agent_id: str, communicator=None):
        super().__init__(agent_id, communicator)
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Register request handlers
        self.register_request_handler("analyze_news", self._handle_news_analysis)
        self.register_request_handler("analyze_social", self._handle_social_analysis)
    
    def _handle_news_analysis(self, message):
        """Handle news sentiment analysis request."""
        content = message.content
        news_items = content.get("news_items", [])
        
        # Analyze sentiment
        results = []
        for item in news_items:
            sentiment_scores = self.sentiment_analyzer.polarity_scores(item["text"])
            results.append({
                "id": item.get("id"),
                "title": item.get("title"),
                "sentiment": sentiment_scores,
                "compound_score": sentiment_scores["compound"],
                "sentiment_class": "positive" if sentiment_scores["compound"] > 0.05 else \
                                  ("negative" if sentiment_scores["compound"] < -0.05 else "neutral")
            })
        
        return results
        
    def _handle_social_analysis(self, message):
        """Handle social media sentiment analysis request."""
        content = message.content
        social_posts = content.get("posts", [])
        
        # Similar to news analysis but with social media specific processing
        # ...
        
        return {"result": "analysis complete"}
```

2. Initialize and use the agent:

```python
from utils.communication.unified_communication import UnifiedCommunicationManager
from agents.sentiment_agent import SentimentAgent

# Initialize communication system
communication_manager = UnifiedCommunicationManager()
communication_manager.start()

# Create sentiment agent
sentiment_agent = SentimentAgent("sentiment_agent_1", communication_manager)
sentiment_agent.start()

# Use the agent (from another agent)
from agents.base_agent import BaseAgent

trading_agent = BaseAgent("trading_agent_1", communication_manager)

# Send a request to the sentiment agent
response = trading_agent.send_request(
    receiver_id="sentiment_agent_1",
    request_type="analyze_news",
    content={
        "news_items": [
            {"id": "1", "title": "Positive earnings report", "text": "The company reported strong earnings, exceeding expectations."},
            {"id": "2", "title": "Market downturn", "text": "Markets plunged today amid recession fears and inflation concerns."}
        ]
    }
)

print("Sentiment analysis results:")
for item in response:
    print(f"News ID: {item['id']}")
    print(f"Title: {item['title']}")
    print(f"Sentiment: {item['sentiment_class']} (score: {item['compound_score']:.2f})")
    print()
```

### Extending Functionality

To add new features or extend existing ones, you can create extension modules in appropriate directories. For example, to add a new risk management module:

```python
# utils/risk/risk_manager.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

class RiskManager:
    """Risk management module for controlling position sizing and exposure."""
    
    def __init__(self, max_portfolio_risk: float = 0.01, max_position_risk: float = 0.005):
        """
        Initialize risk manager.
        
        Args:
            max_portfolio_risk: Maximum allowed portfolio risk (daily VaR as fraction)
            max_position_risk: Maximum allowed position risk (daily VaR as fraction)
        """
        self.max_portfolio_risk = max_portfolio_risk
        self.max_position_risk = max_position_risk
    
    def calculate_position_size(self, ticker: str, price: float, volatility: float, 
                              portfolio_value: float) -> int:
        """
        Calculate optimal position size based on risk constraints.
        
        Args:
            ticker: Ticker symbol
            price: Current price
            volatility: Daily volatility of the asset
            portfolio_value: Total portfolio value
            
        Returns:
            Number of shares to trade
        """
        # Calculate max dollar risk per position
        max_dollar_risk = portfolio_value * self.max_position_risk
        
        # Calculate position size based on volatility
        # Assuming 2 standard deviations for 95% confidence
        risk_per_share = price * volatility * 2.0
        
        # Maximum number of shares based on risk
        max_shares = max_dollar_risk / risk_per_share
        
        # Round down to integer
        return int(max_shares)
    
    def calculate_kelly_position_size(self, win_rate: float, win_loss_ratio: float, 
                                    portfolio_value: float, max_kelly_fraction: float = 0.5) -> float:
        """
        Calculate position size using the Kelly Criterion.
        
        Args:
            win_rate: Probability of winning (0.0 to 1.0)
            win_loss_ratio: Ratio of average win to average loss
            portfolio_value: Total portfolio value
            max_kelly_fraction: Maximum fraction of Kelly to use (usually 0.5 for half-Kelly)
            
        Returns:
            Dollar amount to allocate
        """
        # Kelly formula: f* = (p*b - q)/b where:
        # f* = fraction of bankroll to bet
        # p = probability of winning
        # q = probability of losing (1-p)
        # b = win/loss ratio
        
        q = 1 - win_rate
        kelly_fraction = (win_rate * win_loss_ratio - q) / win_loss_ratio
        
        # Limit to maximum fraction and ensure non-negative
        kelly_fraction = max(0, min(kelly_fraction, max_kelly_fraction))
        
        # Calculate dollar amount
        return portfolio_value * kelly_fraction
```

Then use this in your trading strategy or backtest engine:

```python
from utils.risk.risk_manager import RiskManager

# Initialize risk manager
risk_manager = RiskManager(max_portfolio_risk=0.01, max_position_risk=0.005)

# Calculate position size
volatility = 0.015  # 1.5% daily volatility
price = 150.0
portfolio_value = 100000.0

shares_to_buy = risk_manager.calculate_position_size(
    ticker="AAPL", 
    price=price, 
    volatility=volatility,
    portfolio_value=portfolio_value
)

print(f"Risk-adjusted position size: {shares_to_buy} shares (${shares_to_buy * price:.2f})")
```

These examples show how the Alpha-Agent system can be extended with new components and functionality to suit your specific needs.
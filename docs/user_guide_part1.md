# Alpha-Agent System User Guide - Part 1: Getting Started

## Table of Contents

1. [Getting Started](#getting-started)
   - [Installation](#installation)
   - [API Key Setup](#api-key-setup)
   - [Configuration](#configuration)
   - [Environment Setup](#environment-setup)

2. [System Architecture Overview](#system-architecture-overview)
   - [Multi-Agent Structure](#multi-agent-structure)
   - [Data Flow](#data-flow)
   - [Agent Roles and Responsibilities](#agent-roles-and-responsibilities)

## Getting Started

### Installation

To install the Alpha-Agent system, follow these steps:

1. Clone the repository:
```bash
git clone https://github.com/your-repo/alpha-agent.git
cd alpha-agent
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

The main dependencies include:
- pandas
- numpy
- matplotlib
- seaborn
- scipy
- requests
- ta (Technical Analysis library)

### API Key Setup

The system requires a Polygon.io API key for historical data retrieval. You can obtain a free API key by registering at [Polygon.io](https://polygon.io/).

Once you have your API key, set it up as an environment variable:

```bash
export POLYGON_API_KEY="your_api_key_here"
```

Alternatively, you can provide the API key directly when initializing the `PolygonAPI` class:

```python
from utils.data.polygon_api import PolygonAPI

api = PolygonAPI(api_key="your_api_key_here")
```

### Configuration

The system uses a configuration file to manage settings. You can find the default configuration in `config/default_config.json`. To create a custom configuration:

1. Copy the default configuration:
```bash
cp config/default_config.json config/my_config.json
```

2. Edit the configuration file with your preferred settings:
```json
{
  "api": {
    "polygon_api_key": "YOUR_API_KEY_HERE"
  },
  "data": {
    "cache_dir": "data/cache",
    "default_timeframe": "day"
  },
  "backtest": {
    "default_capital": 100000,
    "transaction_cost": 0.001
  },
  "logging": {
    "level": "INFO",
    "file": "logs/alpha_agent.log"
  }
}
```

3. Load your custom configuration:
```python
from utils.config import load_config

config = load_config("config/my_config.json")
```

### Environment Setup

For optimal performance, we recommend setting up a dedicated environment:

1. Create a virtual environment:
```bash
python -m venv alpha-agent-env
```

2. Activate the environment:
   - On Windows: `alpha-agent-env\Scripts\activate`
   - On macOS/Linux: `source alpha-agent-env/bin/activate`

3. Install dependencies within the environment:
```bash
pip install -r requirements.txt
```

## System Architecture Overview

### Multi-Agent Structure

The Alpha-Agent system uses a multi-agent architecture where specialized agents collaborate to perform different tasks in the trading and analysis pipeline:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│   Data Agent    │────▶│  Analysis Agent │────▶│  Trading Agent  │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        ▲                        │                      │
        │                        │                      │
        │                        ▼                      ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  External Data  │     │ Backtesting    │◀────│  Portfolio      │
│  Sources        │     │ Engine         │     │  Optimizer      │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

### Data Flow

1. **Data Collection**: The Data Agent retrieves historical and real-time market data from external sources (primarily Polygon.io).

2. **Data Processing**: Raw data is processed, normalized, and cached for efficient reuse.

3. **Analysis**: The Analysis Agent applies various technical indicators, statistical methods, and machine learning algorithms to generate insights.

4. **Signal Generation**: Based on analysis, trading signals are generated.

5. **Portfolio Optimization**: The Trading Agent optimizes portfolio allocation based on signals and risk parameters.

6. **Backtesting**: Strategies are evaluated on historical data to measure performance.

7. **Execution**: In a live environment, trades would be executed based on signals and optimization.

### Agent Roles and Responsibilities

#### Data Agent
- Retrieves historical and real-time market data
- Handles data caching and management
- Normalizes data formats across different sources
- Manages API rate limits and connection issues

#### Analysis Agent
- Applies technical indicators to price data
- Generates statistical insights and correlations
- Identifies patterns and potential trading opportunities
- Evaluates market conditions and regime detection

#### Trading Agent
- Converts analysis into actionable trading signals
- Manages portfolio allocation and risk
- Determines position sizing and entry/exit points
- Tracks performance metrics in real-time
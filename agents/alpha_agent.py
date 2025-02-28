import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Any, Optional, Union

from agents.base_agent import BaseAgent
from utils.communication.message import Message
from models.alpha_selection.selection import AlphaSelector


class AlphaAgent(BaseAgent):
    """Enhanced Alpha Agent that integrates both signal generation and trade execution.

    This agent combines alpha factor signal generation with trade execution capabilities.
    It evaluates alpha factors to generate trading signals and can execute trades based
    on those signals.

    Attributes:
        alpha_selector (AlphaSelector): Component for selecting and evaluating alpha factors
        positions (dict): Current portfolio positions
        trades_history (list): History of executed trades
        performance_metrics (dict): Trading performance tracking metrics
    """

    def __init__(self, config=None):
        """Initialize the AlphaAgent.

        Args:
            config (dict, optional): Configuration settings
        """
        super().__init__("AlphaAgent")

        # Default configuration
        self.config = config or {
            'alpha_weights': {},
            'position_sizing': {
                'max_position_pct': 0.05,  # Max 5% of portfolio in a single position
                'max_sector_exposure': 0.25,  # Max 25% in a single sector
            },
            'risk_limits': {
                'max_drawdown_pct': 0.15,  # Exit if 15% drawdown reached
                'stop_loss_pct': 0.08,  # 8% stop loss
                'take_profit_pct': 0.20,  # 20% take profit
            },
            'execution': {
                'market_impact_model': 'linear',  # Linear market impact
                'slippage_model': 'proportional',  # Proportional slippage
                'default_order_type': 'limit',  # Default to limit orders
            }
        }

        # Initialize alpha selector
        self.alpha_selector = AlphaSelector()
        self.alpha_selector.register_all_builtin_alphas()

        # Initialize portfolio tracking
        self.positions = {}  # Current positions {ticker: {quantity, average_price, cost_basis}}
        self.trades_history = []  # List of executed trades
        self.performance_metrics = {  # Performance tracking
            'total_return': 0.0,
            'win_count': 0,
            'loss_count': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'max_drawdown': 0.0
        }

        # Initialize logger
        self.logger = logging.getLogger(self.__class__.__name__)

    async def process_message(self, message):
        """Process incoming messages and respond accordingly.

        Args:
            message (Message): Incoming message with action request

        Returns:
            Message: Response message
        """
        action = message.content.get('action')
        self.logger.info(f"Processing action: {action}")

        if action == 'generate_signal':
            # Generate trading signal based on alpha factors
            ticker = message.content.get('ticker')
            data = message.content.get('data')

            signal = await self.generate_trading_signal(ticker, data)

            return Message(
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                content={
                    'signal': signal,
                    'ticker': ticker,
                    'timestamp': datetime.now().isoformat()
                },
                correlation_id=message.id
            )

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Any, Optional, Union

from agents.base_agent import BaseAgent
from utils.communication.message import Message
from utils.data.polygon_api import PolygonAPI


class RiskAgent(BaseAgent):
    """Enhanced Risk Management Agent for market and position risk assessment.

    Attributes:
        data_fetcher (PolygonAPI): API client for market data
        market_indices (dict): Key market indices to track
        risk_state (dict): Current risk assessment state
    """

    def __init__(self, polygon_api_key=None):
        """Initialize the Risk Agent with required components."""
        super().__init__("RiskAgent")

        # Initialize data fetcher
        self.data_fetcher = PolygonAPI(api_key=polygon_api_key)

        # Key market indices to track
        self.market_indices = {
            'SPY': 'S&P 500',  # Overall market
            'VIX': 'Volatility Index',  # Market volatility
            'TLT': 'Treasury Bond ETF',  # Bond market
            'DXY': 'US Dollar Index',  # Currency strength
            'GLD': 'Gold ETF'  # Safe haven asset
        }

        # Initialize risk state tracking
        self.risk_state = {
            'overall_risk_score': 0.5,  # Default medium risk
            'last_updated': datetime.now().isoformat()
        }

        # Initialize logger
        self.logger = logging.getLogger(self.__class__.__name__)

    async def process_message(self, message):
        """Process incoming messages and respond with risk assessments."""
        self.logger.info(f"Processing message: {message.content.get('action', 'unknown')}")

        if message.content.get('action') == 'assess_market_risk':
            # Assess overall market risk
            risk_assessment = await self.assess_market_risk()

            # Return assessment
            return Message(
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                content={
                    'market_risk_assessment': risk_assessment,
                    'timestamp': datetime.now().isoformat()
                },
                correlation_id=message.id
            )

        elif message.content.get('action') == 'assess_position_risk':
            # Get ticker and position details
            ticker = message.content.get('ticker')
            position_size = message.content.get('position_size')
            price = message.content.get('price')

            # Assess position specific risk
            risk_assessment = await self.assess_position_risk(
                ticker=ticker,
                position_size=position_size,
                price=price,
                portfolio_value=message.content.get('portfolio_value')
            )

            # Return assessment
            return Message(
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                content={
                    'position_risk_assessment': risk_assessment,
                    'ticker': ticker,
                    'timestamp': datetime.now().isoformat()
                },
                correlation_id=message.id
            )

        elif message.content.get('action') == 'assess_options_risk':
            # Get option details
            ticker = message.content.get('ticker')
            option_type = message.content.get('option_type')  # 'call' or 'put'
            strike_price = message.content.get('strike_price')
            expiration_date = message.content.get('expiration_date')
            position_size = message.content.get('position_size')

            # Assess options specific risk
            risk_assessment = await self.assess_options_risk(
                ticker=ticker,
                option_type=option_type,
                strike_price=strike_price,
                expiration_date=expiration_date,
                position_size=position_size
            )

            # Return assessment
            return Message(
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                content={
                    'options_risk_assessment': risk_assessment,
                    'ticker': ticker,
                    'option_type': option_type,
                    'strike_price': strike_price,
                    'expiration_date': expiration_date,
                    'timestamp': datetime.now().isoformat()
                },
                correlation_id=message.id
            )
        else:
            # Unsupported action
            return Message(
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                content={
                    'error': 'Unsupported action',
                    'supported_actions': ['assess_market_risk', 'assess_position_risk', 'assess_options_risk']
                },
                correlation_id=message.id
            )

    def _calculate_drawdown(self, price_series):
        """Calculate maximum drawdown for a price series."""
        # Calculate the running maximum
        rolling_max = price_series.cummax()
        # Calculate the drawdown
        drawdown = (price_series - rolling_max) / rolling_max
        # Return the minimum (maximum drawdown)
        return drawdown.min()

    def _calculate_rsi(self, price_series, window=14):
        """Calculate the Relative Strength Index (RSI)."""
        # Calculate price changes
        delta = price_series.diff()

        # Separate gains and losses
        gains = delta.copy()
        losses = delta.copy()
        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = abs(losses)

        # Calculate average gains and losses
        avg_gain = gains.rolling(window=window).mean()
        avg_loss = losses.rolling(window=window).mean()

        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi.iloc[-1]

    def _generate_risk_recommendations(self, risk_score, market_regime):
        """Generate market risk recommendations based on risk score and regime."""
        recommendations = []

        if risk_score > 0.7:
            recommendations.append("Consider reducing overall market exposure.")
            recommendations.append("Implement protective hedges such as put options or increased cash reserves.")
        elif risk_score > 0.5:
            recommendations.append("Maintain a balanced portfolio with defensive positions.")
            recommendations.append("Consider tactical hedging for key positions.")
        else:
            recommendations.append("Maintain normal market exposure based on investment strategy.")

        if market_regime == "High Volatility":
            recommendations.append("Reduce position sizes to account for increased volatility.")

        return recommendations

    def _generate_position_recommendations(self, ticker, risk_level, metrics, risk_components):
        """Generate position-specific recommendations."""
        recommendations = []

        # Basic recommendation based on risk level
        if risk_level in ["High", "Very High"]:
            recommendations.append(f"Consider reducing position size in {ticker}.")

        # Liquidity recommendations
        if risk_components.get('liquidity_risk', 0) > 0.7:
            recommendations.append("Position may be difficult to exit quickly. Consider staged reduction if needed.")

        # Concentration recommendations
        if risk_components.get('concentration_risk', 0) > 0.7:
            recommendations.append("Position represents significant portfolio concentration. Consider diversification.")

        # Volatility recommendations
        if risk_components.get('volatility_risk', 0) > 0.7:
            recommendations.append("High volatility detected. Consider implementing stop-loss orders.")

        return recommendations

    async def assess_market_risk(self):
        """Perform market risk assessment."""
        try:
            self.logger.info("Assessing overall market risk")

            # Get current date and lookback periods
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)

            # Format dates for API
            end_date_str = end_date.strftime("%Y-%m-%d")
            start_date_str = start_date.strftime("%Y-%m-%d")

            # Get market index data
            market_data = {}
            raw_data = {}
            for ticker in self.market_indices.keys():
                try:
                    data = self.data_fetcher.get_stock_bars(
                        ticker=ticker,
                        from_date=start_date_str,
                        to_date=end_date_str,
                        timespan="day"
                    )

                    if not data.empty:
                        # Store raw data for correlation analysis
                        raw_data[ticker] = data

                        # Calculate key metrics
                        market_data[ticker] = {
                            'current_price': data['close'].iloc[-1],
                            'monthly_return': data['close'].pct_change(20).iloc[-1],
                            'volatility': data['close'].pct_change().std() * np.sqrt(252),
                            'maximum_drawdown': self._calculate_drawdown(data['close']),
                            'rsi': self._calculate_rsi(data['close'])
                        }
                except Exception as e:
                    self.logger.error(f"Error getting data for {ticker}: {e}")
                    market_data[ticker] = {'error': str(e)}

            # Calculate correlation with SPY
            correlation_matrix = {}
            if 'SPY' in raw_data:
                for ticker, data in raw_data.items():
                    if ticker != 'SPY':
                        combined = pd.merge(
                            raw_data['SPY']['close'],
                            data['close'],
                            left_index=True,
                            right_index=True,
                            suffixes=('_spy', f"_{ticker}")
                        )

                        if not combined.empty and combined.shape[0] > 1:
                            correlation = combined.corr().iloc[0, 1]
                            correlation_matrix[ticker] = correlation

            # Assess market regime
            market_regime = "Unknown"
            if 'SPY' in market_data and 'VIX' in market_data:
                spy_data = market_data['SPY']
                vix_data = market_data['VIX']

                if spy_data.get('rsi', 50) > 70 and vix_data.get('current_price', 20) < 15:
                    market_regime = "Bullish"
                elif spy_data.get('rsi', 50) < 30 and vix_data.get('current_price', 20) > 25:
                    market_regime = "Bearish"
                elif vix_data.get('current_price', 20) > 30:
                    market_regime = "High Volatility"
                else:
                    market_regime = "Neutral"

            # Calculate risk components
            vix_value = market_data.get('VIX', {}).get('current_price', 20)
            spy_vol = market_data.get('SPY', {}).get('volatility', 0.15)
            spy_drawdown = market_data.get('SPY', {}).get('maximum_drawdown', -0.1)

            # Risk components
            volatility_risk = min(1.0, (vix_value / 40.0) * 0.7 + (spy_vol / 0.4) * 0.3)

            # Overall risk score
            overall_risk_score = volatility_risk

            # Risk level
            risk_level = "Medium"
            if overall_risk_score < 0.3:
                risk_level = "Low"
            elif overall_risk_score > 0.7:
                risk_level = "High"

            # Store risk state
            self.risk_state = {
                'overall_risk_score': float(overall_risk_score),
                'risk_level': risk_level,
                'last_updated': datetime.now().isoformat()
            }

            # Create assessment
            risk_assessment = {
                'overall_risk': {
                    'score': float(overall_risk_score),
                    'level': risk_level
                },
                'market_regime': market_regime,
                'vix_value': float(vix_value),
                'spy_volatility': float(spy_vol),
                'recommendations': self._generate_risk_recommendations(overall_risk_score, market_regime),
                'timestamp': datetime.now().isoformat()
            }

            return risk_assessment

        except Exception as e:
            self.logger.exception(f"Error assessing market risk: {e}")
            return {
                'error': str(e),
                'status': 'assessment_failed',
                'fallback_risk_level': 'Medium',
                'timestamp': datetime.now().isoformat()
            }

    async def assess_position_risk(self, ticker, position_size, price=None, portfolio_value=None):
        """Assess risk for a specific stock position."""
        try:
            self.logger.info(f"Assessing position risk for {ticker}")

            # Get data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=90)

            end_date_str = end_date.strftime("%Y-%m-%d")
            start_date_str = start_date.strftime("%Y-%m-%d")

            # Get stock and market data
            stock_data = self.data_fetcher.get_stock_bars(
                ticker=ticker,
                from_date=start_date_str,
                to_date=end_date_str,
                timespan="day"
            )

            market_data = self.data_fetcher.get_stock_bars(
                ticker="SPY",
                from_date=start_date_str,
                to_date=end_date_str,
                timespan="day"
            )

            # Position metrics
            if price is None and not stock_data.empty:
                price = stock_data['close'].iloc[-1]

            position_value = price * position_size if price is not None else 0

            # Calculate stock specific metrics
            metrics = {}
            if not stock_data.empty:
                stock_returns = stock_data['close'].pct_change().dropna()

                # Key metrics
                metrics['volatility'] = stock_returns.std() * np.sqrt(252)
                metrics['max_drawdown'] = self._calculate_drawdown(stock_data['close'])
                metrics['avg_daily_volume'] = stock_data['volume'].mean()
                metrics['days_to_liquidate'] = position_size / (metrics['avg_daily_volume'] * 0.1) \
                    if metrics['avg_daily_volume'] > 0 else float('inf')

                # Calculate beta if market data available
                if not market_data.empty:
                    combined = pd.merge(
                        stock_returns,
                        market_data['close'].pct_change(),
                        left_index=True,
                        right_index=True,
                        suffixes=('_stock', '_market')
                    ).dropna()

                    if not combined.empty and combined.shape[0] > 1:
                        covariance = combined.cov().iloc[0, 1]
                        market_variance = combined.iloc[:, 1].var()
                        metrics['beta'] = covariance / market_variance if market_variance > 0 else 1.0
                        metrics['correlation_to_market'] = combined.corr().iloc[0, 1]

            # Risk components
            concentration_risk = 0.5
            if portfolio_value and portfolio_value > 0:
                concentration = position_value / portfolio_value
                if concentration > 0.2:
                    concentration_risk = 0.9
                elif concentration > 0.1:
                    concentration_risk = 0.7
                elif concentration > 0.05:
                    concentration_risk = 0.5
                else:
                    concentration_risk = 0.3

            liquidity_risk = 0.5
            if 'days_to_liquidate' in metrics:
                if metrics['days_to_liquidate'] > 10:
                    liquidity_risk = 0.9
                elif metrics['days_to_liquidate'] > 3:
                    liquidity_risk = 0.7
                elif metrics['days_to_liquidate'] > 1:
                    liquidity_risk = 0.5
                else:
                    liquidity_risk = 0.2

            volatility_risk = 0.5
            if 'volatility' in metrics:
                if metrics['volatility'] > 0.5:
                    volatility_risk = 0.9
                elif metrics['volatility'] > 0.3:
                    volatility_risk = 0.7
                elif metrics['volatility'] > 0.2:
                    volatility_risk = 0.5
                else:
                    volatility_risk = 0.3

            risk_components = {
                'volatility_risk': volatility_risk,
                'liquidity_risk': liquidity_risk,
                'concentration_risk': concentration_risk
            }

            # Calculate overall risk score
            overall_risk_score = (volatility_risk * 0.4 + liquidity_risk * 0.3 + concentration_risk * 0.3)

            # Risk level
            risk_level = "Medium"
            if overall_risk_score < 0.3:
                risk_level = "Low"
            elif overall_risk_score > 0.7:
                risk_level = "High"

            # Generate recommendations
            recommendations = self._generate_position_recommendations(
                ticker=ticker,
                risk_level=risk_level,
                metrics=metrics,
                risk_components=risk_components
            )

            # Create assessment
            position_risk_assessment = {
                'ticker': ticker,
                'position_size': position_size,
                'position_value': position_value,
                'current_price': price,
                'overall_risk': {
                    'score': float(overall_risk_score),
                    'level': risk_level
                },
                'risk_components': {k: float(v) for k, v in risk_components.items()},
                'metrics': {k: float(v) if isinstance(v, (int, float)) else v for k, v in metrics.items()},
                'recommendations': recommendations,
                'timestamp': datetime.now().isoformat()
            }

            return position_risk_assessment

        except Exception as e:
            self.logger.exception(f"Error assessing position risk for {ticker}: {e}")
            return {
                'error': str(e),
                'status': 'assessment_failed',
                'ticker': ticker,
                'timestamp': datetime.now().isoformat()
            }

    async def assess_options_risk(self, ticker, option_type, strike_price, expiration_date, position_size):
        """Assess risk for options positions."""
        try:
            self.logger.info(
                f"Assessing options risk for {ticker} {option_type} at ${strike_price} expiring {expiration_date}")

            # Get current date
            current_date = datetime.now()

            # Calculate days to expiration
            expiry_date = datetime.strptime(expiration_date, "%Y-%m-%d")
            days_to_expiry = max(0, (expiry_date - current_date).days)

            # Get underlying stock data
            end_date_str = current_date.strftime("%Y-%m-%d")
            start_date_str = (current_date - timedelta(days=30)).strftime("%Y-%m-%d")

            stock_data = self.data_fetcher.get_stock_bars(
                ticker=ticker,
                from_date=start_date_str,
                to_date=end_date_str,
                timespan="day"
            )

            # Get option contract data
            option_contract = None
            try:
                contracts = self.data_fetcher.get_options_contracts(
                    underlying_ticker=ticker,
                    expiration_date=expiration_date,
                    strike_price=strike_price,
                    contract_type=option_type.lower()
                )

                if not contracts.empty:
                    option_contract = contracts.iloc[0].to_dict()
            except Exception as e:
                self.logger.error(f"Error fetching option contract: {e}")

            # Basic risk calculation
            if stock_data.empty:
                return {
                    'error': 'Unable to retrieve stock data',
                    'status': 'assessment_failed',
                    'timestamp': datetime.now().isoformat()
                }

            # Current stock price
            current_stock_price = stock_data['close'].iloc[-1]

            # Time decay risk increases as expiration approaches
            time_decay_risk = 0.5
            if days_to_expiry < 7:
                time_decay_risk = 0.9
            elif days_to_expiry < 30:
                time_decay_risk = 0.7
            elif days_to_expiry < 60:
                time_decay_risk = 0.5
            else:
                time_decay_risk = 0.3

            # Strike distance risk (how far from current price)
            moneyness = current_stock_price / strike_price
            if option_type.lower() == 'call':
                # For calls, risk increases as stock price gets further below strike
                if moneyness < 0.9:  # Deep out of the money
                    strike_distance_risk = 0.8
                elif moneyness < 1.0:  # Out of the money
                    strike_distance_risk = 0.6
                else:  # In the money
                    strike_distance_risk = 0.4
            else:  # Put option
                # For puts, risk increases as stock price gets further above strike
                if moneyness > 1.1:  # Deep out of the money
                    strike_distance_risk = 0.8
                elif moneyness > 1.0:  # Out of the money
                    strike_distance_risk = 0.6
                else:  # In the money
                    strike_distance_risk = 0.4

            # Volatility risk based on underlying stock
            stock_volatility = stock_data['close'].pct_change().std() * np.sqrt(252)

            volatility_risk = 0.5
            if stock_volatility > 0.5:  # Extremely volatile
                volatility_risk = 0.9
            elif stock_volatility > 0.3:  # Highly volatile
                volatility_risk = 0.7
            elif stock_volatility > 0.2:  # Moderately volatile
                volatility_risk = 0.5
            else:  # Low volatility
                volatility_risk = 0.3

            # Overall option risk (combine components)
            overall_risk_score = (time_decay_risk * 0.4 +
                                  strike_distance_risk * 0.3 +
                                  volatility_risk * 0.3)

            # Risk level
            risk_level = "Medium"
            if overall_risk_score < 0.4:
                risk_level = "Low"
            elif overall_risk_score > 0.7:
                risk_level = "High"

            # Generate recommendations
            recommendations = []

            if days_to_expiry < 7:
                recommendations.append(
                    "Option is near expiration. Consider closing position to avoid gamma risk and weekend risk.")

            if option_type.lower() == 'call' and moneyness < 0.9:
                recommendations.append(
                    "Call option is deeply out of the money. Consider rolling to a different strike or expiration.")

            if option_type.lower() == 'put' and moneyness > 1.1:
                recommendations.append(
                    "Put option is deeply out of the money. Consider rolling to a different strike or expiration.")

            if volatility_risk > 0.7:
                recommendations.append("Underlying stock has high volatility. Consider reducing position size.")

            # Create assessment
            options_risk_assessment = {
                'ticker': ticker,
                'option_type': option_type,
                'strike_price': strike_price,
                'expiration_date': expiration_date,
                'days_to_expiry': days_to_expiry,
                'current_stock_price': current_stock_price,
                'overall_risk': {
                    'score': float(overall_risk_score),
                    'level': risk_level
                },
                'risk_components': {
                    'time_decay_risk': float(time_decay_risk),
                    'strike_distance_risk': float(strike_distance_risk),
                    'volatility_risk': float(volatility_risk)
                },
                'contract_data': option_contract,
                'recommendations': recommendations,
                'timestamp': datetime.now().isoformat()
            }

            return options_risk_assessment

        except Exception as e:
            self.logger.exception(f"Error assessing options risk: {e}")
            return {
                'error': str(e),
                'status': 'assessment_failed',
                'ticker': ticker,
                'timestamp': datetime.now().isoformat()
            }

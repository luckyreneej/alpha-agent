import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Any, Optional, Union

from utils.communication.base_agent import BaseAgent
from utils.communication.message import Message
from utils.data.polygon_api import PolygonDataFetcher


class RiskAgent(BaseAgent):
    """Enhanced Risk Management Agent for comprehensive market and position risk assessment.

    This agent evaluates multiple dimensions of risk including market risk,
    liquidity risk, volatility risk, correlation risk, and tail risk. It uses
    a combination of statistical models and historical stress testing to provide 
    comprehensive risk assessments.

    Attributes:
        data_fetcher (PolygonDataFetcher): API client for market data
        market_indices (dict): Key market indices to track
        vix_threshold (float): VIX threshold for high volatility
        volatility_window (int): Window for historical volatility
        risk_state (dict): Current risk assessment state
    """

    def __init__(self, polygon_api_key=None):
        """Initialize the Risk Agent with required components.

        Args:
            polygon_api_key (str, optional): API key for Polygon.io
        """
        super().__init__("RiskAgent")

        # Initialize data fetcher
        self.data_fetcher = PolygonDataFetcher(api_key=polygon_api_key)

        # Key market indices to track
        self.market_indices = {
            'SPY': 'S&P 500',  # Overall market
            'VIX': 'Volatility Index',  # Market volatility
            'TLT': 'Treasury Bond ETF',  # Bond market
            'DXY': 'US Dollar Index',  # Currency strength
            'GLD': 'Gold ETF'  # Safe haven asset
        }

        # Risk thresholds
        self.vix_threshold = 25.0  # VIX above this indicates high volatility
        self.volatility_window = 20  # Window for historical volatility

        # Initialize risk state tracking
        self.risk_state = {
            'overall_risk_score': 0.5,  # Default medium risk
            'volatility_risk': 0.5,
            'liquidity_risk': 0.5,
            'correlation_risk': 0.5,
            'tail_risk': 0.5,
            'macro_risk': 0.5,
            'last_updated': datetime.now().isoformat()
        }

        # Initialize logger
        self.logger = logging.getLogger(self.__class__.__name__)

    async def process_message(self, message):
        """Process incoming messages and respond with risk assessments.

        Args:
            message (Message): Incoming message with action request

        Returns:
            Message: Response with risk assessment or error
        """
        self.logger.info(f"Processing message: {message.content.get('action', 'unknown')}")

        if message.content.get('action') == 'assess_market_risk':
            # Assess overall market risk
            risk_assessment = await self.assess_market_risk()

            # Return assessment
            return Message(
                sender=self.name,
                receiver=message.sender,
                content={
                    'market_risk_assessment': risk_assessment,
                    'timestamp': datetime.now().isoformat()
                },
                correlation_id=message.message_id
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
                sender=self.name,
                receiver=message.sender,
                content={
                    'position_risk_assessment': risk_assessment,
                    'ticker': ticker,
                    'timestamp': datetime.now().isoformat()
                },
                correlation_id=message.message_id
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
                sender=self.name,
                receiver=message.sender,
                content={
                    'options_risk_assessment': risk_assessment,
                    'ticker': ticker,
                    'option_type': option_type,
                    'strike_price': strike_price,
                    'expiration_date': expiration_date,
                    'timestamp': datetime.now().isoformat()
                },
                correlation_id=message.message_id
            )
        else:
            # Unsupported action
            return Message(
                sender=self.name,
                receiver=message.sender,
                content={
                    'error': 'Unsupported action',
                    'supported_actions': ['assess_market_risk', 'assess_position_risk', 'assess_options_risk']
                },
                correlation_id=message.message_id
            )

    async def assess_market_risk(self):
        """Perform a comprehensive market risk assessment.

        This method gathers data on multiple market indices, volatility measures,
        correlation matrices, liquidity indicators, and macroeconomic factors to
        provide a multi-dimensional risk assessment.

        Returns:
            dict: Comprehensive market risk assessment
        """
        try:
            self.logger.info("Assessing overall market risk")

            # Get current date and lookback periods
            end_date = datetime.now()
            start_date_short = end_date - timedelta(days=30)
            start_date_long = end_date - timedelta(days=365)

            # Format dates for API
            end_date_str = end_date.strftime("%Y-%m-%d")
            start_date_short_str = start_date_short.strftime("%Y-%m-%d")
            start_date_long_str = start_date_long.strftime("%Y-%m-%d")

            # 1. Get market index data
            market_data = {}
            raw_data = {}
            for ticker in self.market_indices.keys():
                try:
                    # Get both short and long term data
                    data_short = self.data_fetcher.get_historical_data(
                        ticker=ticker,
                        from_date=start_date_short_str,
                        to_date=end_date_str,
                        timeframe="day"
                    )

                    data_long = self.data_fetcher.get_historical_data(
                        ticker=ticker,
                        from_date=start_date_long_str,
                        to_date=end_date_str,
                        timeframe="day"
                    )

                    if not data_short.empty and not data_long.empty:
                        # Store raw data for correlation analysis
                        raw_data[ticker] = data_short

                        # Calculate key metrics
                        market_data[ticker] = {
                            'current_price': data_short['close'].iloc[-1],
                            'daily_return': data_short['close'].pct_change().iloc[-1],
                            'weekly_return': data_short['close'].pct_change(5).iloc[-1],
                            'monthly_return': data_short['close'].pct_change(20).iloc[-1],
                            'volatility_1m': data_short['close'].pct_change().std() * np.sqrt(252),
                            'volatility_1y': data_long['close'].pct_change().std() * np.sqrt(252),
                            'maximum_drawdown': self._calculate_drawdown(data_long['close']),
                            'rsi': self._calculate_rsi(data_short['close']),
                            '50d_200d_sma_ratio': (data_short['close'].rolling(50).mean().iloc[-1] /
                                                   data_long['close'].rolling(200).mean().iloc[-1])
                        }
                except Exception as e:
                    self.logger.error(f"Error getting data for {ticker}: {e}")
                    # Set placeholder data
                    market_data[ticker] = {
                        'error': str(e),
                        'status': 'data_missing'
                    }

            # 2. Calculate correlation matrix (if we have SPY data)
            correlation_matrix = {}
            if 'SPY' in raw_data:
                for ticker, data in raw_data.items():
                    if ticker != 'SPY':
                        # Align data
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

            # 3. Assess market regime
            market_regime = "Unknown"
            regime_confidence = 0.0
            if 'SPY' in market_data and 'VIX' in market_data:
                spy_data = market_data['SPY']
                vix_data = market_data['VIX']

                # Classify market regime
                if spy_data.get('rsi', 50) > 70 and vix_data.get('current_price', 20) < 15:
                    market_regime = "Bullish"
                    regime_confidence = 0.8
                elif spy_data.get('rsi', 50) < 30 and vix_data.get('current_price', 20) > 25:
                    market_regime = "Bearish"
                    regime_confidence = 0.8
                elif 40 <= spy_data.get('rsi', 50) <= 60 and 15 <= vix_data.get('current_price', 20) <= 25:
                    market_regime = "Neutral"
                    regime_confidence = 0.7
                elif vix_data.get('current_price', 20) > 30:
                    market_regime = "High Volatility"
                    regime_confidence = 0.9
                else:
                    # Use SMA ratio as additional signal
                    sma_ratio = spy_data.get('50d_200d_sma_ratio', 1.0)
                    if sma_ratio > 1.05:
                        market_regime = "Bullish Trend"
                        regime_confidence = 0.6
                    elif sma_ratio < 0.95:
                        market_regime = "Bearish Trend"
                        regime_confidence = 0.6
                    else:
                        market_regime = "Sideways"
                        regime_confidence = 0.5

            # 4. Calculate risk metrics using multiple approaches

            # 4.1 Volatility-based risk
            vix_value = market_data.get('VIX', {}).get('current_price', 20)
            spy_vol = market_data.get('SPY', {}).get('volatility_1m', 0.15)

            volatility_risk = min(1.0, (vix_value / 40.0) * 0.7 + (spy_vol / 0.4) * 0.3)

            # 4.2 Correlation-based risk (market stress indicator)
            correlation_values = list(correlation_matrix.values())
            if correlation_values:
                avg_correlation = sum(correlation_values) / len(correlation_values)
                # High average correlation can indicate market stress
                correlation_risk = (avg_correlation + 1) / 2  # Convert from [-1,1] to [0,1]
            else:
                correlation_risk = 0.5  # Default if data unavailable
                avg_correlation = None

            # 4.3 Liquidity risk (simplified proxy using TLT and SPY volumes)
            tlt_volume = raw_data.get('TLT', {}).get('volume', pd.Series([0])).mean()
            spy_volume = raw_data.get('SPY', {}).get('volume', pd.Series([0])).mean()
            spy_volume_ratio = spy_volume / spy_volume.shift(20).mean() if isinstance(spy_volume, pd.Series) else 1.0

            liquidity_risk = 0.5  # Default
            if isinstance(spy_volume_ratio,
                          (int, float)) and spy_volume_ratio < 0.7:  # Volume significantly below average
                liquidity_risk = 0.8
            elif isinstance(spy_volume_ratio,
                            (int, float)) and spy_volume_ratio > 1.3:  # Volume significantly above average
                liquidity_risk = 0.6  # Higher than normal but not necessarily bad

            # 4.4 Tail risk (using maximum drawdown and VIX)
            spy_drawdown = market_data.get('SPY', {}).get('maximum_drawdown', -0.1)
            tail_risk = min(1.0, (abs(spy_drawdown) / 0.2) * 0.7 + (vix_value / 40.0) * 0.3)

            # 4.5 Macroeconomic risk (using bond and dollar indices as proxies)
            tlt_change = market_data.get('TLT', {}).get('monthly_return', 0)
            dxy_change = market_data.get('DXY', {}).get('monthly_return', 0)
            gold_change = market_data.get('GLD', {}).get('monthly_return', 0)

            # Economic stress often shows in bonds rising, USD rising, and gold rising
            macro_risk = 0.5  # Default neutral
            if isinstance(tlt_change, (int, float)) and isinstance(dxy_change, (int, float)) and isinstance(gold_change,
                                                                                                            (int,
                                                                                                             float)):
                if tlt_change > 0.05 and dxy_change > 0.02 and gold_change > 0.03:
                    macro_risk = 0.8  # High risk flight to safety
                elif tlt_change < -0.05 and dxy_change < -0.02:  # Bonds selling off, dollar weak
                    if gold_change > 0.05:  # But gold rising (inflation fears)
                        macro_risk = 0.7
                    else:
                        macro_risk = 0.4  # Risk-on environment

            # 5. Calculate overall risk score (weighted average of components)
            risk_weights = {
                'volatility_risk': 0.3,
                'correlation_risk': 0.15,
                'liquidity_risk': 0.15,
                'tail_risk': 0.2,
                'macro_risk': 0.2
            }

            risk_components = {
                'volatility_risk': volatility_risk,
                'correlation_risk': correlation_risk,
                'liquidity_risk': liquidity_risk,
                'tail_risk': tail_risk,
                'macro_risk': macro_risk
            }

            overall_risk_score = sum([
                score * risk_weights[name] for name, score in risk_components.items()
            ])

            # 6. Create categorical risk level
            risk_level = "Medium"
            if overall_risk_score < 0.3:
                risk_level = "Very Low"
            elif overall_risk_score < 0.4:
                risk_level = "Low"
            elif overall_risk_score < 0.6:
                risk_level = "Medium"
            elif overall_risk_score < 0.75:
                risk_level = "High"
            else:
                risk_level = "Very High"

            # 7. Store risk state
            self.risk_state = {
                'overall_risk_score': float(overall_risk_score),
                'risk_level': risk_level,
                'volatility_risk': float(volatility_risk),
                'liquidity_risk': float(liquidity_risk),
                'correlation_risk': float(correlation_risk),
                'tail_risk': float(tail_risk),
                'macro_risk': float(macro_risk),
                'last_updated': datetime.now().isoformat()
            }

            # 8. Create detailed risk assessment
            risk_assessment = {
                'overall_risk': {
                    'score': float(overall_risk_score),
                    'level': risk_level,
                    'interpretation': f"Current market risk is {risk_level.lower()} based on multiple indicators."
                },
                'market_regime': {
                    'regime': market_regime,
                    'confidence': float(regime_confidence),
                    'interpretation': f"Market appears to be in a {market_regime.lower()} regime."
                },
                'risk_components': risk_components,
                'market_indicators': {k: {sk: float(sv) if isinstance(sv, (int, float)) else sv
                                          for sk, sv in v.items()}
                                      for k, v in market_data.items()},
                'correlation_matrix': {k: float(v) for k, v in correlation_matrix.items()},
                'risk_factors': {
                    'vix_value': float(vix_value),
                    'spy_volatility': float(spy_vol),
                    'spy_drawdown': float(spy_drawdown),
                    'average_correlation': float(avg_correlation) if avg_correlation is not None else None,
                },
                'recommendations': self._generate_risk_recommendations(overall_risk_score, market_regime),
                'timestamp': datetime.now().isoformat()
            }

            return risk_assessment

        except Exception as e:
            self.logger.exception(f"Error assessing market risk: {e}")
            return {
                'error': str(e),
                'status': 'assessment_failed',
                'fallback_risk_level': 'Medium',  # Conservative default
                'timestamp': datetime.now().isoformat()
            }

    async def assess_position_risk(self, ticker, position_size, price=None, portfolio_value=None):
        """Assess risk for a specific stock position.

        Evaluates position-specific risks including ticker volatility,
        concentration risk, liquidity risk, and correlation with market.

        Args:
            ticker (str): Stock symbol
            position_size (float): Size of position in number of shares
            price (float, optional): Current price per share
            portfolio_value (float, optional): Total portfolio value for concentration calculation

        Returns:
            dict: Position risk assessment
        """
        try:
            self.logger.info(f"Assessing position risk for {ticker}")

            # Get current date and lookback periods
            end_date = datetime.now()
            start_date = end_date - timedelta(days=90)  # 3 months history

            # Format dates
            end_date_str = end_date.strftime("%Y-%m-%d")
            start_date_str = start_date.strftime("%Y-%m-%d")

            # 1. Get stock and market data
            stock_data = self.data_fetcher.get_historical_data(
                ticker=ticker,
                from_date=start_date_str,
                to_date=end_date_str,
                timeframe="day"
            )

            market_data = self.data_fetcher.get_historical_data(
                ticker="SPY",  # Use SPY as market proxy
                from_date=start_date_str,
                to_date=end_date_str,
                timeframe="day"
            )

            # 2. Calculate position metrics
            if price is None and not stock_data.empty:
                price = stock_data['close'].iloc[-1]

            position_value = price * position_size if price is not None else 0

            # 3. Calculate stock specific risk metrics
            metrics = {}
            if not stock_data.empty:
                # Calculate returns
                stock_returns = stock_data['close'].pct_change().dropna()

                # Volatility (annualized)
                metrics['volatility'] = stock_returns.std() * np.sqrt(252)

                # Sharpe ratio (simplified, assuming risk-free rate of 0)
                metrics['sharpe_ratio'] = stock_returns.mean() / stock_returns.std() * np.sqrt(252) \
                    if stock_returns.std() > 0 else 0

                # Maximum drawdown
                metrics['max_drawdown'] = self._calculate_drawdown(stock_data['close'])

                # Average daily volume
                metrics['avg_daily_volume'] = stock_data['volume'].mean()

                # Days to liquidate (assuming can trade 10% of daily volume)
                metrics['days_to_liquidate'] = position_size / (metrics['avg_daily_volume'] * 0.1) \
                    if metrics['avg_daily_volume'] > 0 else float('inf')

                # Beta (if market data available)
                if not market_data.empty:
                    # Align dates
                    combined = pd.merge(
                        stock_returns,
                        market_data['close'].pct_change(),
                        left_index=True,
                        right_index=True,
                        suffixes=('_stock', '_market')
                    ).dropna()

                    if not combined.empty and combined.shape[0] > 1:
                        # Calculate beta
                        covariance = combined.cov().iloc[0, 1]
                        market_variance = combined.iloc[:, 1].var()
                        metrics['beta'] = covariance / market_variance if market_variance > 0 else 1.0

                        # Calculate correlation
                        metrics['correlation_to_market'] = combined.corr().iloc[0, 1]

            # 4. Concentration risk
            concentration_risk = 0.5  # Default medium
            if portfolio_value and portfolio_value > 0:
                concentration = position_value / portfolio_value
                if concentration > 0.2:  # Position > 20% of portfolio
                    concentration_risk = 0.9
                elif concentration > 0.1:  # Position > 10% of portfolio
                    concentration_risk = 0.7
                elif concentration > 0.05:  # Position > 5% of portfolio
                    concentration_risk = 0.5
                else:  # Position < 5% of portfolio
                    concentration_risk = 0.3

            # 5. Liquidity risk
            liquidity_risk = 0.5  # Default medium
            if 'days_to_liquidate' in metrics:
                if metrics['days_to_liquidate'] > 20:  # Would take 20+ days to exit
                    liquidity_risk = 0.9
                elif metrics['days_to_liquidate'] > 5:  # 5-20 days to exit
                    liquidity_risk = 0.7
                elif metrics['days_to_liquidate'] > 1:  # 1-5 days to exit
                    liquidity_risk = 0.5
                else:  # < 1 day to exit
                    liquidity_risk = 0.2

            # 6. Volatility risk
            volatility_risk = 0.5  # Default medium
            if 'volatility' in metrics:
                if metrics['volatility'] > 0.5:  # Extremely volatile (50%+ annualized)
                    volatility_risk = 0.9
                elif metrics['volatility'] > 0.3:  # Highly volatile (30-50% annualized)
                    volatility_risk = 0.8
                elif metrics['volatility'] > 0.2:  # Above average volatility (20-30% annualized)
                    volatility_risk = 0.7
                elif metrics['volatility'] > 0.15:  # Average volatility (15-20% annualized)
                    volatility_risk = 0.5
                else:  # Below average volatility (<15% annualized)
                    volatility_risk = 0.3

            # 7. Market correlation risk (high correlation can be risky during market downturns)
            correlation_risk = 0.5  # Default medium
            if 'correlation_to_market' in metrics:
                correlation = metrics['correlation_to_market']
                if correlation > 0.8:  # Highly correlated to market
                    correlation_risk = 0.7
                elif correlation > 0.5:  # Moderately correlated
                    correlation_risk = 0.5
                elif correlation > 0:  # Low positive correlation
                    correlation_risk = 0.3
                elif correlation > -0.5:  # Low negative correlation (potentially beneficial)
                    correlation_risk = 0.3
                else:  # Highly negatively correlated
                    correlation_risk = 0.4  # Could be useful for hedging but has its own risks

            # 8. Calculate VaR (Value at Risk)
            var_95 = None
            cvar_95 = None
            if 'volatility' in metrics and price is not None:
                # Simplified VaR calculation assuming normal distribution
                # 95% confidence level (1.645 standard deviations)
                var_95 = position_value * metrics['volatility'] * 1.645 / np.sqrt(252)

                # Conditional VaR (Expected Shortfall) - estimate average loss beyond VaR
                cvar_95 = position_value * metrics['volatility'] * 2.063 / np.sqrt(252)  # ~95% confidence

                metrics['var_95_daily'] = var_95
                metrics['cvar_95_daily'] = cvar_95

            # 9. Overall position risk score (weighted average)
            risk_weights = {
                'volatility_risk': 0.3,
                'liquidity_risk': 0.2,
                'concentration_risk': 0.25,
                'correlation_risk': 0.15,
                'market_risk': 0.1  # Use overall market risk as a factor
            }

            # Get current market risk or use default
            market_risk = self.risk_state.get('overall_risk_score', 0.5)

            risk_components = {
                'volatility_risk': volatility_risk,
                'liquidity_risk': liquidity_risk,
                'concentration_risk': concentration_risk,
                'correlation_risk': correlation_risk,
                'market_risk': market_risk
            }

            overall_risk_score = sum([
                score * risk_weights[name] for name, score in risk_components.items()
            ])

            # 10. Risk level category
            risk_level = "Medium"
            if overall_risk_score < 0.3:
                risk_level = "Very Low"
            elif overall_risk_score < 0.4:
                risk_level = "Low"
            elif overall_risk_score < 0.6:
                risk_level = "Medium"
            elif overall_risk_score < 0.75:
                risk_level = "High"
            else:
                risk_level = "Very High"

            # 11. Generate position-specific recommendations
            recommendations = self._generate_position_recommendations(
                ticker=ticker,
                risk_level=risk_level,
                metrics=metrics,
                risk_components=risk_components
            )

            # 12. Create the complete risk assessment
            position_risk_assessment = {
                'ticker': ticker,
                'position_size': position_size,
                'position_value': position_value if price is not None else None,
                'current_price': price,
                'overall_risk': {
                    'score': float(overall_risk_score),
                    'level': risk_level,
                    'interpretation': f"Position risk for {ticker} is {risk_level.lower()}."
                },
                'risk_components': {k: float(v) for k, v in risk_components.items()},
                'metrics': {k: float(v) if isinstance(v, (int, float)) else v for k, v in metrics.items()},
                'recommendations': recommendations,
                'timestamp': datetime.now().isoformat()

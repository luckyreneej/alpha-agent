import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import datetime
import logging
import os
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OptionsCalculator:
    """Options pricing calculator with Black-Scholes model and Greeks."""

    @staticmethod
    def black_scholes(S, K, T, r, sigma, option_type='call'):
        """
        Calculate option price using Black-Scholes model.
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiry (in years)
            r: Risk-free interest rate (decimal)
            sigma: Volatility (decimal)
            option_type: 'call' or 'put'
            
        Returns:
            Option price
        """
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type.lower() == 'call':
            price = S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
        else:  # put option
            price = K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)

        return price

    @staticmethod
    def calculate_delta(S, K, T, r, sigma, option_type='call'):
        """Calculate Delta (change in option price with respect to underlying price)."""
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

        if option_type.lower() == 'call':
            delta = stats.norm.cdf(d1)
        else:  # put option
            delta = stats.norm.cdf(d1) - 1

        return delta

    @staticmethod
    def calculate_gamma(S, K, T, r, sigma):
        """Calculate Gamma (second derivative of option price with respect to underlying price)."""
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        gamma = stats.norm.pdf(d1) / (S * sigma * np.sqrt(T))
        return gamma

    @staticmethod
    def calculate_theta(S, K, T, r, sigma, option_type='call'):
        """Calculate Theta (change in option price with respect to time decay)."""
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type.lower() == 'call':
            theta = -S * stats.norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * stats.norm.cdf(d2)
        else:  # put option
            theta = -S * stats.norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * stats.norm.cdf(-d2)

        # Convert to daily theta (from annual)
        theta = theta / 365
        return theta

    @staticmethod
    def calculate_vega(S, K, T, r, sigma):
        """Calculate Vega (change in option price with respect to volatility)."""
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        vega = S * np.sqrt(T) * stats.norm.pdf(d1)
        # Standard vega is for 1% change in volatility
        vega = vega / 100
        return vega

    @staticmethod
    def calculate_rho(S, K, T, r, sigma, option_type='call'):
        """Calculate Rho (change in option price with respect to risk-free rate)."""
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type.lower() == 'call':
            rho = K * T * np.exp(-r * T) * stats.norm.cdf(d2)
        else:  # put option
            rho = -K * T * np.exp(-r * T) * stats.norm.cdf(-d2)

        # Standard rho is for 1% change in interest rate
        rho = rho / 100
        return rho

    @staticmethod
    def calculate_implied_volatility(S, K, T, r, option_price, option_type='call'):
        """Calculate implied volatility from option price."""

        # Define objective function: difference between BS price and observed price
        def objective(sigma):
            return abs(OptionsCalculator.black_scholes(S, K, T, r, sigma, option_type) - option_price)

        # Use optimization to find implied volatility
        initial_guess = 0.3  # Start with 30% volatility as guess
        bounds = [(0.001, 5.0)]  # IV bounds between 0.1% and 500%
        result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')

        if result.success:
            return result.x[0]
        else:
            logger.warning("Implied volatility calculation did not converge")
            return None

    @staticmethod
    def calculate_all_greeks(S, K, T, r, sigma, option_type='call'):
        """Calculate all option Greeks."""
        price = OptionsCalculator.black_scholes(S, K, T, r, sigma, option_type)
        delta = OptionsCalculator.calculate_delta(S, K, T, r, sigma, option_type)
        gamma = OptionsCalculator.calculate_gamma(S, K, T, r, sigma)
        theta = OptionsCalculator.calculate_theta(S, K, T, r, sigma, option_type)
        vega = OptionsCalculator.calculate_vega(S, K, T, r, sigma)
        rho = OptionsCalculator.calculate_rho(S, K, T, r, sigma, option_type)

        return {
            'price': price,
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }


class OptionsStrategy:
    """Options strategy generation and evaluation."""

    def __init__(self, model_path=None):
        self.model_path = model_path if model_path else "models/trained/options_ml_model.pkl"
        self.model = None
        self.scaler = StandardScaler()
        self.ml_model = None

    def build_ml_model(self):
        """Build machine learning model for options premium adjustment."""
        # Default to XGBoost model
        self.ml_model = xgb.XGBRegressor(
            max_depth=7,
            learning_rate=0.01,
            n_estimators=1000,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='reg:squarederror'
        )
        return self.ml_model

    def train_ml_model(self, X, y):
        """Train ML model for options adjustment."""
        if self.ml_model is None:
            self.ml_model = self.build_ml_model()

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train model
        self.ml_model.fit(X_scaled, y)
        return self.ml_model

    def save_model(self):
        """Save trained model to disk."""
        if self.ml_model is None:
            logger.error("No model to save")
            return False

        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, 'wb') as f:
            pickle.dump({
                'model': self.ml_model,
                'scaler': self.scaler
            }, f)

        logger.info(f"Model saved to {self.model_path}")
        return True

    def load_model(self):
        """Load trained model from disk."""
        try:
            with open(self.model_path, 'rb') as f:
                saved_model = pickle.load(f)
                self.ml_model = saved_model['model']
                self.scaler = saved_model['scaler']

            logger.info(f"Model loaded from {self.model_path}")
            return True
        except (FileNotFoundError, KeyError, pickle.UnpicklingError) as e:
            logger.error(f"Error loading model: {e}")
            return False

    def predict_option_value(self, features):
        """Predict option value with ML adjustment."""
        if self.ml_model is None:
            logger.error("Model not trained or loaded")
            return None

        # Extract option parameters from features
        S = features['stock_price']
        K = features['strike_price']
        T = features['time_to_expiry']
        r = features['risk_free_rate']
        sigma = features['implied_volatility']
        option_type = features.get('option_type', 'call')

        # Calculate base price using Black-Scholes
        bs_price = OptionsCalculator.black_scholes(S, K, T, r, sigma, option_type)

        # Prepare ML features
        ml_features = np.array([
            features['volume'],
            features['open_interest'],
            features.get('delta', OptionsCalculator.calculate_delta(S, K, T, r, sigma, option_type)),
            features.get('gamma', OptionsCalculator.calculate_gamma(S, K, T, r, sigma)),
            features.get('theta', OptionsCalculator.calculate_theta(S, K, T, r, sigma, option_type)),
            features.get('vega', OptionsCalculator.calculate_vega(S, K, T, r, sigma))
        ]).reshape(1, -1)

        # Scale features
        ml_features_scaled = self.scaler.transform(ml_features)

        # Get ML adjustment
        ml_adjustment = self.ml_model.predict(ml_features_scaled)[0]

        # Apply adjustment to Black-Scholes price
        adjusted_price = bs_price * (1 + ml_adjustment)

        return {
            'bs_price': bs_price,
            'ml_adjustment': ml_adjustment,
            'adjusted_price': adjusted_price
        }

    def calculate_risk_metrics(self, position, S, volatility, days=30):
        """Calculate risk metrics for an options position."""
        # VaR calculation (95% confidence)
        daily_change = S * volatility / np.sqrt(252)
        var_95 = daily_change * 1.65  # 95% confidence interval

        # Expected shortfall
        expected_shortfall = daily_change * 2.06  # Approx for 95% ES

        # Max drawdown estimate
        max_drawdown_est = var_95 * np.sqrt(days)

        return {
            'var_95': var_95,
            'expected_shortfall': expected_shortfall,
            'max_drawdown_est': max_drawdown_est
        }

    def create_strategy_recommendation(self, stock_price, strike_price, time_to_expiry, risk_free_rate,
                                       implied_volatility, volume, open_interest, predict_price=None,
                                       market_condition='neutral'):
        """Create options strategy recommendation based on market conditions and price prediction."""
        # Calculate option Greeks
        call_greeks = OptionsCalculator.calculate_all_greeks(
            stock_price, strike_price, time_to_expiry, risk_free_rate, implied_volatility, 'call')
        put_greeks = OptionsCalculator.calculate_all_greeks(
            stock_price, strike_price, time_to_expiry, risk_free_rate, implied_volatility, 'put')

        # Create features for ML adjustment
        call_features = {
            'stock_price': stock_price,
            'strike_price': strike_price,
            'time_to_expiry': time_to_expiry,
            'risk_free_rate': risk_free_rate,
            'implied_volatility': implied_volatility,
            'option_type': 'call',
            'volume': volume,
            'open_interest': open_interest,
            'delta': call_greeks['delta'],
            'gamma': call_greeks['gamma'],
            'theta': call_greeks['theta'],
            'vega': call_greeks['vega']
        }

        put_features = call_features.copy()
        put_features['option_type'] = 'put'
        put_features['delta'] = put_greeks['delta']
        put_features['theta'] = put_greeks['theta']

        # Apply ML adjustment if model is loaded
        if self.ml_model is not None:
            call_prices = self.predict_option_value(call_features)
            put_prices = self.predict_option_value(put_features)
        else:
            call_prices = {'adjusted_price': call_greeks['price'], 'bs_price': call_greeks['price'], 'ml_adjustment': 0}
            put_prices = {'adjusted_price': put_greeks['price'], 'bs_price': put_greeks['price'], 'ml_adjustment': 0}

        # Get risk metrics
        risk_metrics = self.calculate_risk_metrics(
            1, stock_price, implied_volatility, days=int(time_to_expiry * 365))

        # Generate strategy recommendations based on market conditions and price prediction
        strategies = []

        # If we have a price prediction, use it for directional strategies
        if predict_price is not None:
            price_change_pct = (predict_price - stock_price) / stock_price

            # Significant upside potential
            if price_change_pct > 0.05:
                strategies.append({
                    'name': 'Long Call',
                    'type': 'directional_bullish',
                    'description': 'Buy call options to profit from significant price increase',
                    'implementation': [{'action': 'buy', 'option_type': 'call', 'strike': strike_price,
                                        'premium': call_prices['adjusted_price']}],
                    'max_profit': 'Unlimited',
                    'max_loss': call_prices['adjusted_price'],
                    'breakeven': strike_price + call_prices['adjusted_price']
                })

                # Consider Bull Call Spread for more conservative approach
                higher_strike = strike_price * 1.10  # 10% higher strike
                higher_call = OptionsCalculator.black_scholes(
                    stock_price, higher_strike, time_to_expiry, risk_free_rate, implied_volatility, 'call')

                strategies.append({
                    'name': 'Bull Call Spread',
                    'type': 'directional_bullish_limited',
                    'description': 'Buy call at lower strike, sell call at higher strike to reduce cost',
                    'implementation': [
                        {'action': 'buy', 'option_type': 'call', 'strike': strike_price,
                         'premium': call_prices['adjusted_price']},
                        {'action': 'sell', 'option_type': 'call', 'strike': higher_strike,
                         'premium': higher_call}
                    ],
                    'max_profit': (higher_strike - strike_price) - (call_prices['adjusted_price'] - higher_call),
                    'max_loss': call_prices['adjusted_price'] - higher_call,
                    'breakeven': strike_price + (call_prices['adjusted_price'] - higher_call)
                })

            # Significant downside potential
            elif price_change_pct < -0.05:
                strategies.append({
                    'name': 'Long Put',
                    'type': 'directional_bearish',
                    'description': 'Buy put options to profit from significant price decrease',
                    'implementation': [{'action': 'buy', 'option_type': 'put', 'strike': strike_price,
                                        'premium': put_prices['adjusted_price']}],
                    'max_profit': strike_price - put_prices['adjusted_price'],
                    'max_loss': put_prices['adjusted_price'],
                    'breakeven': strike_price - put_prices['adjusted_price']
                })

                # Consider Bear Put Spread for more conservative approach
                lower_strike = strike_price * 0.90  # 10% lower strike
                lower_put = OptionsCalculator.black_scholes(
                    stock_price, lower_strike, time_to_expiry, risk_free_rate, implied_volatility, 'put')

                strategies.append({
                    'name': 'Bear Put Spread',
                    'type': 'directional_bearish_limited',
                    'description': 'Buy put at higher strike, sell put at lower strike to reduce cost',
                    'implementation': [
                        {'action': 'buy', 'option_type': 'put', 'strike': strike_price,
                         'premium': put_prices['adjusted_price']},
                        {'action': 'sell', 'option_type': 'put', 'strike': lower_strike,
                         'premium': lower_put}
                    ],
                    'max_profit': (strike_price - lower_strike) - (put_prices['adjusted_price'] - lower_put),
                    'max_loss': put_prices['adjusted_price'] - lower_put,
                    'breakeven': strike_price - (put_prices['adjusted_price'] - lower_put)
                })

        # Strategies based on market conditions regardless of price prediction
        if market_condition == 'high_volatility':
            # High volatility - consider selling options
            strategies.append({
                'name': 'Iron Condor',
                'type': 'neutral_range',
                'description': 'Sell call and put spreads to profit from a range-bound market with high volatility',
                'implementation': [
                    {'action': 'sell', 'option_type': 'call', 'strike': strike_price * 1.05},
                    {'action': 'buy', 'option_type': 'call', 'strike': strike_price * 1.10},
                    {'action': 'sell', 'option_type': 'put', 'strike': strike_price * 0.95},
                    {'action': 'buy', 'option_type': 'put', 'strike': strike_price * 0.90}
                ],
                'volatility_exposure': 'negative',
                'profit_from': 'time decay and decreasing volatility'
            })

        elif market_condition == 'low_volatility':
            # Low volatility - consider buying options
            strategies.append({
                'name': 'Straddle',
                'type': 'volatility_long',
                'description': 'Buy call and put at the same strike to profit from volatility increase',
                'implementation': [
                    {'action': 'buy', 'option_type': 'call', 'strike': strike_price,
                     'premium': call_prices['adjusted_price']},
                    {'action': 'buy', 'option_type': 'put', 'strike': strike_price,
                     'premium': put_prices['adjusted_price']}
                ],
                'max_loss': call_prices['adjusted_price'] + put_prices['adjusted_price'],
                'breakeven_upper': strike_price + call_prices['adjusted_price'] + put_prices['adjusted_price'],
                'breakeven_lower': strike_price - (call_prices['adjusted_price'] + put_prices['adjusted_price']),
                'volatility_exposure': 'positive'
            })

        else:  # neutral market condition
            strategies.append({
                'name': 'Covered Call',
                'type': 'income_strategy',
                'description': 'Hold the underlying stock and sell call options for income',
                'implementation': [
                    {'action': 'buy', 'asset_type': 'stock', 'quantity': 100},
                    {'action': 'sell', 'option_type': 'call', 'strike': strike_price * 1.05,
                     'quantity': 1}
                ],
                'max_profit': 'Limited',
                'max_loss': 'Substantial (stock price can go to zero)'
            })

            strategies.append({
                'name': 'Cash-Secured Put',
                'type': 'income_strategy',
                'description': 'Sell put options for income with cash reserves to secure the position',
                'implementation': [
                    {'action': 'sell', 'option_type': 'put', 'strike': strike_price * 0.95,
                     'quantity': 1}
                ],
                'max_profit': 'Limited to premium received',
                'max_loss': 'Substantial (must buy stock at strike if assigned)'
            })

        # Add risk metrics to all strategies
        for strategy in strategies:
            strategy['risk_metrics'] = risk_metrics

        return strategies


class OptionsBacktester:
    """Backtesting framework for options strategies."""

    def __init__(self, initial_capital=10000):
        self.initial_capital = initial_capital
        self.portfolio_value = initial_capital
        self.positions = []
        self.history = []

    def add_position(self, position_type, entry_date, exit_date, premium, contracts=1):
        """Add a position to the backtester."""
        position = {
            'type': position_type,  # 'long_call', 'short_put', etc.
            'entry_date': entry_date,
            'exit_date': exit_date,
            'premium': premium,
            'contracts': contracts,
            'value_history': [],
            'pnl': None
        }

        self.positions.append(position)
        return len(self.positions) - 1  # Return position index

    def update_position(self, position_idx, date, value):
        """Update a position's value for a given date."""
        if 0 <= position_idx < len(self.positions):
            self.positions[position_idx]['value_history'].append({
                'date': date,
                'value': value
            })
        else:
            logger.error(f"Invalid position index: {position_idx}")

    def close_position(self, position_idx, exit_value):
        """Close a position and calculate P&L."""
        if 0 <= position_idx < len(self.positions):
            position = self.positions[position_idx]
            entry_premium = position['premium']
            contracts = position['contracts']

            # Calculate P&L based on position type
            if position['type'].startswith('long'):
                pnl = (exit_value - entry_premium) * contracts * 100  # Each contract is for 100 shares
            else:  # short position
                pnl = (entry_premium - exit_value) * contracts * 100

            position['pnl'] = pnl
            self.portfolio_value += pnl

            return pnl
        else:
            logger.error(f"Invalid position index: {position_idx}")
            return None

    def simulate_strategy(self, strategy_func, market_data, params=None):
        """Simulate an options strategy over historical data."""
        self.portfolio_value = self.initial_capital
        self.positions = []
        self.history = []

        # Reset portfolio to initial capital
        portfolio_history = [{
            'date': market_data.iloc[0]['date'],
            'value': self.initial_capital
        }]

        # Apply strategy to each date in the market data
        for i in range(len(market_data) - 1):  # -1 to allow for next-day evaluation
            current_data = market_data.iloc[i]
            next_data = market_data.iloc[i + 1]

            # Execute strategy
            actions = strategy_func(current_data, params=params)

            # Process actions
            for action in actions:
                if action['action'] == 'open':
                    position_idx = self.add_position(
                        action['position_type'],
                        current_data['date'],
                        None,  # Exit date to be determined
                        action['premium'],
                        action.get('contracts', 1)
                    )

                elif action['action'] == 'close' and 'position_idx' in action:
                    self.close_position(action['position_idx'], action['exit_value'])

            # Update portfolio value
            portfolio_history.append({
                'date': next_data['date'],
                'value': self.portfolio_value
            })

        # Close any remaining open positions using the last data point
        last_data = market_data.iloc[-1]
        for i, position in enumerate(self.positions):
            if position['pnl'] is None:  # Position still open
                # Use a simple estimate for exit value - this should be more sophisticated in reality
                exit_value = position['premium']  # Placeholder
                self.close_position(i, exit_value)

        # Calculate performance metrics
        initial_value = portfolio_history[0]['value']
        final_value = portfolio_history[-1]['value']
        returns = (final_value - initial_value) / initial_value

        # Calculate daily returns for Sharpe ratio
        daily_returns = []
        for i in range(1, len(portfolio_history)):
            prev_value = portfolio_history[i - 1]['value']
            curr_value = portfolio_history[i]['value']
            daily_return = (curr_value - prev_value) / prev_value
            daily_returns.append(daily_return)

        avg_daily_return = np.mean(daily_returns)
        std_daily_return = np.std(daily_returns)

        # Assuming risk-free rate of 0 for simplicity
        sharpe_ratio = avg_daily_return / std_daily_return * np.sqrt(252) if std_daily_return > 0 else 0

        # Calculate max drawdown
        peak = portfolio_history[0]['value']
        max_drawdown = 0

        for point in portfolio_history:
            if point['value'] > peak:
                peak = point['value']
            drawdown = (peak - point['value']) / peak
            max_drawdown = max(max_drawdown, drawdown)

        performance = {
            'initial_capital': self.initial_capital,
            'final_value': self.portfolio_value,
            'total_return': returns,
            'total_return_pct': returns * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'positions': len(self.positions),
            'winning_positions': sum(1 for p in self.positions if p['pnl'] > 0),
            'losing_positions': sum(1 for p in self.positions if p['pnl'] < 0)
        }

        return performance, portfolio_history

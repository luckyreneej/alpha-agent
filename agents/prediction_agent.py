import numpy as np
import pandas as pd
import logging
import datetime
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base_agent import BaseAgent
from models.stock_price_model import (
    TechnicalIndicators, StockPriceModel, 
    LSTMModel, XGBoostModel, ProphetModel, EnsembleModel
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PredictionAgent(BaseAgent):
    """
    Agent responsible for stock price predictions using various models.
    Integrates with the Agent Communicator for data sharing.
    """
    
    def __init__(self, agent_id, communicator, config=None):
        super().__init__(agent_id, communicator)
        self.config = config or {
            'default_model': 'ensemble',
            'prediction_horizon': 7,  # days
            'use_alpha_factors': True,
            'models': {
                'lstm': {
                    'enabled': True,
                    'model_path': 'models/trained/lstm_model.pkl'
                },
                'xgboost': {
                    'enabled': True,
                    'model_path': 'models/trained/xgboost_model.pkl'
                },
                'prophet': {
                    'enabled': True,
                    'model_path': 'models/trained/prophet_model.pkl'
                },
                'ensemble': {
                    'enabled': True,
                    'model_path': 'models/trained/ensemble_model.pkl',
                    'weights': {'lstm': 0.4, 'xgboost': 0.4, 'prophet': 0.2}
                }
            }
        }
        self.models = {}
        self._load_models()
        
        # Register for data updates
        self.communicator.register_event_handler('stock_data_updated', self._on_stock_data_updated)
    
    def _load_models(self):
        """
        Load trained prediction models.
        """
        for model_name, model_config in self.config['models'].items():
            if model_config['enabled']:
                try:
                    if model_name == 'lstm':
                        model = LSTMModel(model_path=model_config['model_path'])
                    elif model_name == 'xgboost':
                        model = XGBoostModel(model_path=model_config['model_path'])
                    elif model_name == 'prophet':
                        model = ProphetModel(model_path=model_config['model_path'])
                    elif model_name == 'ensemble':
                        model = EnsembleModel(model_path=model_config['model_path'])
                    else:
                        logger.warning(f"Unknown model type: {model_name}")
                        continue
                        
                    if model.load_model():
                        self.models[model_name] = model
                        logger.info(f"Loaded model: {model_name}")
                    else:
                        logger.warning(f"Failed to load model: {model_name}, will train new model if needed")
                        self.models[model_name] = model  # Still add the model for potential training
                except Exception as e:
                    logger.error(f"Error loading model {model_name}: {e}")
    
    def _on_stock_data_updated(self, key, value):
        """
        Handle stock data updates and trigger predictions.
        """
        logger.info("Stock data updated, triggering predictions")
        self.generate_predictions()
    
    def preprocess_data(self, stock_data):
        """
        Preprocess stock data for prediction models.
        
        Args:
            stock_data: DataFrame with stock price data
            
        Returns:
            Preprocessed data for each model type
        """
        processed_data = {}
        
        # Add technical indicators
        df_with_indicators = TechnicalIndicators.add_indicators(stock_data)
        
        # Add alpha factors if enabled
        if self.config['use_alpha_factors']:
            df_with_indicators = TechnicalIndicators.calculate_alpha_factors(df_with_indicators)
        
        # Drop NaN values
        df_clean = df_with_indicators.dropna()
        
        # Prepare data for each model type
        if 'lstm' in self.models:
            # LSTM requires sequences
            X_lstm, y_lstm = self.models['lstm'].preprocess_data(df_clean)
            processed_data['lstm'] = (X_lstm, y_lstm)
        
        if 'xgboost' in self.models:
            X_xgb, y_xgb = self.models['xgboost'].preprocess_data(df_clean)
            processed_data['xgboost'] = (X_xgb, y_xgb)
        
        if 'prophet' in self.models:
            prophet_df, _ = self.models['prophet'].preprocess_data(df_clean)
            processed_data['prophet'] = prophet_df
        
        # Original data for ensemble model
        processed_data['raw_data'] = df_clean
        
        return processed_data
    
    def train_models(self, processed_data):
        """
        Train or update prediction models with new data.
        
        Args:
            processed_data: Dictionary of preprocessed data for each model
        """
        for model_name, model in self.models.items():
            try:
                if model_name == 'lstm':
                    X, y = processed_data['lstm']
                    model.train(X, y)
                elif model_name == 'xgboost':
                    X, y = processed_data['xgboost']
                    model.train(X, y)
                elif model_name == 'prophet':
                    prophet_df = processed_data['prophet']
                    model.train(prophet_df)
                elif model_name == 'ensemble':
                    # Ensemble model requires all other models to be trained first
                    continue
                    
                model.save_model()
                logger.info(f"Trained and saved model: {model_name}")
            except Exception as e:
                logger.error(f"Error training model {model_name}: {e}")
        
        # Now train ensemble if available
        if 'ensemble' in self.models:
            try:
                # Collect data for ensemble
                ensemble_X = {}
                ensemble_y = {}
                
                if 'lstm' in processed_data:
                    ensemble_X['lstm'] = processed_data['lstm'][0]
                    ensemble_y['lstm'] = processed_data['lstm'][1]
                
                if 'xgboost' in processed_data:
                    ensemble_X['xgboost'] = processed_data['xgboost'][0]
                    ensemble_y['xgboost'] = processed_data['xgboost'][1]
                
                self.models['ensemble'].train(
                    ensemble_X, ensemble_y, 
                    df_raw=processed_data['raw_data'] if 'prophet' in self.models else None
                )
                self.models['ensemble'].save_model()
                logger.info("Trained and saved ensemble model")
            except Exception as e:
                logger.error(f"Error training ensemble model: {e}")
    
    def generate_predictions(self):
        """
        Generate stock price predictions using trained models.
        """
        # Get latest stock data
        stock_data = self.communicator.get_data('stock_data')
        if stock_data is None or stock_data.empty:
            logger.error("No stock data available for predictions")
            return
            
        # Preprocess data
        processed_data = self.preprocess_data(stock_data)
        
        # Train or update models if needed (when models don't have persisted files)
        need_training = any(not hasattr(model, 'model') or model.model is None 
                           for model in self.models.values())
        if need_training:
            logger.info("One or more models need training")
            self.train_models(processed_data)
        
        # Generate predictions
        predictions = {}
        horizon = self.config['prediction_horizon']
        
        # Use the default model for prediction
        default_model = self.config['default_model']
        
        if default_model in self.models:
            try:
                model = self.models[default_model]
                
                if default_model == 'lstm':
                    X_pred = processed_data['lstm'][0][-1:] if len(processed_data['lstm'][0]) > 0 else None
                    if X_pred is not None:
                        pred = model.predict(X_pred)
                        pred = model.scaler_y.inverse_transform(pred)
                        predictions[default_model] = pred[0][0]
                
                elif default_model == 'xgboost':
                    X_pred = processed_data['xgboost'][0][-1:] if len(processed_data['xgboost'][0]) > 0 else None
                    if X_pred is not None:
                        pred = model.predict(X_pred)
                        pred = model.scaler_y.inverse_transform(pred)
                        predictions[default_model] = pred[0][0]
                
                elif default_model == 'prophet':
                    # Generate future dataframe
                    future = model.model.make_future_dataframe(periods=horizon)
                    forecast = model.predict(future)
                    # Get the prediction for tomorrow
                    tomorrow = pd.Timestamp.now().normalize() + pd.Timedelta(days=1)
                    tomorrow_pred = forecast[forecast['ds'] == tomorrow]
                    if not tomorrow_pred.empty:
                        predictions[default_model] = tomorrow_pred['yhat'].values[0]
                
                elif default_model == 'ensemble':
                    # Collect data for ensemble prediction
                    ensemble_X = {}
                    
                    if 'lstm' in self.models and 'lstm' in processed_data:
                        ensemble_X['lstm'] = processed_data['lstm'][0][-1:]
                    
                    if 'xgboost' in self.models and 'xgboost' in processed_data:
                        ensemble_X['xgboost'] = processed_data['xgboost'][0][-1:]
                    
                    # Prophet future dataframe
                    prophet_future = None
                    if 'prophet' in self.models:
                        prophet = self.models['prophet']
                        prophet_future = prophet.model.make_future_dataframe(periods=horizon)
                    
                    pred = model.predict(ensemble_X, df_prophet=prophet_future)
                    # Apply inverse scaling
                    if 'xgboost' in self.models:
                        pred = self.models['xgboost'].scaler_y.inverse_transform(pred)
                    predictions[default_model] = pred[0][0]
            
            except Exception as e:
                logger.error(f"Error generating prediction with {default_model}: {e}")
        else:
            logger.error(f"Default model {default_model} not available")
        
        # Update data store with predictions
        if predictions:
            # Store the prediction from the default model
            self.communicator.update_data('stock_price_prediction', predictions.get(default_model))
            
            # Store all model predictions for comparison
            self.communicator.update_data('model_predictions', predictions)
            
            logger.info(f"Generated predictions: {predictions}")
            
            # Notify other agents about new predictions
            self.communicator.broadcast_message(
                self.agent_id,
                {'type': 'prediction_update', 'predictions': predictions}
            )
            
    def analyze_market_conditions(self):
        """
        Analyze current market conditions to inform prediction adjustments.
        """
        # Get market conditions data
        vix_data = self.communicator.get_data('vix_index')
        sector_performance = self.communicator.get_data('sector_performance')
        market_breadth = self.communicator.get_data('market_breadth')
        economic_indicators = self.communicator.get_data('economic_indicators')
        
        # Default market condition if data is not available
        market_condition = 'neutral'
        
        # Analyze VIX for volatility regime
        if vix_data is not None:
            latest_vix = vix_data[-1] if isinstance(vix_data, list) else vix_data
            if latest_vix > 30:
                market_condition = 'high_volatility'
            elif latest_vix < 15:
                market_condition = 'low_volatility'
        
        # Store market condition assessment
        self.communicator.update_data('market_condition', market_condition)
        logger.info(f"Market condition assessed as: {market_condition}")
        
        return market_condition
    
    def adjust_prediction_with_alpha_factors(self, base_prediction):
        """
        Adjust price prediction using alpha factors.
        
        Args:
            base_prediction: Base price prediction
            
        Returns:
            Adjusted prediction
        """
        if not self.config['use_alpha_factors']:
            return base_prediction
            
        # Get latest alpha factors
        stock_data = self.communicator.get_data('stock_data')
        if stock_data is None or stock_data.empty:
            return base_prediction
        
        try:
            # Calculate alpha factors
            df_with_alphas = TechnicalIndicators.calculate_alpha_factors(stock_data)
            latest_data = df_with_alphas.iloc[-1]
            
            # Use alphas to adjust prediction
            adjustment = 0
            
            # Alpha1 impact - normalized between -0.01 and 0.01
            if 'alpha1' in latest_data:
                alpha1_impact = latest_data['alpha1'] * 0.01
                adjustment += alpha1_impact
            
            # Alpha12 impact - normalized between -0.01 and 0.01
            if 'alpha12' in latest_data:
                alpha12_impact = latest_data['alpha12'] * 0.01
                adjustment += alpha12_impact
            
            # Alpha101 impact - normalized between -0.02 and 0.02
            if 'alpha101' in latest_data:
                alpha101_impact = latest_data['alpha101'] * 0.02
                adjustment += alpha101_impact
            
            # Apply adjustment
            adjusted_prediction = base_prediction * (1 + adjustment)
            
            logger.info(f"Adjusted prediction from {base_prediction} to {adjusted_prediction} using alpha factors")
            return adjusted_prediction
        
        except Exception as e:
            logger.error(f"Error adjusting prediction with alpha factors: {e}")
            return base_prediction
    
    def generate_confidence_intervals(self, prediction, stock_data):
        """
        Generate confidence intervals for the prediction.
        
        Args:
            prediction: Base price prediction
            stock_data: Historical stock data
            
        Returns:
            Dictionary with confidence intervals
        """
        try:
            # Calculate historical volatility
            returns = stock_data['close'].pct_change().dropna()
            hist_volatility = returns.std()
            
            # Calculate daily price change
            avg_price = stock_data['close'].mean()
            daily_price_change = avg_price * hist_volatility
            
            # Generate confidence intervals
            ci_95_lower = prediction - (daily_price_change * 1.96)  # 95% CI
            ci_95_upper = prediction + (daily_price_change * 1.96)
            
            ci_99_lower = prediction - (daily_price_change * 2.576)  # 99% CI
            ci_99_upper = prediction + (daily_price_change * 2.576)
            
            confidence_intervals = {
                'prediction': prediction,
                'ci_95': {'lower': ci_95_lower, 'upper': ci_95_upper},
                'ci_99': {'lower': ci_99_lower, 'upper': ci_99_upper},
                'volatility': hist_volatility
            }
            
            self.communicator.update_data('prediction_confidence', confidence_intervals)
            return confidence_intervals
        
        except Exception as e:
            logger.error(f"Error generating confidence intervals: {e}")
            return {
                'prediction': prediction,
                'ci_95': {'lower': prediction * 0.95, 'upper': prediction * 1.05},
                'ci_99': {'lower': prediction * 0.93, 'upper': prediction * 1.07}
            }
    
    def run(self):
        """
        Main agent loop for continuous operation.
        """
        logger.info(f"{self.agent_id} starting operation")
        
        while True:
            try:
                # Process incoming messages
                messages = self.communicator.get_messages(self.agent_id)
                for message in messages:
                    self._process_message(message)
                
                # Generate predictions if we have data and models
                if self.communicator.get_data('stock_data') is not None and self.models:
                    self.generate_predictions()
                    
                    # Analyze market conditions
                    market_condition = self.analyze_market_conditions()
                    
                    # Get base prediction
                    base_prediction = self.communicator.get_data('stock_price_prediction')
                    if base_prediction is not None:
                        # Adjust prediction with alpha factors
                        adjusted_prediction = self.adjust_prediction_with_alpha_factors(base_prediction)
                        self.communicator.update_data('adjusted_prediction', adjusted_prediction)
                        
                        # Generate confidence intervals
                        stock_data = self.communicator.get_data('stock_data')
                        if stock_data is not None:
                            self.generate_confidence_intervals(adjusted_prediction, stock_data)
                
                # Sleep to avoid high CPU usage
                time.sleep(60)  # Check every minute
            
            except Exception as e:
                logger.error(f"Error in {self.agent_id} operation: {e}")
                time.sleep(60)  # Wait before retrying
    
    def _process_message(self, message):
        """
        Process incoming messages from other agents.
        
        Args:
            message: Incoming message
        """
        msg_type = message.get('type')
        
        if msg_type == 'request_prediction':
            logger.info(f"Received prediction request from {message['from_agent']}")
            self.generate_predictions()
        
        elif msg_type == 'update_market_data':
            logger.info(f"Received market data update from {message['from_agent']}")
            # Market data should already be in the data store
            # Just trigger a prediction update
            self.analyze_market_conditions()
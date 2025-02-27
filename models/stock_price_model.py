import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
from prophet import Prophet
import matplotlib.pyplot as plt
import talib
from scipy import stats
import logging
import os
import pickle

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TechnicalIndicators:
    """Generate technical indicators for stock price prediction."""
    
    @staticmethod
    def add_indicators(df):
        """
        Add technical indicators to a dataframe with OHLCV data.
        
        Args:
            df: DataFrame with at least 'open', 'high', 'low', 'close', 'volume' columns
            
        Returns:
            DataFrame with added technical indicators
        """
        # Make a copy to avoid modifying the original dataframe
        df_with_indicators = df.copy()
        
        # Ensure necessary columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")
        
        # Price-based indicators
        df_with_indicators['ma_5'] = talib.SMA(df['close'], timeperiod=5)
        df_with_indicators['ma_20'] = talib.SMA(df['close'], timeperiod=20)
        df_with_indicators['ma_50'] = talib.SMA(df['close'], timeperiod=50)
        df_with_indicators['ma_200'] = talib.SMA(df['close'], timeperiod=200)
        
        # Golden Cross / Death Cross
        df_with_indicators['golden_cross'] = (df_with_indicators['ma_50'] > df_with_indicators['ma_200']).astype(int)
        
        # Momentum indicators
        df_with_indicators['rsi'] = talib.RSI(df['close'], timeperiod=14)
        df_with_indicators['macd'], df_with_indicators['macd_signal'], df_with_indicators['macd_hist'] = talib.MACD(
            df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        
        # Volatility indicators
        df_with_indicators['bbands_upper'], df_with_indicators['bbands_middle'], df_with_indicators['bbands_lower'] = talib.BBANDS(
            df['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        df_with_indicators['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        
        # Volume indicators
        df_with_indicators['obv'] = talib.OBV(df['close'], df['volume'])
        df_with_indicators['ad'] = talib.AD(df['high'], df['low'], df['close'], df['volume'])
        
        # Return lags (1, 3, 5 days)
        for i in [1, 3, 5]:
            df_with_indicators[f'return_{i}d'] = df['close'].pct_change(i)
        
        return df_with_indicators
    
    @staticmethod
    def calculate_alpha_factors(df):
        """
        Calculate alpha factors based on the 101 alphas.
        
        Args:
            df: DataFrame with at least 'open', 'high', 'low', 'close', 'volume' columns
            
        Returns:
            DataFrame with added alpha factors
        """
        df_with_alphas = df.copy()
        
        # Alpha1: (rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) - 0.5)
        returns = df['close'].pct_change()
        returns_std = returns.rolling(window=20).std()
        close = df['close']
        
        # Create the conditional array
        condition = returns < 0
        result = np.where(condition, returns_std, close)
        
        # Apply signed power
        signed_power = np.power(result, 2)
        
        # For each point, look back 5 periods and find the index of max value
        rolling_argmax = np.zeros_like(signed_power)
        for i in range(5, len(signed_power)):
            rolling_argmax[i] = np.argmax(signed_power[i-5:i+1]) + (i-5)
        
        # Rank and subtract 0.5
        alpha1 = pd.Series(rolling_argmax).rank(pct=True) - 0.5
        df_with_alphas['alpha1'] = alpha1
        
        # Alpha101: ((close - open) / ((high - low) + 0.001))
        df_with_alphas['alpha101'] = (df['close'] - df['open']) / ((df['high'] - df['low']) + 0.001)
        
        # Alpha12: (sign(delta(volume, 1)) * (-1 * delta(close, 1)))
        volume_delta = df['volume'].diff(1)
        close_delta = df['close'].diff(1)
        df_with_alphas['alpha12'] = np.sign(volume_delta) * (-1 * close_delta)
        
        return df_with_alphas

class StockPriceModel:
    """Base class for stock price prediction models."""
    
    def __init__(self, model_name, model_path=None):
        self.model_name = model_name
        self.model_path = model_path if model_path else f"models/trained/{model_name}.pkl"
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
    def preprocess_data(self, df, target_col='close', sequence_length=60):
        """
        Preprocess data for model training/prediction
        
        Args:
            df: DataFrame with features
            target_col: Target column to predict
            sequence_length: Length of sequences for sequential models
            
        Returns:
            X, y: Processed features and target
        """
        # Add technical indicators
        df_processed = TechnicalIndicators.add_indicators(df)
        
        # Add alpha factors
        df_processed = TechnicalIndicators.calculate_alpha_factors(df_processed)
        
        # Drop NaN values resulting from indicators calculation
        df_processed.dropna(inplace=True)
        
        # Create target variable (next day close price)
        df_processed['target'] = df_processed[target_col].shift(-1)
        df_processed.dropna(inplace=True)
        
        # Select features and target
        feature_cols = [col for col in df_processed.columns if col not in 
                        ['target', 'date', 'timestamp', 'symbol', 'Unnamed: 0']]
        
        X = df_processed[feature_cols].values
        y = df_processed['target'].values.reshape(-1, 1)
        
        # Scale data
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y)
        
        # For sequence models (LSTM), reshape data into sequences
        if self.model_name.lower() == 'lstm':
            X_seq = []
            y_seq = []
            
            for i in range(len(X_scaled) - sequence_length):
                X_seq.append(X_scaled[i:i + sequence_length])
                y_seq.append(y_scaled[i + sequence_length])
                
            return np.array(X_seq), np.array(y_seq)
        
        return X_scaled, y_scaled
    
    def save_model(self):
        """Save model to disk."""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler_X': self.scaler_X,
                'scaler_y': self.scaler_y
            }, f)
        logger.info(f"Model saved to {self.model_path}")
    
    def load_model(self):
        """Load model from disk."""
        try:
            with open(self.model_path, 'rb') as f:
                saved_model = pickle.load(f)
                self.model = saved_model['model']
                self.scaler_X = saved_model['scaler_X']
                self.scaler_y = saved_model['scaler_y']
            logger.info(f"Model loaded from {self.model_path}")
            return True
        except (FileNotFoundError, KeyError, pickle.UnpicklingError) as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def train(self, X, y):
        """Train model with provided data."""
        raise NotImplementedError("Subclasses must implement train()")
    
    def predict(self, X):
        """Make predictions with trained model."""
        raise NotImplementedError("Subclasses must implement predict()")
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance."""
        if self.model is None:
            logger.error("Model not trained or loaded")
            return None
            
        y_pred = self.predict(X_test)
        
        # Inverse transform if data was scaled
        y_test_original = self.scaler_y.inverse_transform(y_test)
        y_pred_original = self.scaler_y.inverse_transform(y_pred)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test_original, y_pred_original)
        mse = mean_squared_error(y_test_original, y_pred_original)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test_original, y_pred_original)
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2
        }

class LSTMModel(StockPriceModel):
    """LSTM model for stock price prediction."""
    
    def __init__(self, model_path=None, config=None):
        super().__init__('lstm', model_path)
        self.config = config or {
            'units': [50, 100, 50],
            'dropout': 0.2,
            'batch_size': 32,
            'epochs': 100,
            'patience': 10
        }
    
    def build_model(self, input_shape):
        """Build LSTM model architecture."""
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(self.config['units'][0], return_sequences=True, input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(Dropout(self.config['dropout']))
        
        # Second LSTM layer
        model.add(LSTM(self.config['units'][1], return_sequences=True))
        model.add(BatchNormalization())
        model.add(Dropout(self.config['dropout']))
        
        # Third LSTM layer
        model.add(LSTM(self.config['units'][2]))
        model.add(BatchNormalization())
        model.add(Dropout(self.config['dropout']))
        
        # Output layer
        model.add(Dense(1))
        
        # Compile model
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        return model
    
    def train(self, X, y):
        """Train LSTM model."""
        if self.model is None:
            self.model = self.build_model((X.shape[1], X.shape[2]))
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=self.config['patience'],
            restore_best_weights=True
        )
        
        # Split data into train and validation sets
        val_split = 0.2
        split_idx = int(X.shape[0] * (1 - val_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            callbacks=[early_stopping],
            verbose=1
        )
        
        return history
    
    def predict(self, X):
        """Make predictions with LSTM model."""
        if self.model is None:
            logger.error("Model not trained or loaded")
            return None
        
        return self.model.predict(X)

class XGBoostModel(StockPriceModel):
    """XGBoost model for stock price prediction."""
    
    def __init__(self, model_path=None, config=None):
        super().__init__('xgboost', model_path)
        self.config = config or {
            'max_depth': 7,
            'learning_rate': 0.01,
            'n_estimators': 1000,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'early_stopping_rounds': 50
        }
    
    def train(self, X, y):
        """Train XGBoost model."""
        # Split data into train and validation sets
        val_split = 0.2
        split_idx = int(X.shape[0] * (1 - val_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Create DMatrix objects
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        # Set up evaluation list
        evallist = [(dtrain, 'train'), (dval, 'validation')]
        
        # Train model
        self.model = xgb.train(
            self.config,
            dtrain,
            num_boost_round=self.config['n_estimators'],
            evals=evallist,
            early_stopping_rounds=self.config['early_stopping_rounds'],
            verbose_eval=100
        )
        
        return self.model
    
    def predict(self, X):
        """Make predictions with XGBoost model."""
        if self.model is None:
            logger.error("Model not trained or loaded")
            return None
        
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest).reshape(-1, 1)

class ProphetModel(StockPriceModel):
    """Prophet model for stock price prediction."""
    
    def __init__(self, model_path=None, config=None):
        super().__init__('prophet', model_path)
        self.config = config or {
            'changepoint_prior_scale': 0.05,
            'seasonality_mode': 'multiplicative',
            'weekly_seasonality': True,
            'daily_seasonality': False,
            'yearly_seasonality': True
        }
    
    def preprocess_data(self, df, target_col='close', **kwargs):
        """
        Preprocess data specifically for Prophet model.
        
        Args:
            df: DataFrame with date and price columns
            target_col: Target column to predict
            
        Returns:
            prophet_df: DataFrame formatted for Prophet
        """
        # Prophet requires 'ds' (date) and 'y' (target) columns
        prophet_df = pd.DataFrame()
        prophet_df['ds'] = pd.to_datetime(df.index) if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df['date'])
        prophet_df['y'] = df[target_col]
        
        # Add additional regressors
        if 'volume' in df.columns:
            prophet_df['volume'] = df['volume']
        
        return prophet_df, None  # No separate y needed for Prophet
    
    def train(self, X, y=None):
        """Train Prophet model."""
        # For Prophet, X contains both features and target in Prophet's format
        self.model = Prophet(
            changepoint_prior_scale=self.config['changepoint_prior_scale'],
            seasonality_mode=self.config['seasonality_mode'],
            weekly_seasonality=self.config['weekly_seasonality'],
            daily_seasonality=self.config['daily_seasonality'],
            yearly_seasonality=self.config['yearly_seasonality']
        )
        
        # Add additional regressors
        if 'volume' in X.columns:
            self.model.add_regressor('volume')
        
        self.model.fit(X)
        return self.model
    
    def predict(self, X=None, periods=30, freq='D'):
        """
        Make predictions with Prophet model.
        
        Args:
            X: For Prophet, this can be None. If provided, should be a dataframe with 'ds' column
            periods: Number of periods to forecast
            freq: Frequency of forecast
            
        Returns:
            Forecast dataframe
        """
        if self.model is None:
            logger.error("Model not trained or loaded")
            return None
        
        if X is not None:
            future = X
        else:
            future = self.model.make_future_dataframe(periods=periods, freq=freq)
            
        forecast = self.model.predict(future)
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    
    def save_model(self):
        """Save Prophet model with custom serialization."""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        # Prophet model doesn't pickle well, so we use its built-in serialization
        prophet_path = self.model_path.replace('.pkl', '.json')
        with open(prophet_path, 'w') as f:
            self.model.serialize_model(f)
            
        logger.info(f"Prophet model saved to {prophet_path}")
    
    def load_model(self):
        """Load Prophet model with custom serialization."""
        try:
            prophet_path = self.model_path.replace('.pkl', '.json')
            
            # Load the model from the JSON file
            with open(prophet_path, 'r') as f:
                self.model = Prophet.deserialize_model(f.read())
                
            logger.info(f"Prophet model loaded from {prophet_path}")
            return True
        except (FileNotFoundError, ValueError) as e:
            logger.error(f"Error loading Prophet model: {e}")
            return False

class EnsembleModel(StockPriceModel):
    """Ensemble model combining LSTM, XGBoost, and Prophet predictions."""
    
    def __init__(self, model_path=None, config=None):
        super().__init__('ensemble', model_path)
        self.config = config or {
            'weights': {'lstm': 0.4, 'xgboost': 0.4, 'prophet': 0.2}
        }
        self.models = {}
        
    def train(self, X, y, df_raw=None):
        """
        Train all component models in the ensemble.
        
        Args:
            X: Features for LSTM/XGBoost
            y: Target for LSTM/XGBoost
            df_raw: Raw dataframe for Prophet
        """
        # Train LSTM model
        lstm_model = LSTMModel()
        lstm_model.train(X['lstm'], y['lstm'])
        self.models['lstm'] = lstm_model
        
        # Train XGBoost model
        xgb_model = XGBoostModel()
        xgb_model.train(X['xgboost'], y['xgboost'])
        self.models['xgboost'] = xgb_model
        
        # Train Prophet model
        if df_raw is not None:
            prophet_model = ProphetModel()
            prophet_data, _ = prophet_model.preprocess_data(df_raw)
            prophet_model.train(prophet_data)
            self.models['prophet'] = prophet_model
            
        return self.models
    
    def predict(self, X, df_prophet=None, periods=1):
        """
        Make predictions using weighted average of all models.
        
        Args:
            X: Dictionary containing model-specific input data
            df_prophet: Prophet-formatted dataframe
            periods: Number of periods to forecast
            
        Returns:
            Weighted average prediction
        """
        if not self.models:
            logger.error("No models in ensemble")
            return None
        
        predictions = {}
        
        # Get LSTM predictions
        if 'lstm' in self.models and 'lstm' in X:
            predictions['lstm'] = self.models['lstm'].predict(X['lstm'])
            
        # Get XGBoost predictions
        if 'xgboost' in self.models and 'xgboost' in X:
            predictions['xgboost'] = self.models['xgboost'].predict(X['xgboost'])
            
        # Get Prophet predictions
        if 'prophet' in self.models and df_prophet is not None:
            prophet_forecast = self.models['prophet'].predict(df_prophet)
            # Extract the prediction for the next period
            predictions['prophet'] = prophet_forecast['yhat'].values[-periods:].reshape(-1, 1)
            
        # Calculate weighted average
        weighted_sum = np.zeros_like(list(predictions.values())[0])
        weight_sum = 0
        
        for model_name, pred in predictions.items():
            weight = self.config['weights'].get(model_name, 0)
            weighted_sum += pred * weight
            weight_sum += weight
            
        # Normalize by sum of weights
        if weight_sum > 0:
            weighted_pred = weighted_sum / weight_sum
        else:
            weighted_pred = weighted_sum
            
        return weighted_pred
    
    def save_model(self):
        """Save all models in the ensemble."""
        for model_name, model in self.models.items():
            model_path = self.model_path.replace('.pkl', f'_{model_name}.pkl')
            model.model_path = model_path
            model.save_model()
            
        # Save ensemble configuration
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, 'wb') as f:
            pickle.dump({
                'config': self.config,
                'model_names': list(self.models.keys())
            }, f)
            
        logger.info(f"Ensemble configuration saved to {self.model_path}")
    
    def load_model(self):
        """Load all models in the ensemble."""
        try:
            # Load ensemble configuration
            with open(self.model_path, 'rb') as f:
                saved_config = pickle.load(f)
                self.config = saved_config['config']
                model_names = saved_config['model_names']
            
            # Load individual models
            for model_name in model_names:
                model_path = self.model_path.replace('.pkl', f'_{model_name}.pkl')
                
                if model_name == 'lstm':
                    model = LSTMModel(model_path=model_path)
                elif model_name == 'xgboost':
                    model = XGBoostModel(model_path=model_path)
                elif model_name == 'prophet':
                    model = ProphetModel(model_path=model_path)
                else:
                    logger.warning(f"Unknown model type: {model_name}")
                    continue
                    
                if model.load_model():
                    self.models[model_name] = model
                    
            logger.info(f"Ensemble model loaded from {self.model_path}")
            return len(self.models) > 0
            
        except (FileNotFoundError, KeyError, pickle.UnpicklingError) as e:
            logger.error(f"Error loading ensemble model: {e}")
            return False
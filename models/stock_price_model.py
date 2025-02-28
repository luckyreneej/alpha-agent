import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
from prophet import Prophet
import talib
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
        """Add technical indicators to a dataframe with OHLCV data."""
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

        # Momentum indicators
        df_with_indicators['rsi'] = talib.RSI(df['close'], timeperiod=14)
        df_with_indicators['macd'], df_with_indicators['macd_signal'], _ = talib.MACD(
            df['close'], fastperiod=12, slowperiod=26, signalperiod=9)

        # Volatility indicators
        df_with_indicators['bbands_upper'], _, df_with_indicators['bbands_lower'] = talib.BBANDS(
            df['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        df_with_indicators['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)

        # Volume indicators
        df_with_indicators['obv'] = talib.OBV(df['close'], df['volume'])

        # Return lags (1, 5 days)
        for i in [1, 5]:
            df_with_indicators[f'return_{i}d'] = df['close'].pct_change(i)

        return df_with_indicators

    @staticmethod
    def calculate_alpha_factors(df):
        """Calculate key alpha factors."""
        df_with_alphas = df.copy()

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
        """Preprocess data for model training/prediction."""
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
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

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

    def train(self, X, y):
        """Train model with provided data."""
        raise NotImplementedError("Subclasses must implement train()")

    def predict(self, X):
        """Make predictions with trained model."""
        raise NotImplementedError("Subclasses must implement predict()")


class PyTorchLSTMModel(nn.Module):
    """PyTorch LSTM model for time series prediction."""

    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        super(PyTorchLSTMModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM layers
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Batch normalization
        self.bn = nn.BatchNorm1d(hidden_dim)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Output layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """Forward pass through the network."""
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))

        # Take the output from the last time step
        out = out[:, -1, :]

        # Batch normalization
        out = self.bn(out)

        # Dropout
        out = self.dropout(out)

        # Final fully connected layer
        out = self.fc(out)

        return out


class LSTMModel(StockPriceModel):
    """LSTM model for stock price prediction using PyTorch."""

    def __init__(self, model_path=None, config=None):
        """Initialize LSTM model."""
        super().__init__('lstm', model_path)
        self.config = config or {
            'hidden_dim': 100,
            'num_layers': 2,
            'dropout': 0.2,
            'batch_size': 32,
            'epochs': 100,
            'learning_rate': 0.001,
            'patience': 10
        }

        # Set up PyTorch device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

    def train(self, X, y):
        """Train LSTM model with PyTorch."""
        # Convert numpy arrays to PyTorch tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        # Create datasets and dataloaders
        dataset = TensorDataset(X_tensor, y_tensor)

        # Split into training and validation sets
        val_split = 0.2
        train_size = int(len(dataset) * (1 - val_split))
        val_size = len(dataset) - train_size

        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False
        )

        # Initialize model
        input_dim = X.shape[2]  # Number of features
        self.model = PyTorchLSTMModel(
            input_dim=input_dim,
            hidden_dim=self.config['hidden_dim'],
            num_layers=self.config['num_layers'],
            output_dim=1,
            dropout=self.config['dropout']
        ).to(self.device)

        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])

        # Early stopping setup
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        history = {'train_loss': [], 'val_loss': []}

        # Training loop
        for epoch in range(self.config['epochs']):
            # Training phase
            self.model.train()
            train_loss = 0.0

            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)
            history['train_loss'].append(train_loss)

            # Validation phase
            self.model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            history['val_loss'].append(val_loss)

            # Print progress
            logger.info(f'Epoch {epoch + 1}/{self.config["epochs"]} - '
                        f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                best_model_state = self.model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= self.config['patience']:
                    logger.info(f'Early stopping at epoch {epoch + 1}')
                    break

        # Load the best model weights
        if best_model_state:
            self.model.load_state_dict(best_model_state)

        return history

    def predict(self, X):
        """Make predictions with LSTM model."""
        if self.model is None:
            logger.error("Model not trained or loaded")
            return None

        # Convert to PyTorch tensor
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

        # Set model to evaluation mode
        self.model.eval()

        # Make predictions
        with torch.no_grad():
            y_pred = self.model(X_tensor).cpu().numpy()

        return y_pred

    def save_model(self):
        """Save PyTorch LSTM model."""
        if self.model is None:
            logger.error("No model to save")
            return

        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

        # Save model state and scalers
        save_dict = {
            'model_state': self.model.state_dict(),
            'model_config': self.config,
            'input_dim': self.model.lstm.input_size,
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y
        }

        torch.save(save_dict, self.model_path)
        logger.info(f"PyTorch model saved to {self.model_path}")

    def load_model(self):
        """Load PyTorch LSTM model."""
        try:
            # Check if file exists
            if not os.path.exists(self.model_path):
                logger.error(f"Model file not found: {self.model_path}")
                return False

            # Load saved state
            save_dict = torch.load(self.model_path, map_location=self.device)

            # Create model with the same configuration
            input_dim = save_dict['input_dim']
            self.config = save_dict['model_config']

            # Initialize model
            self.model = PyTorchLSTMModel(
                input_dim=input_dim,
                hidden_dim=self.config['hidden_dim'],
                num_layers=self.config['num_layers'],
                output_dim=1,
                dropout=self.config['dropout']
            ).to(self.device)

            # Load model weights
            self.model.load_state_dict(save_dict['model_state'])
            self.model.eval()

            # Load scalers
            self.scaler_X = save_dict['scaler_X']
            self.scaler_y = save_dict['scaler_y']

            logger.info(f"PyTorch model loaded from {self.model_path}")
            return True

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False


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
        """Preprocess data specifically for Prophet model."""
        # Prophet requires 'ds' (date) and 'y' (target) columns
        prophet_df = pd.DataFrame()
        prophet_df['ds'] = pd.to_datetime(df.index) if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(
            df['date'])
        prophet_df['y'] = df[target_col]

        # Add additional regressors
        if 'volume' in df.columns:
            prophet_df['volume'] = df['volume']

        return prophet_df, None  # No separate y needed for Prophet

    def train(self, X, y=None):
        """Train Prophet model."""
        # For Prophet, X contains both features and target in Prophet's format
        self.model = Prophet(**self.config)

        # Add additional regressors
        if 'volume' in X.columns:
            self.model.add_regressor('volume')

        self.model.fit(X)
        return self.model

    def predict(self, X=None, periods=30, freq='D'):
        """Make predictions with Prophet model."""
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
        except Exception as e:
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
        """Train all component models in the ensemble."""
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
        """Make predictions using weighted average of all models."""
        if not self.models:
            logger.error("No models in ensemble")
            return None

        predictions = {}

        # Get predictions from each model
        if 'lstm' in self.models and 'lstm' in X:
            predictions['lstm'] = self.models['lstm'].predict(X['lstm'])

        if 'xgboost' in self.models and 'xgboost' in X:
            predictions['xgboost'] = self.models['xgboost'].predict(X['xgboost'])

        if 'prophet' in self.models and df_prophet is not None:
            prophet_forecast = self.models['prophet'].predict(df_prophet)
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

        except Exception as e:
            logger.error(f"Error loading ensemble model: {e}")
            return False
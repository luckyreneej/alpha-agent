# Alpha Agent Trading Framework - Refactored Architecture Design

## Implementation Approach

After analyzing the alpha-agent codebase, we have identified the following key issues that need to be addressed in the refactored architecture:

1. **Duplicate Base Agent Implementations**: There are two different base agent implementations that need to be consolidated:
   - `agents/base_agent.py` (minimal implementation)
   - `utils/communication/base_agent.py` (full implementation with communication features)

2. **TensorFlow to PyTorch Conversion**: The LSTM model in `models/stock_price_model.py` uses TensorFlow/Keras and needs to be refactored to use PyTorch.

3. **Redundant Communication Classes**: There may be redundant components in the communication system that could be simplified.

Our implementation approach will be:

1. **Consolidate Base Agent**: Create a single, unified `BaseAgent` class that combines the best features from both implementations.

2. **Replace TensorFlow with PyTorch**: Implement equivalent PyTorch versions of the LSTM model while maintaining the same API interface to minimize changes in dependent code.

3. **Simplify Communication System**: Streamline the communication components while preserving the robust messaging capabilities.

### Selected Libraries

- **PyTorch** (torch): For deep learning models, replacing TensorFlow/Keras
- **pandas**: For data manipulation and analysis
- **numpy**: For numerical operations
- **scikit-learn**: For data preprocessing and metrics
- **matplotlib**: For visualization
- **talib**: For technical indicators
- **xgboost**: For gradient boosting models
- **prophet**: For time series forecasting

## Data Structures and Interfaces

### Agent System

```python
from typing import Dict, List, Any, Optional, Union, Callable
import time
import logging
import uuid

# Core classes for the unified agent system
class Message:
    def __init__(self, sender_id: str, receiver_id: str, message_type, content=None,
                 correlation_id=None, reply_to=None, metadata=None, priority=None):
        self.id = str(uuid.uuid4())
        self.sender_id = sender_id
        self.receiver_id = receiver_id
        self.message_type = message_type
        self.content = content
        self.correlation_id = correlation_id or self.id
        self.reply_to = reply_to
        self.metadata = metadata or {}
        self.timestamp = time.time()
        self.priority = priority

class MessageType(Enum):
    DATA = "data"
    COMMAND = "command"
    REQUEST = "request"
    RESPONSE = "response"
    BROADCAST = "broadcast"
    ERROR = "error"

class MessagePriority(Enum):
    HIGH = 0
    NORMAL = 1
    LOW = 2

class BaseAgent:
    """Unified base agent class combining functionality from both implementations"""
    
    def __init__(self, agent_id: str, communicator=None):
        self.agent_id = agent_id
        self.communicator = communicator
        self.state: Dict[str, Any] = {}  # Agent's internal state
        self.context: Dict[str, Any] = {}  # Current working context
        self.memory: Dict[str, Any] = {}  # Agent's memory of past events
        self.running = False
        
        # Register with communication manager if provided
        if self.communicator:
            self.communicator.register_agent(self.agent_id)
    
    # Communication methods
    def send_message(self, receiver_id: str, message_type, content, **kwargs):
        # Implementation of send_message
        pass
        
    def broadcast_message(self, content, message_type=MessageType.BROADCAST, **kwargs):
        # Implementation of broadcast_message
        pass
    
    def send_request(self, receiver_id: str, request_type: str, content, timeout=30.0, **kwargs):
        # Implementation of send_request
        pass
    
    # Agent lifecycle methods
    def start(self):
        self.running = True
        logging.info(f"Agent {self.agent_id} started")
    
    def stop(self):
        self.running = False
        logging.info(f"Agent {self.agent_id} stopped")
        
    # Message processing methods
    def process_messages(self):
        # Process all pending messages
        pass
        
    def process_message(self, message):
        # Process a single message - to be overridden by subclasses
        pass
    
    # State management methods
    def update_state(self, key: str, value):
        self.state[key] = value
        
    def get_state(self, key: str, default=None):
        return self.state.get(key, default)
    
    # Memory and context methods (from advanced implementation)
    def update_memory(self, key: str, value):
        self.memory[key] = {
            'value': value,
            'timestamp': time.time()
        }
    
    def recall(self, key: str, default=None):
        memory_entry = self.memory.get(key)
        return memory_entry['value'] if memory_entry else default
    
    def update_context(self, key: str, value):
        self.context[key] = value
        
    def get_context(self, key: str, default=None):
        return self.context.get(key, default)
    
    def clear_context(self):
        self.context = {}
        
    # To be implemented by subclasses
    def run(self):
        """Main agent loop - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement run()")
```

### Models (PyTorch Implementation)

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import pickle

class StockPriceModel:
    """Base class for stock price prediction models."""
    
    def __init__(self, model_name, model_path=None):
        self.model_name = model_name
        self.model_path = model_path if model_path else f"models/trained/{model_name}.pkl"
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
    
    # Common methods remain the same
    def preprocess_data(self, df, target_col='close', sequence_length=60):
        pass
        
    def save_model(self):
        pass
        
    def load_model(self):
        pass
    
    def evaluate(self, X_test, y_test):
        pass

class LSTMModel(StockPriceModel):
    """LSTM model for stock price prediction using PyTorch."""
    
    def __init__(self, model_path=None, config=None):
        super().__init__('lstm', model_path)
        self.config = config or {
            'hidden_size': [50, 100, 50],
            'dropout': 0.2,
            'batch_size': 32,
            'epochs': 100,
            'learning_rate': 0.001,
            'patience': 10
        }
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def build_model(self, input_shape):
        """Build PyTorch LSTM model architecture."""
        # PyTorch implementation instead of TensorFlow
        input_dim = input_shape[1]  # Number of features
        
        class LSTMNet(nn.Module):
            def __init__(self, input_dim, hidden_dim, dropout_prob):
                super(LSTMNet, self).__init__()
                self.hidden_dim = hidden_dim
                
                # First LSTM layer
                self.lstm1 = nn.LSTM(input_dim, hidden_dim[0], batch_first=True)
                self.bn1 = nn.BatchNorm1d(hidden_dim[0])
                self.dropout1 = nn.Dropout(dropout_prob)
                
                # Second LSTM layer
                self.lstm2 = nn.LSTM(hidden_dim[0], hidden_dim[1], batch_first=True)
                self.bn2 = nn.BatchNorm1d(hidden_dim[1])
                self.dropout2 = nn.Dropout(dropout_prob)
                
                # Third LSTM layer
                self.lstm3 = nn.LSTM(hidden_dim[1], hidden_dim[2], batch_first=True)
                self.bn3 = nn.BatchNorm1d(hidden_dim[2])
                self.dropout3 = nn.Dropout(dropout_prob)
                
                # Output layer
                self.fc = nn.Linear(hidden_dim[2], 1)
            
            def forward(self, x):
                # First LSTM layer
                x, _ = self.lstm1(x)
                # Reshape for BatchNorm
                batch_size, seq_len, hidden_dim = x.shape
                x_reshaped = x.contiguous().view(-1, hidden_dim)
                x_bn = self.bn1(x_reshaped)
                x = x_bn.view(batch_size, seq_len, hidden_dim)
                x = self.dropout1(x)
                
                # Second LSTM layer
                x, _ = self.lstm2(x)
                # Reshape for BatchNorm
                batch_size, seq_len, hidden_dim = x.shape
                x_reshaped = x.contiguous().view(-1, hidden_dim)
                x_bn = self.bn2(x_reshaped)
                x = x_bn.view(batch_size, seq_len, hidden_dim)
                x = self.dropout2(x)
                
                # Third LSTM layer
                x, _ = self.lstm3(x)
                # Use only the output from the last time step
                x = x[:, -1, :]
                x = self.bn3(x)
                x = self.dropout3(x)
                
                # Output layer
                x = self.fc(x)
                return x
        
        model = LSTMNet(input_dim, self.config['hidden_size'], self.config['dropout']).to(self.device)
        return model
    
    def train(self, X, y):
        """Train PyTorch LSTM model."""
        if self.model is None:
            self.model = self.build_model((X.shape[1], X.shape[2]))
        
        # Convert numpy arrays to PyTorch tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        # Create DataLoader for batching
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=True)
        
        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        
        # Training loop with early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        val_size = int(len(X) * 0.2)
        train_size = len(X) - val_size
        
        train_dataset = TensorDataset(X_tensor[:train_size], y_tensor[:train_size])
        val_dataset = TensorDataset(X_tensor[train_size:], y_tensor[train_size:])
        train_dataloader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=self.config['batch_size'])
        
        for epoch in range(self.config['epochs']):
            # Training
            self.model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_dataloader:
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model state
                best_model_state = self.model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= self.config['patience']:
                    print(f"Early stopping at epoch {epoch}")
                    # Restore best model state
                    self.model.load_state_dict(best_model_state)
                    break
            
            print(f"Epoch {epoch+1}/{self.config['epochs']}, "
                  f"Train Loss: {train_loss/len(train_dataloader):.4f}, "
                  f"Val Loss: {val_loss/len(val_dataloader):.4f}")
        
        return {'train_loss': train_loss/len(train_dataloader), 'val_loss': val_loss/len(val_dataloader)}
    
    def predict(self, X):
        """Make predictions with PyTorch LSTM model."""
        if self.model is None:
            logging.error("Model not trained or loaded")
            return None
        
        # Convert to PyTorch tensor
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Make prediction
        with torch.no_grad():
            y_pred = self.model(X_tensor)
        
        # Convert back to numpy
        return y_pred.cpu().numpy()
```

## Program Call Flow

### System Initialization and Message Flow

```text
sequenceDiagram
    participant Main as Main
    participant CM as CommunicationManager
    participant PA as PredictionAgent
    participant DA as DataAgent
    participant TA as TradingAgent
    participant PM as PyTorchLSTMModel
    participant XGB as XGBoostModel
    participant ENS as EnsembleModel

    Main->>CM: initialize()
    Main->>PA: create(agent_id, communicator)
    Main->>DA: create(agent_id, communicator)
    Main->>TA: create(agent_id, communicator)
    
    PA->>CM: register_agent(agent_id)
    DA->>CM: register_agent(agent_id)
    TA->>CM: register_agent(agent_id)
    
    PA->>CM: subscribe_to_topic("stock_data_updated")
    TA->>CM: subscribe_to_topic("prediction_update")
    
    Main->>PA: start()
    Main->>DA: start()
    Main->>TA: start()
    
    DA->>CM: publish_to_topic("stock_data_updated", data)
    CM->>PA: notify("stock_data_updated", data)
    
    PA->>PM: load_model()
    PA->>PM: preprocess_data(stock_data)
    PA->>PM: predict(X)
    PA->>XGB: predict(X)
    PA->>ENS: predict(X, df_prophet)
    
    PA->>CM: publish_to_topic("prediction_update", predictions)
    CM->>TA: notify("prediction_update", predictions)
    
    TA->>CM: send_message(DA, "request_data")
    CM->>DA: deliver_message("request_data")
    DA->>CM: send_message(TA, "data_response", data)
    CM->>TA: deliver_message("data_response", data)
```

### PyTorch LSTM Model Training Flow

```text
sequenceDiagram
    participant PA as PredictionAgent
    participant PM as PyTorchLSTMModel
    participant Data as DataPreprocessor
    
    PA->>Data: preprocess_data(stock_data)
    Data-->>PA: processed_data
    
    PA->>PM: train(X, y)
    
    PM->>PM: build_model(input_shape)
    PM->>PM: create DataLoader
    
    loop Each Epoch
        PM->>PM: train batch
        PM->>PM: compute loss
        PM->>PM: backward()
        PM->>PM: optimize()
        PM->>PM: validate batch
        PM->>PM: check early stopping
    end
    
    PM-->>PA: training_history
    PA->>PM: save_model()
```

### Prediction Workflow

```text
sequenceDiagram
    participant PA as PredictionAgent
    participant PM as PyTorchLSTMModel
    participant XGB as XGBoostModel
    participant ENS as EnsembleModel
    participant CM as CommunicationManager
    participant TA as TradingAgent
    
    PA->>CM: get_data("stock_data")
    CM-->>PA: stock_data
    
    PA->>PA: preprocess_data(stock_data)
    
    PA->>PM: predict(X_lstm)
    PM-->>PA: lstm_predictions
    
    PA->>XGB: predict(X_xgb)
    XGB-->>PA: xgb_predictions
    
    PA->>ENS: predict(ensemble_X, df_prophet)
    ENS-->>PA: ensemble_predictions
    
    PA->>PA: adjust_prediction_with_alpha_factors(base_prediction)
    PA->>PA: generate_confidence_intervals(prediction, stock_data)
    
    PA->>CM: update_data("stock_price_prediction", prediction)
    PA->>CM: update_data("model_predictions", all_model_predictions)
    PA->>CM: broadcast_message(prediction_update)
    
    CM->>TA: deliver_message(prediction_update)
    TA->>TA: process_prediction(prediction_update)
```

## Anything UNCLEAR

1. The relationship between `agents/base_agent.py` and the actual agent implementations like `PredictionAgent` is not entirely clear. The simplified base agent doesn't seem to be used consistently across the codebase.

2. The overall architecture appears to have two competing agent systems. The refactored design assumes that we'll standardize on the more complete implementation in `utils/communication/base_agent.py` and eliminate the simpler one.

3. It's not clear from the codebase whether any additional components might rely on specific TensorFlow functionality beyond the LSTM model that we're replacing with PyTorch.

To implement this refactoring:

1. Create the consolidated `BaseAgent` class and update all agent implementations to use it
2. Implement the PyTorch version of the LSTM model with the same API interface
3. Ensure that the prediction agent correctly uses the new PyTorch model
4. Update any model serialization/deserialization for compatibility with PyTorch
5. Test the system thoroughly to ensure feature parity with the original implementation
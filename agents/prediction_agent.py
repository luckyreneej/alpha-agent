from base_agent import BaseAgent


# Prediction Agent: Implements ML models for stock price forecasting
class PredictionAgent(BaseAgent):
    def predict_price(self):
        stock_price = self.coordinator.get_data("stock_price")
        predicted_price = stock_price * 1.02  # Example: Simple 2% increase prediction
        self.coordinator.update_data("predicted_price", predicted_price)

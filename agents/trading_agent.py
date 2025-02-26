from base_agent import BaseAgent


# Trading Agent: Evaluates option chains and suggests trades
class TradingAgent(BaseAgent):
    def suggest_trades(self, threshold):
        options_data = self.coordinator.get_data("option_chain")
        risk_filtered_trades = self.coordinator.get_data("filtered_trades")
        sentiment_analysis = self.coordinator.get_data("sentiment_analysis")
        predicted_price = self.coordinator.get_data("predicted_price")

        trade_suggestions = [trade for trade in risk_filtered_trades if predicted_price > trade['strike_price']]
        self.coordinator.update_data("final_trades", trade_suggestions)

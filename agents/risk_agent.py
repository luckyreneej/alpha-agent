from agents.base_agent import BaseAgent


# Risk Agent: Evaluates risk control measures
class RiskAgent(BaseAgent):
    def assess_risk(self):
        stock_price = self.coordinator.get_data("stock_price")
        trade_suggestions = self.coordinator.get_data("trade_suggestions")
        filtered_trades = [trade for trade in trade_suggestions if trade['strike_price'] < stock_price * 1.1]
        self.coordinator.update_data("filtered_trades", filtered_trades)
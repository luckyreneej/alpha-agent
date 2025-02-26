from agent_communicator import AgentCoordinator
from data_agent import DataAgent
from risk_agent import RiskAgent
from sentiment_agent import SentimentLLMAgent
from prediction_agent import PredictionAgent
from trading_agent import TradingAgent
import logging


# Alpha Agent: Coordinates all agents
class AlphaAgent:
    def __init__(self, polygon_api_key, openai_api_key):
        self.coordinator = AgentCoordinator()
        self.data_agent = DataAgent(polygon_api_key, self.coordinator)
        self.sentiment_agent = SentimentLLMAgent(self.coordinator)
        # Initialize other agents as needed

    def run(self, ticker, expiration_date):
        self.data_agent.fetch_data(ticker, expiration_date)
        self.sentiment_agent.analyze_sentiment()
        # Execute other agents' methods as needed
        logging.info(f"Shared data: {self.coordinator.shared_data}")

if __name__ == "__main__":
    agent = AlphaAgent(POLYGON_API_KEY, OPENAI_API_KEY)
    agent.run("AAPL", "2025-03-21")

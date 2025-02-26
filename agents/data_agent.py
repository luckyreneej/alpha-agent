# Data Agent: Fetches stock and options data from Polygon.io
from base_agent import BaseAgent
import requests
import pandas as pd


class DataAgent(BaseAgent):
    def __init__(self, api_key, coordinator):
        super().__init__(coordinator)
        self.api_key = api_key
        self.base_url = "https://api.polygon.io/v2"

    def fetch_data(self, ticker, expiration_date):
        stock_price = self.get_stock_price(ticker)
        option_chain = self.get_option_chain(ticker, expiration_date)
        news_articles = self.get_news_articles(ticker)
        self.coordinator.update_data("stock_price", stock_price)
        self.coordinator.update_data("option_chain", option_chain)
        self.coordinator.update_data("news_articles", news_articles)

    def get_news_articles(self, ticker):
        url = f"{self.base_url}/reference/news"
        params = {
            "ticker": ticker,
            "order": "desc",
            "limit": 5,  # Number of articles to fetch
            "apiKey": self.api_key
        }
        response = requests.get(url, params=params)
        data = response.json()
        return data.get("results", [])

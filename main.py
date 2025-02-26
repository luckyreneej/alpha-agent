import requests
import datetime
import pandas as pd


class AlphaAgent:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.polygon.io/v3"

    def get_option_chain(self, underlying_symbol: str, expiration_date: str):
        url = f"https://api.polygon.io/v3/reference/options/contracts"
        params = {
            "underlying_ticker": underlying_symbol,
            "expiration_date": expiration_date,
            "apiKey": self.api_key
        }
        response = requests.get(url, params=params)
        data = response.json()
        return pd.DataFrame(data.get("results", []))

    def get_option_price(self, option_ticker: str):
        url = f"https://api.polygon.io/v2/aggs/ticker/{option_ticker}/prev"
        params = {"apiKey": self.api_key}
        response = requests.get(url, params=params)
        data = response.json()
        return data.get("results", [{}])[0].get("c", None)  # Closing price

    def suggest_trades(self, underlying_symbol: str, expiration_date: str, price_threshold: float):
        options_df = self.get_option_chain(underlying_symbol, expiration_date)
        options_df = options_df.sort_values(by=["strike_price"])  # Sort by strike price
        suggestions = []

        for _, row in options_df.iterrows():
            option_ticker = row["ticker"]
            strike_price = row["strike_price"]
            price = self.get_option_price(option_ticker)

            if price and price < price_threshold:
                suggestions.append({"ticker": option_ticker, "strike_price": strike_price, "premium": price})

        return suggestions


def main():
    API_KEY = "your_polygon_api_key_here"
    agent = AlphaAgent(API_KEY)
    suggestions = agent.suggest_trades("AAPL", "2025-03-21", price_threshold=5.0)

    print("Suggested Trades:")
    for suggestion in suggestions:
        print(
            f"Option: {suggestion['ticker']}, Strike Price: {suggestion['strike_price']}, Premium: {suggestion['premium']}")


if __name__ == "__main__":
    main()

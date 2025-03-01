import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
import json

from agents.base_agent import BaseAgent
from utils.communication.message import Message
from utils.data.polygon_api import PolygonAPI

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SentimentAgent(BaseAgent):
    """
    Enhanced Sentiment Analysis Agent:
    1. Fetches market news directly from Polygon.io Ticker News API
    2. Performs sophisticated sentiment analysis using OpenAI API
    3. Provides numerical sentiment scores and confidence levels
    4. Aggregates sentiment across multiple news items with weightings
    5. Integrates with the trading framework via direct communication
    """

    def __init__(self,
                 agent_id: str,
                 communicator=None,
                 api_key: Optional[str] = None,
                 polygon_api_key: Optional[str] = None,
                 model: str = "gpt-4",
                 news_limit: int = 10):
        """
        Initialize the SentimentAgent.

        Args:
            agent_id: Unique identifier for this agent
            communicator: Communication interface for inter-agent messaging
            api_key: OpenAI API key
            polygon_api_key: Polygon.io API key
            model: OpenAI model to use for sentiment analysis
            news_limit: Maximum number of news articles to fetch per ticker
        """
        super().__init__(agent_id, communicator)

        # Set OpenAI API key
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY')
        if not self.api_key:
            logger.warning("No OpenAI API key provided. Sentiment analysis will be unavailable.")

        # Initialize Polygon API client
        self.polygon_api = None
        polygon_key = polygon_api_key or os.environ.get('POLYGON_API_KEY')
        if polygon_key:
            try:
                self.polygon_api = PolygonAPI(polygon_key)
            except Exception as e:
                logger.error(f"Failed to initialize Polygon API: {e}")
        else:
            logger.warning("No Polygon API key provided. News data fetching will be unavailable.")

        # Configuration
        self.model = model
        self.news_limit = news_limit

        # Data storage
        self.ticker_sentiments = {}  # {ticker: sentiment_score}
        self.ticker_news_data = {}  # {ticker: [news_items]}
        self.sentiment_history = {}  # {ticker: {date: sentiment_score}}
        self.last_update = {}  # {ticker: last_update_timestamp}

        # Register request handlers
        self.register_request_handler('get_sentiment', self.handle_sentiment_request)

    async def process_message(self, message: Message) -> Optional[Message]:
        """
        Process incoming messages from other agents.

        Args:
            message: The message to process

        Returns:
            Optional response message
        """
        action = message.content.get('action')

        if action == 'analyze_sentiment':
            ticker = message.content.get('ticker')
            force_update = message.content.get('force_update', False)

            if not ticker:
                return Message(
                    sender_id=self.agent_id,
                    receiver_id=message.sender_id,
                    content={
                        'status': 'error',
                        'error': 'No ticker specified'
                    },
                    correlation_id=message.id
                )

            sentiment_result = await self.analyze_sentiment(ticker, force_update)

            return Message(
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                content=sentiment_result,
                correlation_id=message.id
            )

        # Delegate to parent class for other messages
        return await super().process_message(message)

    async def fetch_ticker_news(self, ticker: str, days_back: int = 7) -> List[Dict[str, Any]]:
        """
        Fetch news for a specific ticker from Polygon.io API.

        Args:
            ticker: Stock ticker symbol
            days_back: How many days back to fetch news for

        Returns:
            List of news items
        """
        if not self.polygon_api:
            logger.warning("Polygon API not initialized. Cannot fetch news.")
            return []

        try:
            # Calculate from_date as X days ago
            from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            to_date = datetime.now().strftime('%Y-%m-%d')

            # Fetch news articles
            news_df = self.polygon_api.get_market_news(
                ticker=ticker,
                limit=self.news_limit,
                from_date=from_date,
                to_date=to_date
            )

            # Convert to list of dictionaries
            if news_df.empty:
                logger.warning(f"No news found for ticker {ticker}")
                return []

            # Extract and process relevant fields
            news_items = []
            for _, row in news_df.iterrows():
                news_item = {
                    "title": row.get("title", ""),
                    "description": row.get("description", ""),
                    "published_utc": row.get("published_utc", ""),
                    "article_url": row.get("article_url", ""),
                    "tickers": row.get("tickers", []),
                    "keywords": row.get("keywords", [])
                }
                news_items.append(news_item)

            logger.info(f"Fetched {len(news_items)} news articles for {ticker}")
            return news_items

        except Exception as e:
            logger.error(f"Error fetching news for {ticker}: {e}")
            return []

    async def analyze_sentiment(self, ticker: str, force_update: bool = False) -> Dict[str, Any]:
        """
        Analyze sentiment for a specific ticker.

        Args:
            ticker: Stock ticker symbol
            force_update: Whether to force a fresh analysis

        Returns:
            Dictionary with sentiment analysis results
        """
        # Check if we have recent data (< 6 hours old) unless force_update
        current_time = datetime.now()
        if not force_update and ticker in self.last_update:
            time_diff = current_time - self.last_update[ticker]
            if time_diff.total_seconds() < 21600:  # 6 hours
                logger.info(f"Using cached sentiment data for {ticker}")
                return {
                    "ticker": ticker,
                    "sentiment_score": self.ticker_sentiments.get(ticker, 0),
                    "last_updated": self.last_update[ticker].isoformat(),
                    "count": len(self.ticker_news_data.get(ticker, []))
                }

        # Fetch fresh news data
        news_items = await self.fetch_ticker_news(ticker)
        if not news_items:
            logger.warning(f"No news articles available for {ticker}. Cannot perform sentiment analysis.")
            return {
                "ticker": ticker,
                "sentiment_score": 0,
                "sentiment_label": "neutral",
                "confidence": 0,
                "last_updated": current_time.isoformat(),
                "count": 0,
                "error": "No news data available"
            }

        # Store news data
        self.ticker_news_data[ticker] = news_items

        # Analyze sentiment for each news item
        analyzed_items = []
        for item in news_items:
            sentiment_result = await self._analyze_news_item(item)
            analyzed_items.append({**item, **sentiment_result})

        # Aggregate sentiment scores with weights based on recency and confidence
        if analyzed_items:
            # Calculate weighted average sentiment score
            total_weight = 0
            weighted_sum = 0

            for idx, item in enumerate(analyzed_items):
                # Recency factor - more recent news has higher weight
                recency_weight = 1.0 / (idx + 1)

                # Confidence weight - higher confidence has higher weight
                confidence_weight = item.get('confidence', 0.5)

                # Calculate combined weight
                combined_weight = recency_weight * confidence_weight

                weighted_sum += item['sentiment_score'] * combined_weight
                total_weight += combined_weight

            # Calculate final sentiment score
            if total_weight > 0:
                aggregate_score = round(weighted_sum / total_weight, 2)
            else:
                aggregate_score = 0

            # Convert score to label
            if aggregate_score > 0.2:
                sentiment_label = "positive"
            elif aggregate_score < -0.2:
                sentiment_label = "negative"
            else:
                sentiment_label = "neutral"

            # Update ticker sentiment data
            self.ticker_sentiments[ticker] = aggregate_score
            self.last_update[ticker] = current_time

            # Store in sentiment history
            date_key = current_time.strftime('%Y-%m-%d')
            if ticker not in self.sentiment_history:
                self.sentiment_history[ticker] = {}
            self.sentiment_history[ticker][date_key] = aggregate_score

            # Prepare result
            result = {
                "ticker": ticker,
                "sentiment_score": aggregate_score,
                "sentiment_label": sentiment_label,
                "last_updated": current_time.isoformat(),
                "count": len(analyzed_items)
            }

            logger.info(f"Sentiment analysis for {ticker}: {sentiment_label} ({aggregate_score})")
            return result
        else:
            logger.warning(f"Failed to analyze sentiment for {ticker}")
            return {
                "ticker": ticker,
                "sentiment_score": 0,
                "sentiment_label": "neutral",
                "confidence": 0,
                "last_updated": current_time.isoformat(),
                "count": 0,
                "error": "Sentiment analysis failed"
            }

    async def _analyze_news_item(self, news_item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze sentiment of a single news item using OpenAI API.

        Args:
            news_item: News item dictionary with title and description

        Returns:
            Dictionary with sentiment analysis results
        """
        try:
            title = news_item.get("title", "")
            description = news_item.get("description", "")

            # Create comprehensive prompt for sentiment analysis
            prompt = (
                f"You are a financial sentiment analyst. Analyze the sentiment of the following financial news.\n\n"
                f"Title: {title}\n"
                f"Description: {description}\n\n"
                f"Analyze the sentiment on a scale from -1.0 (extremely negative) to 1.0 (extremely positive), "
                f"where 0.0 is neutral. Provide a numeric score and confidence level (0.0-1.0).\n\n"
                f"Focus on implications for stock price movement, not general sentiment.\n"
                f"Return your response in this format: 'score: [numeric_value], confidence: [numeric_value], explanation: [brief reason]'"
            )

            # Call OpenAI API
            import openai
            openai.api_key = self.api_key

            response = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=[
                    {"role": "system",
                     "content": "You are a financial sentiment analyst specializing in stock market implications."},
                    {"role": "user", "content": prompt}
                ]
            )

            # Extract response text
            result_text = response["choices"][0]["message"]["content"].strip()

            # Parse the response
            score = 0
            confidence = 0.5
            explanation = ""

            if "score:" in result_text.lower():
                try:
                    score_text = result_text.lower().split("score:")[1].split(",")[0].strip()
                    score = float(score_text)
                except:
                    logger.warning(f"Failed to parse sentiment score: {result_text}")

            if "confidence:" in result_text.lower():
                try:
                    confidence_text = result_text.lower().split("confidence:")[1].split(",")[0].strip()
                    confidence = float(confidence_text)
                except:
                    logger.warning(f"Failed to parse confidence: {result_text}")

            if "explanation:" in result_text.lower():
                try:
                    explanation = result_text.split("explanation:")[1].strip()
                except:
                    explanation = result_text

            # Ensure score is in correct range
            score = max(min(score, 1.0), -1.0)
            confidence = max(min(confidence, 1.0), 0.0)

            # Map numeric score to sentiment label
            if score > 0.2:
                sentiment_label = "positive"
            elif score < -0.2:
                sentiment_label = "negative"
            else:
                sentiment_label = "neutral"

            return {
                "sentiment_score": score,
                "sentiment_label": sentiment_label,
                "confidence": confidence,
                "explanation": explanation
            }

        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return {
                "sentiment_score": 0,
                "sentiment_label": "neutral",
                "confidence": 0.5,
                "explanation": f"Error: {str(e)}"
            }

    async def analyze_multiple_tickers(self, tickers: List[str]) -> Dict[str, Any]:
        """
        Analyze sentiment for multiple tickers.

        Args:
            tickers: List of ticker symbols

        Returns:
            Dictionary with sentiment analysis results for each ticker
        """
        results = {}
        for ticker in tickers:
            results[ticker] = await self.analyze_sentiment(ticker)
        return results

    def get_ticker_sentiment_history(self, ticker: str, days: int = 30) -> Dict[str, float]:
        """
        Get historical sentiment for a ticker.

        Args:
            ticker: Stock ticker symbol
            days: Number of days of history to return

        Returns:
            Dictionary with dates and sentiment scores
        """
        if ticker not in self.sentiment_history:
            return {}

        history = self.sentiment_history[ticker]

        # Filter to only include the specified number of days
        cutoff_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        filtered_history = {k: v for k, v in history.items() if k >= cutoff_date}

        return filtered_history

    def get_market_sentiment(self, tickers: List[str] = None) -> float:
        """
        Get overall market sentiment based on a basket of tickers.

        Args:
            tickers: List of ticker symbols to include, or None for default basket

        Returns:
            Aggregate sentiment score for the market
        """
        if not tickers:
            # Default to major market indices and tech giants
            tickers = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'AMZN', 'GOOGL']

        scores = []
        for ticker in tickers:
            if ticker in self.ticker_sentiments:
                scores.append(self.ticker_sentiments[ticker])

        if not scores:
            return 0

        return sum(scores) / len(scores)

    async def handle_sentiment_request(self, sender_id: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle requests for sentiment data from other agents.

        Args:
            sender_id: ID of the requesting agent
            request_data: Request parameters

        Returns:
            Dictionary with sentiment data
        """
        ticker = request_data.get('ticker')
        tickers = request_data.get('tickers', [])

        if ticker:
            # Single ticker request
            if ticker in self.ticker_sentiments:
                return {
                    "ticker": ticker,
                    "sentiment_score": self.ticker_sentiments[ticker],
                    "last_updated": self.last_update.get(ticker,
                                                         datetime.now()).isoformat() if ticker in self.last_update else None
                }
            else:
                # If we don't have sentiment data for this ticker, try to get it
                try:
                    result = await self.analyze_sentiment(ticker)
                    return result
                except Exception as e:
                    logger.error(f"Error analyzing sentiment for {ticker}: {e}")
                    return {"ticker": ticker, "error": "Failed to analyze sentiment"}

        elif tickers:
            # Multiple tickers request
            results = {}
            for t in tickers:
                if t in self.ticker_sentiments:
                    results[t] = {
                        "sentiment_score": self.ticker_sentiments[t],
                        "last_updated": self.last_update.get(t,
                                                             datetime.now()).isoformat() if t in self.last_update else None
                    }
                else:
                    # Try to get sentiment for tickers we don't have data for
                    try:
                        result = await self.analyze_sentiment(t)
                        results[t] = {
                            "sentiment_score": result["sentiment_score"],
                            "last_updated": result["last_updated"]
                        }
                    except Exception as e:
                        logger.error(f"Error analyzing sentiment for {t}: {e}")
                        results[t] = {"error": "Failed to analyze sentiment"}
            return {"tickers": results}

        else:
            # Market sentiment request
            market_sentiment = self.get_market_sentiment()
            return {"market_sentiment": market_sentiment}

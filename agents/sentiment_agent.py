# Sentiment LLM Agent: Uses OpenAI API to analyze market sentiment
from agents.base_agent import BaseAgent


class SentimentLLMAgent(BaseAgent):
    def analyze_sentiment(self):
        news_articles = self.coordinator.get_data("news_articles")
        if not news_articles:
            logging.warning("No news articles available for sentiment analysis.")
            return

        sentiments = []
        for article in news_articles:
            title = article.get("title", "")
            description = article.get("description", "")
            content = f"Title: {title}\nDescription: {description}"
            prompt = f"Analyze the sentiment of the following news article:\n\n{content}\n\nSentiment (positive, neutral, negative):"

            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a financial sentiment analyst."},
                    {"role": "user", "content": prompt}
                ]
            )
            sentiment = response["choices"][0]["message"]["content"].strip()
            sentiments.append({"article": content, "sentiment": sentiment})

        self.coordinator.update_data("news_sentiments", sentiments)

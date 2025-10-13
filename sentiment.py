import numpy as np
from textblob import TextBlob
import requests
import praw
import os

# Reddit API setup
reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID", ""),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET", ""),
    user_agent=os.getenv("REDDIT_USER_AGENT", "StockSentimentBot")
)

def get_reddit_sentiment(ticker, max_posts=50):
    scores = []
    karma_total = 0
    try:
        for submission in reddit.subreddit("stocks").search(ticker, limit=max_posts):
            title = submission.title or ""
            karma = submission.score or 0
            sent = TextBlob(title).sentiment.polarity
            scores.append(sent * max(1, karma))
            karma_total += karma
        avg_sent = np.sum(scores) / karma_total if karma_total > 0 else 0.0
        return avg_sent, karma_total
    except Exception:
        return 0.0, 0

def get_news_sentiment(ticker, api_key, max_headlines=5):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": ticker,
        "pageSize": max_headlines,
        "sortBy": "publishedAt",
        "apiKey": api_key,
        "language": "en"
    }
    try:
        r = requests.get(url, params=params)
        articles = r.json().get("articles", [])
        scores = [TextBlob(a["title"]).sentiment.polarity for a in articles if a.get("title")]
        return float(np.mean(scores)) if scores else 0.0
    except Exception:
        return 0.0
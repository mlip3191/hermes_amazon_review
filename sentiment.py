#!/usr/bin/env python3
"""
Three-tier sentiment classification utilities.

Sentiment mapping:
    1-2 stars -> NEGATIVE
    3 stars  -> NEUTRAL
    4-5 stars -> POSITIVE
"""

SENTIMENT_MAP = {
    0: "negative",
    1: "neutral",
    2: "positive",
}

REVERSE_SENTIMENT = {v: k for k, v in SENTIMENT_MAP.items()}


def star_to_sentiment(star_rating: int) -> str:
    """Convert star rating (1-5) to sentiment category."""
    if star_rating <= 2:
        return "negative"
    elif star_rating == 3:
        return "neutral"
    else:  # 4-5
        return "positive"


def star_to_sentiment_idx(star_rating: int) -> int:
    """Convert star rating (1-5) to sentiment index (0=negative, 1=neutral, 2=positive)."""
    if star_rating <= 2:
        return 0
    elif star_rating == 3:
        return 1
    else:  # 4-5
        return 2


def idx_to_sentiment(idx: int) -> str:
    """Convert sentiment index to sentiment category."""
    return SENTIMENT_MAP.get(int(idx), "unknown")


def sentiment_to_idx(sentiment: str) -> int:
    """Convert sentiment category to index."""
    return REVERSE_SENTIMENT.get(sentiment.lower(), -1)

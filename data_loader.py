#!/usr/bin/env python3
"""
Data loading utilities with random sampling support.
"""
import json
import random
from typing import List, Tuple


def load_all_reviews(path: str) -> List[Tuple[str, int]]:
    """Load all reviews from JSONL file. Return list of (text, rating) tuples."""
    reviews = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rating = rec.get("rating")
            if rating is None:
                continue
            title = (rec.get("title") or "").strip()
            text = (rec.get("text") or "").strip()
            content = f"{title}. {text}".strip()
            if not content:
                continue
            reviews.append((content, rating))
    return reviews


def load_random_sample(
    path: str, sample_size: int, seed: int = 42
) -> List[Tuple[str, int]]:
    """Load all reviews and return a random sample of specified size.

    Args:
        path: Path to JSONL reviews file
        sample_size: Number of reviews to sample
        seed: Random seed for reproducibility (default 42)

    Returns:
        List of (text, rating) tuples
    """
    all_reviews = load_all_reviews(path)
    if sample_size >= len(all_reviews):
        return all_reviews

    random.seed(seed)
    return random.sample(all_reviews, sample_size)

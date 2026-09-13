#!/usr/bin/env python3
"""
NRC Word-Emotion Association Lexicon (EmoLex) scorer.

No model calls: tokenizes review text, looks each word up in the NRC
lexicon, sums per-emotion hit counts, and returns the argmax emotion.

Standalone usage (no API key needed, works over an existing report_data.json):
    ./.venv/bin/python emotion_lexicon.py [report_data.json]
"""
import json
import re
import sys
from pathlib import Path

EMOTIONS = ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"]
DEFAULT_LEXICON_PATH = Path(__file__).resolve().parent / "data" / "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"
DEFAULT_DATA = "report_data.json"

_WORD_RE = re.compile(r"[a-z']+")
_lexicon_cache = None


def load_lexicon(path: Path = DEFAULT_LEXICON_PATH) -> dict:
    """Return {emotion: set(words)} for the 8 target emotions.

    Skips the lexicon's "positive"/"negative" sentiment rows (not part of
    the 8-way emotion taxonomy) and rows with association == 0.
    """
    emotion_set = set(EMOTIONS)
    lexicon = {e: set() for e in EMOTIONS}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            word, emotion, assoc = parts
            if emotion not in emotion_set or assoc != "1":
                continue
            lexicon[emotion].add(word)
    return lexicon


def get_lexicon() -> dict:
    """Lazily load and cache the lexicon so repeated calls don't reparse the file."""
    global _lexicon_cache
    if _lexicon_cache is None:
        _lexicon_cache = load_lexicon()
    return _lexicon_cache


def tokenize(text: str) -> list:
    return _WORD_RE.findall(text.lower())


def score_text(text: str, lexicon: dict = None):
    """Return (per_emotion_counts, top_emotion). top_emotion is None if no
    word in the text hit any emotion. Ties broken deterministically by
    EMOTIONS order."""
    lexicon = lexicon if lexicon is not None else get_lexicon()
    counts = {e: 0 for e in EMOTIONS}
    for word in tokenize(text):
        for emotion in EMOTIONS:
            if word in lexicon[emotion]:
                counts[emotion] += 1
    top = max(EMOTIONS, key=lambda e: counts[e]) if any(counts.values()) else None
    return counts, top


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    lexicon = get_lexicon()
    for row in data["rows"]:
        counts, top = score_text(row["text"], lexicon)
        row["lexicon_counts"] = counts
        row["lexicon_emotion"] = top

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Scored {len(data['rows'])} rows in {path} with lexicon_emotion (no model calls).")


if __name__ == "__main__":
    main()

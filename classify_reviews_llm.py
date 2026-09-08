#!/usr/bin/env python3
"""
Binary sentiment classification of Amazon Gift Card reviews using an LLM
served at an OpenAI-compatible endpoint (defaults to Anthropic's Claude via
their OpenAI SDK compatibility layer).

Labels (truth) are derived from the star rating: 1-3 = NEGATIVE, 4-5 = POSITIVE.
The rating is never sent to the model and never used as an input feature —
only the review title + text.

Configuration (env vars):
    ANTHROPIC_API_KEY  (or OPENAI_API_KEY)  required to authenticate
    OPENAI_BASE_URL                         default https://api.anthropic.com/v1/
        (override with any OpenAI-compatible server, e.g. a vLLM/Ollama proxy)

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    ./.venv/bin/python classify_reviews_llm.py --sample 30 [--model claude-haiku-4-5-20251001]
"""
import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

from openai import OpenAI

DEFAULT_DATA = "Gift_Cards.jsonl"
DEFAULT_BASE = "https://api.anthropic.com/v1/"
DEFAULT_MODEL = "claude-haiku-4-5-20251001"


def load_dotenv(path: Path):
    """Minimal .env loader: KEY=VALUE lines into os.environ (no override)."""
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val


def load_reviews(path: str):
    """Yield (text, label) pairs. Label from rating; text from title+text."""
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
            label = "NEGATIVE" if rating <= 3 else "POSITIVE"
            yield content, label


SYSTEM_PROMPT = (
    "You are a sentiment classifier for Amazon product reviews. You will be "
    "shown the title and text of one review. Decide whether the reviewer felt "
    "POSITIVE or NEGATIVE about the product. "
    "Reply with exactly one token: POSITIVE or NEGATIVE. No explanation."
)


def classify(client, model, text: str, retries: int = 3) -> Optional[str]:
    """Return 'POSITIVE'/'NEGATIVE' for a review, or None if it fails."""
    last_err = None
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0,
                max_tokens=8,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text[:2000]},
                ],
            )
            raw = resp.choices[0].message.content.strip()
            m = re.search(r"(POSITIVE|NEGATIVE)", raw.upper())
            if m:
                return m.group(1)
            last_err = f"unparseable reply: {raw!r}"
        except Exception as e:  # noqa: BLE001 - surface any API error
            last_err = str(e)
            time.sleep(min(2 ** attempt, 8))
    print(f"  [warn] classified failed for review: {last_err}", file=sys.stderr)
    return None


def main():
    parser = argparse.ArgumentParser(description="LLM-based pos/neg review classifier via OpenAI-compatible endpoint.")
    parser.add_argument("data", nargs="?", default=DEFAULT_DATA)
    parser.add_argument("--sample", type=int, default=30, help="Number of reviews to classify.")
    parser.add_argument("--model", default=None, help="Model id (default claude-haiku-4-5-20251001).")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL (default Anthropic).")
    parser.add_argument("--api-key", default=None, help="API key (default $ANTHROPIC_API_KEY / $OPENAI_API_KEY).")
    parser.add_argument("--out", default="predictions.csv", help="CSV output path.")
    args = parser.parse_args()

    # Load .env from the same directory as this script (key lives here, never in git).
    load_dotenv(Path(__file__).resolve().parent / ".env")

    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit(
            "ERROR: no API key found. Set ANTHROPIC_API_KEY (or OPENAI_API_KEY), "
            "or pass --api-key."
        )

    base_url = args.base_url or os.environ.get("OPENAI_BASE_URL") or DEFAULT_BASE
    model = args.model or DEFAULT_MODEL
    client = OpenAI(api_key=api_key, base_url=base_url)
    print(f"Endpoint : {base_url}")
    print(f"Model    : {model}")

    # Load the first `sample` usable reviews (title/text non-empty).
    reviews = []
    for content, label in load_reviews(args.data):
        reviews.append((content, label))
        if len(reviews) >= args.sample:
            break
    print(f"Loaded {len(reviews):,} reviews from {args.data}\n")

    t0 = time.time()
    rows, correct = [], 0
    for i, (content, truth) in enumerate(reviews, 1):
        pred = classify(client, model, content)
        if pred is None:
            rows.append([truth, "", content])
            continue
        ok = pred == truth
        correct += ok
        print(f"[{i:<3}/{len(reviews)}] {'OK ' if ok else 'ERR'} pred={pred:<8} true={truth:<8} | {content[:70]}")
        rows.append([truth, pred, content])

    # Results
    n = len(reviews)
    acc = correct / n if n else 0.0
    print("\n===== RESULTS =====")
    print(f"Reviewed : {n}")
    print(f"Correct  : {correct}")
    print(f"Accuracy : {acc:.1%}")

    tp = sum(1 for t, p, _ in rows if t == "POSITIVE" and p == "POSITIVE")
    fp = sum(1 for t, p, _ in rows if t == "NEGATIVE" and p == "POSITIVE")
    fn = sum(1 for t, p, _ in rows if t == "POSITIVE" and p == "NEGATIVE")
    tn = sum(1 for t, p, _ in rows if t == "NEGATIVE" and p == "NEGATIVE")
    print("\nConfusion matrix [[TN FP], [FN TP]]:")
    print(f"  [[{tn} {fp}], [{fn} {tp}]]")
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    if not (prec != prec or rec != rec):
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        print(f"Positive-class: precision={prec:.3f} recall={rec:.3f} F1={f1:.3f}")
    print(f"Elapsed : {time.time()-t0:.1f}s")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["true_label", "predicted_label", "review_text"])
        wr.writerows(rows)
    print(f"\nWrote predictions to {args.out}")


if __name__ == "__main__":
    sys.exit(main())

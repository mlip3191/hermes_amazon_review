#!/usr/bin/env python3
"""
Star-rating (1-5) prediction for Amazon Gift Card reviews using an LLM
served at an OpenAI-compatible endpoint (defaults to Anthropic's Claude via
their OpenAI SDK compatibility layer).

IMPORTANT CONSTRAINTS
---------------------
* SCORING NEVER SEES THE RATING. The true star rating is used ONLY as
  ground truth to measure accuracy AFTER scoring, in memory. It is never
  sent to the model as input and never written to the output file.
* The output CSV saves ONLY each review's PREDICTED score and the review
  text -- the actual ratings are never persisted.

Model is asked to return a single integer 1-5 based only on title + text.

Configuration (env vars):
    ANTHROPIC_API_KEY  (or OPENAI_API_KEY)  required to authenticate
    OPENAI_BASE_URL                         default https://api.anthropic.com/v1/

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    ./.venv/bin/python classify_reviews_llm.py --sample 50
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
    """Yield (text, true_rating) pairs. Rating is ground truth ONLY.

    The review text is title + text; the rating is yielded separately so the
    caller can pass it only to the post-hoc accuracy check, never to the model.
    """
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
            yield content, rating


SYSTEM_PROMPT = (
    "You are a review scorer for an e-commerce site. You will be shown the "
    "title and text of one product review. Based ONLY on that text, estimate "
    "the star rating (from 1 to 5) the reviewer most likely gave, where "
    "1 = worst and 5 = best. Reply with exactly one integer digit 1-5 and "
    "nothing else. No explanation."
)


def score_review(client, model, text: str, retries: int = 3) -> Optional[int]:
    """Return a predicted star rating 1-5, or None if the call fails."""
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
            m = re.search(r"[1-5]", raw)  # first digit in 1-5
            if m:
                return int(m.group(0))
            last_err = f"unparseable reply: {raw!r}"
        except Exception as e:  # noqa: BLE001 - surface any API error
            last_err = str(e)
            time.sleep(min(2 ** attempt, 8))
    print(f"  [warn] scoring failed for review: {last_err}", file=sys.stderr)
    return None


def main():
    parser = argparse.ArgumentParser(description="Predict 1-5 star rating from review title+text via an LLM.")
    parser.add_argument("data", nargs="?", default=DEFAULT_DATA)
    parser.add_argument("--sample", type=int, default=30, help="Number of reviews to score.")
    parser.add_argument("--model", default=None, help="Model id (default claude-haiku-4-5-20251001).")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL (default Anthropic).")
    parser.add_argument("--api-key", default=None, help="API key (default $ANTHROPIC_API_KEY / $OPENAI_API_KEY).")
    parser.add_argument("--out", default="score_predictions.csv", help="CSV output path (predictions only).")
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

    # Load first `sample` reviews. Keep text and true_rating SEPARATE: the
    # rating is only for the post-hoc accuracy check, never for the model.
    reviews = []
    for content, rating in load_reviews(args.data):
        reviews.append((content, rating))
        if len(reviews) >= args.sample:
            break
    print(f"Loaded {len(reviews):,} reviews from {args.data}\n")

    t0 = time.time()
    rows, errors = [], 0
    for i, (content, _true_rating) in enumerate(reviews, 1):
        # NOTE: _true_rating is intentionally ignored here (never sent to model).
        pred = score_review(client, model, content)
        if pred is None:
            errors += 1
            rows.append(["", content])
            continue
        print(f"[{i:<3}/{len(reviews)}] predicted={pred} | {content[:70]}")
        rows.append([pred, content])

    print(f"\nScored {len(rows)-errors} of {len(reviews)} reviews ({time.time()-t0:.1f}s)")

    # ---- Post-hoc accuracy against the rating (in memory only) ----
    n_rated, exact, within1 = 0, 0, 0
    errs = []
    for (content, true_rating), (pred_score, _txt) in zip(reviews, rows):
        if pred_score == "":
            continue  # failed call: no prediction to compare
        n_rated += 1
        diff = abs(int(pred_score) - true_rating)
        exact += 1 if diff == 0 else 0
        within1 += 1 if diff <= 1 else 0
        errs.append(diff)
    maes = sum(errs) / len(errs) if errs else 0.0

    print("\n===== SCORE ACCURACY (vs rating, computed in-memory, NOT saved) =====")
    print(f"Rated     : {n_rated:,}")
    print(f"Exact match (pred == rating)      : {exact}  ({exact/n_rated:.1%})" if n_rated else "Exact: N/A")
    print(f"Within +/-1 star                  : {within1}  ({within1/n_rated:.1%})" if n_rated else "Within1: N/A")
    print(f"Mean absolute error              : {maes:.3f} stars")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["predicted_score", "review_text"])
        wr.writerows(rows)
    print(f"\nSaved PREDICTIONS ONLY to {args.out} (true ratings never written)")


if __name__ == "__main__":
    sys.exit(main())

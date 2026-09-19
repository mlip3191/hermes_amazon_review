#!/usr/bin/env python3
"""
Phase 4 data generator: for each review, get the model's PREDICTED star (1-5),
a primary EMOTION, and a one-line REASON, from title+text only. Also scores
the same text with the no-model-calls NRC lexicon (emotion_lexicon.py) so the
two independent emotion takes can be compared. Pairs with the true rating
post-hoc (in memory) to compute accuracy. Emits report_data.json for the HTML
report. True ratings are used only here, never sent to the model.
"""
import json
import os
import re
import sys
import time
from pathlib import Path

from openai import OpenAI

from classify_reviews_llm import DEFAULT_BASE, load_dotenv
from data_loader import load_random_sample
from emotion_lexicon import EMOTIONS, get_lexicon, score_text
from sentiment import star_to_sentiment

DEFAULT_MODEL = "claude-haiku-4-5-20251001"
OUT = "report_data.json"

EMOTION_LIST = ", ".join(EMOTIONS)
SYSTEM_PROMPT = (
    "You are a review scorer for an e-commerce site. You will be shown the "
    "title and text of one product review. Based ONLY on that text, estimate "
    "the star rating (1 to 5) the reviewer most likely gave (1 = worst, "
    "5 = best), and the reviewer's single primary emotion, which must be "
    f"exactly one of: {EMOTION_LIST}. Reply with exactly three fields "
    'separated by "|" and nothing else: an integer 1-5, then the emotion '
    "word, then a reason under ~15 words. For example, if a review is a "
    "disappointed 2-star complaint about slow shipping, a correct reply "
    "looks like: 2|disgust|Reviewer is unhappy about slow shipping."
)


def score_with_emotion(client, model, text, retries=3):
    """Return (star:int|None, emotion:str|None, reason:str)."""
    emotion_set = set(EMOTIONS)
    last = None
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model, temperature=0, max_tokens=55,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text[:2000]},
                ],
            )
            raw = resp.choices[0].message.content.strip()
            star = re.search(r"[1-5]", raw)
            if not star:
                last = f"unparseable: {raw!r}"
                continue
            parts = raw.split("|")
            emotion = parts[1].strip().lower() if len(parts) >= 2 else ""
            if emotion not in emotion_set:
                emotion = None
            reason = parts[2].strip() if len(parts) >= 3 else ""
            return int(star.group(0)), emotion, reason
        except Exception as e:  # noqa: BLE001
            last = str(e)
            time.sleep(min(2 ** attempt, 8))
    print(f"  [warn] failed: {last}", file=sys.stderr)
    return None, None, ""


def main():
    sample = int(sys.argv[1]) if len(sys.argv) > 1 else 250
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    model = os.environ.get("SCORE_MODEL", DEFAULT_MODEL)
    load_dotenv(Path(__file__).resolve().parent / ".env")
    api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("no API key")
    client = OpenAI(api_key=api_key, base_url=DEFAULT_BASE)

    reviews = load_random_sample("Gift_Cards.jsonl", sample, seed=seed)
    print(f"Loaded {len(reviews):,} random reviews (seed={seed})\n")

    lexicon = get_lexicon()

    rows, exact, within, errs = [], 0, 0, []
    t0 = time.time()
    for i, (text, true) in enumerate(reviews, 1):
        pred, llm_emotion, reason = score_with_emotion(client, model, text)
        _, lexicon_emotion = score_text(text, lexicon)
        if pred is None:
            rows.append({
                "text": text, "true": true, "pred": None, "reason": "", "correct": None,
                "true_sentiment": star_to_sentiment(true), "pred_sentiment": None,
                "llm_emotion": llm_emotion, "lexicon_emotion": lexicon_emotion, "emotion_agree": None,
            })
            continue
        diff = abs(pred - true)
        exact += diff == 0
        within += diff <= 1
        errs.append(diff)
        emotion_agree = (llm_emotion == lexicon_emotion) if (llm_emotion and lexicon_emotion) else None
        rows.append({
            "text": text, "title": text.split(".",1)[0][:60], "true": int(true),
            "pred": pred, "reason": reason, "correct": pred == true, "diff": diff,
            "true_sentiment": star_to_sentiment(true), "pred_sentiment": star_to_sentiment(pred),
            "llm_emotion": llm_emotion, "lexicon_emotion": lexicon_emotion, "emotion_agree": emotion_agree,
        })
        print(f"[{i}/{len(reviews)}] pred={pred} true={true} ok={pred==true} "
              f"emotion={llm_emotion}/{lexicon_emotion} | {text[:50]}")
    n = len([r for r in rows if r["pred"] is not None])
    stats = {
        "n": n, "exact": exact, "within1": within,
        "pct_exact": round(100*exact/n, 1) if n else 0,
        "pct_within1": round(100*within/n, 1) if n else 0,
        "mae": round(sum(errs)/len(errs), 3) if errs else 0,
    }

    valid = [r for r in rows if r.get("llm_emotion") and r.get("lexicon_emotion")]
    agree = sum(1 for r in valid if r["llm_emotion"] == r["lexicon_emotion"])
    emotion_stats = {
        "n": len(valid), "agree": agree,
        "pct_agree": round(100*agree/len(valid), 1) if valid else 0,
        "llm_distribution": {e: sum(1 for r in valid if r["llm_emotion"] == e) for e in EMOTIONS},
        "lexicon_distribution": {e: sum(1 for r in valid if r["lexicon_emotion"] == e) for e in EMOTIONS},
        "confusion": {
            e1: {e2: sum(1 for r in valid if r["llm_emotion"] == e1 and r["lexicon_emotion"] == e2)
                 for e2 in EMOTIONS}
            for e1 in EMOTIONS
        },
    }

    valid_stars = [r for r in rows if r["pred"] is not None]
    true_dist = {c: sum(1 for r in valid_stars if r["true"] == c) for c in range(1, 6)}
    pred_dist = {c: sum(1 for r in valid_stars if r["pred"] == c) for c in range(1, 6)}
    confusion_stars = {
        t: {p: sum(1 for r in valid_stars if r["true"] == t and r["pred"] == p) for p in range(1, 6)}
        for t in range(1, 6)
    }
    per_class = {
        c: {
            "n": true_dist[c],
            "correct": confusion_stars[c][c],
            "pct_correct": round(100 * confusion_stars[c][c] / true_dist[c], 1) if true_dist[c] else 0,
        }
        for c in range(1, 6)
    }
    rating_stats = {
        "n": len(valid_stars), "true_distribution": true_dist, "pred_distribution": pred_dist,
        "confusion": confusion_stars, "per_class": per_class,
    }

    # Three-tier sentiment stats
    sentiment_labels = ["negative", "neutral", "positive"]
    valid_sentiment = [r for r in rows if r.get("pred_sentiment") is not None]
    sentiment_correct = sum(1 for r in valid_sentiment if r["true_sentiment"] == r["pred_sentiment"])
    sentiment_true_dist = {s: sum(1 for r in valid_sentiment if r["true_sentiment"] == s) for s in sentiment_labels}
    sentiment_pred_dist = {s: sum(1 for r in valid_sentiment if r["pred_sentiment"] == s) for s in sentiment_labels}
    sentiment_confusion = {
        t: {p: sum(1 for r in valid_sentiment if r["true_sentiment"] == t and r["pred_sentiment"] == p)
            for p in sentiment_labels}
        for t in sentiment_labels
    }
    sentiment_per_class = {
        s: {
            "n": sentiment_true_dist[s],
            "correct": sentiment_confusion[s][s],
            "pct_correct": round(100 * sentiment_confusion[s][s] / sentiment_true_dist[s], 1) if sentiment_true_dist[s] else 0,
        }
        for s in sentiment_labels
    }
    sentiment_stats = {
        "n": len(valid_sentiment), "correct": sentiment_correct,
        "pct_correct": round(100 * sentiment_correct / len(valid_sentiment), 1) if valid_sentiment else 0,
        "true_distribution": sentiment_true_dist, "pred_distribution": sentiment_pred_dist,
        "confusion": sentiment_confusion, "per_class": sentiment_per_class,
    }

    json.dump({"stats": stats, "emotion_stats": emotion_stats, "rating_stats": rating_stats, "sentiment_stats": sentiment_stats, "rows": rows}, open(OUT, "w"), indent=2)
    print(f"\nWrote {OUT}: {json.dumps(stats)}  ({time.time()-t0:.0f}s)")
    print(f"Emotion agreement: {emotion_stats['pct_agree']}% ({agree}/{len(valid)})")


if __name__ == "__main__":
    main()

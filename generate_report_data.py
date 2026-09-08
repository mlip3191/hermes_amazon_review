#!/usr/bin/env python3
"""
Phase 3 data generator: for each review, get the model's PREDICTED star (1-5)
plus a one-line REASON, from title+text only. Pairs with the true rating
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

from classify_reviews_llm import DEFAULT_BASE, load_dotenv, load_reviews

DEFAULT_MODEL = "claude-haiku-4-5-20251001"
OUT = "report_data.json"

SYSTEM_PROMPT = (
    "You are a review scorer for an e-commerce site. You will be shown the "
    "title and text of one product review. Based ONLY on that text, estimate "
    "the star rating (1 to 5) the reviewer most likely gave (1 = worst, "
    "5 = best). Reply strictly in this format:\n"
    "PREDICTED_STAR|one_sentence_reason\n"
    "PREDICTED_STAR is a single integer 1-5. Keep the reason under ~15 words. "
    "No other text."
)


def score_with_reason(client, model, text, retries=3):
    last = None
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model, temperature=0, max_tokens=40,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text[:2000]},
                ],
            )
            raw = resp.choices[0].message.content.strip()
            star = re.search(r"[1-5]", raw)
            if star and "|" in raw:
                reason = raw.split("|", 1)[1].strip()
                return int(star.group(0)), reason
            if star:  # tolerate missing pipe
                return int(star.group(0)), reason if (reason := raw.split("|",1)[-1].strip()) else ""
            last = f"unparseable: {raw!r}"
        except Exception as e:  # noqa: BLE001
            last = str(e)
            time.sleep(min(2 ** attempt, 8))
    print(f"  [warn] failed: {last}", file=sys.stderr)
    return None, ""


def main():
    sample = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    model = os.environ.get("SCORE_MODEL", DEFAULT_MODEL)
    load_dotenv(Path(__file__).resolve().parent / ".env")
    api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("no API key")
    client = OpenAI(api_key=api_key, base_url=DEFAULT_BASE)

    reviews = []
    for text, rating in load_reviews("Gift_Cards.jsonl"):
        reviews.append((text, rating))
        if len(reviews) >= sample:
            break

    rows, exact, within, errs = [], 0, 0, []
    t0 = time.time()
    for i, (text, true) in enumerate(reviews, 1):
        pred, reason = score_with_reason(client, model, text)
        if pred is None:
            rows.append({"text": text, "true": true, "pred": None, "reason": "", "correct": None})
            continue
        diff = abs(pred - true)
        exact += diff == 0
        within += diff <= 1
        errs.append(diff)
        rows.append({
            "text": text, "title": text.split(".",1)[0][:60], "true": int(true),
            "pred": pred, "reason": reason, "correct": pred == true, "diff": diff,
        })
        print(f"[{i}/{len(reviews)}] pred={pred} true={true} ok={pred==true} | {text[:50]}")
    n = len([r for r in rows if r["pred"] is not None])
    stats = {
        "n": n, "exact": exact, "within1": within,
        "pct_exact": round(100*exact/n, 1) if n else 0,
        "pct_within1": round(100*within/n, 1) if n else 0,
        "mae": round(sum(errs)/len(errs), 3) if errs else 0,
    }
    json.dump({"stats": stats, "rows": rows}, open(OUT, "w"), indent=2)
    print(f"\nWrote {OUT}: {json.dumps(stats)}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()

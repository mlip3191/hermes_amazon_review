# hermes_amazon_review

Agent that predicts Amazon Gift Card review scores (1–5 stars) and primary emotions from title and text.
![[Pasted image 20260913210003.png]]

## Phases

### Phase 1: Binary sentiment classification
`classify_reviews.py` — TF-IDF + Logistic Regression on review text vs. star rating (1–3 → negative, 4–5 → positive).

### Phase 2: Star rating prediction (LLM)
`classify_reviews_llm.py` — Claude (via OpenAI-compatible endpoint) predicts exact star rating 1–5 from title + text.

### Phase 3: Report with reasoning
`generate_report_data.py` + `build_report.py` — LLM predicts star + one-line reasoning. Report includes failure-mode analysis ("Complaint dominates", "Hedged phrasing", etc.) grouped from misclassified rows.

### Phase 4: Dual primary-emotion detection
Two independent takes on primary emotion (anger, anticipation, disgust, fear, joy, sadness, surprise, trust):
- **LLM take** — extended prompt in `generate_report_data.py` returns `STAR|EMOTION|reason` from same API call
- **Lexicon take** — `emotion_lexicon.py` scores using NRC Word-Emotion Association Lexicon (no model calls, no API key needed)

Reports agreement % and confusion matrix; can rescore existing `report_data.json` via standalone `emotion_lexicon.py` CLI.

## Running

Requires `.venv` with `openai`, `numpy`, `pandas`, `scikit-learn` installed; API key in `.env` (ANTHROPIC_API_KEY or OPENAI_API_KEY).

```bash
./.venv/bin/python generate_report_data.py [sample_size]  # default 24; phase 4: generates stars + emotions
./.venv/bin/python build_report.py                         # builds self-contained report.html
./.venv/bin/python emotion_lexicon.py report_data.json    # rescores existing data with lexicon (no API needed)
```

## Output

`report.html` — single-file self-contained report with client-side rendering (no external assets). Includes:
- Overview grid: emotion agreement %, star accuracy, rating distribution
- Detailed tables with filters (by emotion, result, actual/predicted stars)
- Failure mode classification for wrong predictions

`report_data.json` — raw predictions: `{stats, emotion_stats, rows}`

## Recent Changes

### v0.4.1 — UI/UX Polish & Bug Fixes
- **Dynamic tabbed tables** (a14a5c2) — Single "View details" section with tabs to switch between emotion, star ratings, and failure classes tables (instead of three separate sections)
- **Reordered layout** (38c11e5) — Headline numbers → Primary emotion → Predicted rating as vertical stack, focused flow
- **Bar chart label alignment** (f7c941f) — Fixed count labels at top of bars overlapping with section titles; all labels now on same baseline
- **LLM prompt fix** (f7c941f) — Replaced literal placeholder tokens with worked example; fixed parse failures on ambiguous reviews (5/150 → 0/150)
- **Collapsible tables** (e697232) — Tables expand/collapse via buttons instead of always visible

### v0.4.0 — Phase 4: Dual Primary-Emotion Detection
- **LLM emotion scoring** (1c6a7b1) — Extended `generate_report_data.py` prompt to return `STAR|EMOTION|reason` in single API call
- **NRC lexicon scorer** (1c6a7b1) — Standalone `emotion_lexicon.py` (no API calls, no key needed) scores reviews against Word-Emotion Association Lexicon; can rescore existing `report_data.json`
- **Emotion comparison** (1c6a7b1) — Report includes LLM vs. lexicon emotion agreement %, confusion matrix, and side-by-side distributions

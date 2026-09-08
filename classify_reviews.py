#!/usr/bin/env python3
"""
Binary sentiment classification of Amazon Gift Card reviews.

Labels are derived ONLY from the star rating:
    rating 1, 2, 3  -> NEGATIVE (0)
    rating 4, 5     -> POSITIVE (1)

Features are built ONLY from the review title + text (the rating is NOT
used as an input feature). A TF-IDF + Logistic Regression pipeline is
trained and evaluated on a stratified train/test split.

Usage:
    ./.venv/bin/python classify_reviews.py [path_to_jsonl]
"""
import argparse
import json
import sys
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

DEFAULT_DATA = "Gift_Cards.jsonl"


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
            # Features: title + text only. Never the rating itself.
            title = (rec.get("title") or "").strip()
            text = (rec.get("text") or "").strip()
            content = f"{title}. {text}".strip()
            if not content:
                continue
            label = 0 if rating <= 3 else 1  # 1-3 negative, 4-5 positive
            yield content, label


def main():
    parser = argparse.ArgumentParser(description="Classify Amazon reviews as pos/neg from title+text.")
    parser.add_argument("data", nargs="?", default=DEFAULT_DATA, help="Path to JSONL reviews file.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction held out for testing.")
    parser.add_argument("--max-features", type=int, default=50000, help="TF-IDF max_features.")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Optional cap on number of reviews to use (0 = all).")
    args = parser.parse_args()

    t0 = time.time()
    contents, labels = [], []
    for content, label in load_reviews(args.data):
        contents.append(content)
        labels.append(label)
        if args.max_samples and len(contents) >= args.max_samples:
            break
    labels = np.asarray(labels)

    print(f"Loaded {len(contents):,} reviews ({time.time()-t0:.1f}s)")
    neg, pos = int((labels == 0).sum()), int((labels == 1).sum())
    print(f"Class distribution:  negative (1-3) = {neg:,}  |  positive (4-5) = {pos:,}")

    def mask_sum(a, m):
        return f"{int(a[m].sum()):,}"

    m0, m1 = labels == 0, labels == 1
    print(f"  negative: {mask_sum(np.ones_like(labels), m0)}  |  positive: {mask_sum(np.ones_like(labels), m1)}")

    X_train, X_test, y_train, y_test = train_test_split(
        contents, labels, test_size=args.test_size, stratify=labels, random_state=42
    )
    print(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            sublinear_tf=True,
            ngram_range=(1, 2),
            min_df=2,
            max_features=args.max_features,
            stop_words="english",
        )),
        # class_weight handles the positive/negative imbalance
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)),
    ])

    t0 = time.time()
    pipeline.fit(X_train, y_train)
    print(f"Training done ({time.time()-t0:.1f}s)")

    y_pred = pipeline.predict(X_test)

    print("\n===== RESULTS =====")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
    print(f"F1-score : {f1_score(y_test, y_pred):.4f}")
    print("\nConfusion matrix [[TN FP], [FN TP]]:")
    print(confusion_matrix(y_test, y_pred))
    print("\n" + classification_report(y_test, y_pred, target_names=["negative", "positive"]))

    # Most informative features per class
    tfidf = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]
    if len(clf.coef_.shape) == 2:  # both classes present in training
        feature_names = np.asarray(tfidf.get_feature_names_out())
        coef = clf.coef_[0]
        top_pos = np.argsort(coef)[-10:][::-1]
        top_neg = np.argsort(coef)[:10]
        print("\nTop 10 features for POSITIVE class:")
        for i in top_pos:
            print(f"  {feature_names[i]:>22}  {coef[i]:+.4f}")
        print("\nTop 10 features for NEGATIVE class:")
        for i in top_neg:
            print(f"  {feature_names[i]:>22}  {coef[i]:+.4f}")

    # A few example predictions
    print("\n===== EXAMPLE PREDICTIONS (TEST SET) =====")
    shown = 0
    for x, y_true in zip(X_test, y_test):
        if shown >= 5:
            break
        y_hat = pipeline.predict([x])[0]
        pred = "positive" if y_hat == 1 else "negative"
        truth = "positive" if y_true == 1 else "negative"
        status = "OK " if y_hat == y_true else "ERR"
        print(f"[{status}] pred={pred:>8} true={truth:>8} | {x[:90]}...")
        shown += 1


if __name__ == "__main__":
    sys.exit(main())

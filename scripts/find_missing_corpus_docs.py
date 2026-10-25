"""
Find source reviews that do not have a corresponding document in corpus.jsonl.

Usage:

    uv run python -m scripts.find_missing_corpus_docs \
        --source ../clean_reviews.jsonl \
        --corpus ../corpus.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--corpus",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("../missing_corpus_docs.json"),
    )

    return parser.parse_args()


def main():

    args = parse_args()

    print("=" * 70)
    print("FINDING SOURCE REVIEWS MISSING FROM CORPUS")
    print("=" * 70)

    # ---------------------------------------------------------------
    # Read corpus doc_ids
    # ---------------------------------------------------------------

    print("\n[1/2] Reading corpus doc_ids...")

    corpus_doc_ids = set()

    with args.corpus.open("r", encoding="utf-8") as f:

        for line in f:

            if not line.strip():
                continue

            record = json.loads(line)

            doc_id = record.get("doc_id")

            if doc_id:
                corpus_doc_ids.add(str(doc_id))

    print(f"Corpus documents: {len(corpus_doc_ids):,}")

    # ---------------------------------------------------------------
    # Read source reviews
    # ---------------------------------------------------------------

    print("\n[2/2] Checking source reviews...")

    missing = []

    total = 0

    with args.source.open("r", encoding="utf-8") as f:

        for line_number, line in enumerate(f, start=1):

            if not line.strip():
                continue

            review = json.loads(line)

            total += 1

            review_id = review.get("review_id")

            if not review_id:
                continue

            review_id = str(review_id)

            if review_id not in corpus_doc_ids:

                missing.append(
                    {
                        "source_line": line_number,
                        "review_id": review_id,
                        "parent_asin": review.get("parent_asin"),
                        "asin": review.get("asin"),
                        "brand": review.get("brand"),
                        "model": review.get("model"),
                        "product_title": review.get("product_title"),
                        "text": review.get("text"),
                        "text_norm": review.get("text_norm"),
                        "n_words": review.get("n_words"),
                        "rating": review.get("rating"),
                        "date": review.get("date"),
                        "verified_purchase": review.get(
                            "verified_purchase"
                        ),
                    }
                )

    # ---------------------------------------------------------------
    # Results
    # ---------------------------------------------------------------

    print()
    print("=" * 70)
    print("RESULT")
    print("=" * 70)

    print(f"Source reviews:       {total:,}")
    print(f"Missing corpus docs:  {len(missing):,}")

    if missing:

        print()
        print("FIRST 30 MISSING REVIEWS")
        print("-" * 70)

        for i, review in enumerate(missing[:30], start=1):

            text = review.get("text") or ""

            print()
            print(f"#{i}")
            print(f"  Review ID:     {review['review_id']}")
            print(f"  Source line:   {review['source_line']}")
            print(f"  Brand:         {review['brand']}")
            print(f"  Model:         {review['model']}")
            print(f"  Parent ASIN:   {review['parent_asin']}")
            print(f"  ASIN:          {review['asin']}")
            print(f"  Rating:        {review['rating']}")
            print(f"  n_words:       {review['n_words']}")
            print(f"  text length:   {len(text)}")
            print(f"  Product title: {review['product_title']}")
            print(f"  Text preview:  {text[:300]!r}")

    # ---------------------------------------------------------------
    # Save complete list
    # ---------------------------------------------------------------

    with args.output.open("w", encoding="utf-8") as f:

        json.dump(
            missing,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print(f"Full missing-review list saved to:")
    print(f"  {args.output}")

    print("=" * 70)


if __name__ == "__main__":
    main()
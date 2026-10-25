"""
Verify that missing model values in clean_reviews.jsonl
are preserved correctly in corpus.jsonl.

The corpus contains chunks, while the source contains reviews.
Therefore, this script maps:

    source review -> doc_id -> corpus chunks

and verifies that:

1. Every source review with a missing model has corresponding corpus chunks.
2. Those corpus chunks also have a missing model.
3. Source reviews with a model do not lose the model in the corpus.
4. The number of source reviews with missing models is reported.
5. Missing-model chunks are reported separately from missing-model documents.

Usage:

    uv run python -m scripts.verify_missing_models_preserved \
        --source ../clean_reviews.jsonl \
        --corpus ../corpus.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify that model metadata is preserved from "
            "clean_reviews.jsonl to corpus.jsonl."
        )
    )

    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Path to clean_reviews.jsonl",
    )

    parser.add_argument(
        "--corpus",
        type=Path,
        required=True,
        help="Path to corpus.jsonl",
    )

    parser.add_argument(
        "--examples",
        type=int,
        default=20,
        help="Number of problematic examples to display",
    )

    return parser.parse_args()


def is_missing(value) -> bool:
    """Return True when value is None or empty/whitespace."""
    if value is None:
        return True

    return not str(value).strip()


def normalise(value) -> str:
    """Normalize metadata values for comparison."""
    if value is None:
        return ""

    return str(value).strip().lower()


def main() -> None:
    args = parse_args()

    source_path = args.source
    corpus_path = args.corpus

    if not source_path.exists():
        print(f"ERROR: Source file not found: {source_path}")
        return

    if not corpus_path.exists():
        print(f"ERROR: Corpus file not found: {corpus_path}")
        return

    print("=" * 70)
    print("VERIFYING SOURCE -> CORPUS MODEL PRESERVATION")
    print("=" * 70)

    print(f"Source: {source_path}")
    print(f"Corpus: {corpus_path}")
    print()

    # ------------------------------------------------------------------
    # STEP 1: Read source reviews
    # ------------------------------------------------------------------

    print("[1/2] Reading source reviews...")

    source_reviews = {}

    total_source_reviews = 0
    source_missing_model = 0
    source_has_model = 0
    source_malformed = 0

    source_missing_model_examples = []

    with source_path.open("r", encoding="utf-8") as f:

        for line_number, line in enumerate(f, start=1):

            if not line.strip():
                continue

            try:
                review = json.loads(line)
            except json.JSONDecodeError:
                source_malformed += 1
                continue

            total_source_reviews += 1

            # The corpus uses doc_id.
            # clean_reviews normally uses review_id.
            review_id = review.get("review_id")

            if not review_id:
                continue

            review_id = str(review_id)

            model = review.get("model")

            missing = is_missing(model)

            source_reviews[review_id] = {
                "model": model,
                "model_missing": missing,
                "brand": review.get("brand"),
                "product_title": review.get("product_title"),
                "parent_asin": review.get("parent_asin"),
                "review_id": review_id,
            }

            if missing:
                source_missing_model += 1

                if len(source_missing_model_examples) < args.examples:
                    source_missing_model_examples.append(review_id)
            else:
                source_has_model += 1

    print(f"Source reviews:       {total_source_reviews:,}")
    print(f"Source missing model: {source_missing_model:,}")
    print(f"Source has model:     {source_has_model:,}")
    print(f"Malformed source:     {source_malformed:,}")
    print()

    # ------------------------------------------------------------------
    # STEP 2: Read corpus
    # ------------------------------------------------------------------

    print("[2/2] Reading corpus chunks...")

    corpus_chunks = 0

    corpus_by_doc = defaultdict(list)

    corpus_missing_model_docs = set()
    corpus_model_docs = set()

    corpus_model_values = {}

    malformed_corpus = 0

    with corpus_path.open("r", encoding="utf-8") as f:

        for line_number, line in enumerate(f, start=1):

            if not line.strip():
                continue

            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                malformed_corpus += 1
                continue

            corpus_chunks += 1

            doc_id = chunk.get("doc_id")

            if not doc_id:
                continue

            doc_id = str(doc_id)

            corpus_by_doc[doc_id].append(chunk)

            model = chunk.get("model")

            if is_missing(model):
                corpus_missing_model_docs.add(doc_id)
            else:
                corpus_model_docs.add(doc_id)
                corpus_model_values[doc_id] = model

    print(f"Corpus chunks:        {corpus_chunks:,}")
    print(f"Corpus documents:     {len(corpus_by_doc):,}")
    print(f"Malformed corpus:     {malformed_corpus:,}")
    print()

    # ------------------------------------------------------------------
    # STEP 3: Compare source reviews with corpus
    # ------------------------------------------------------------------

    print("-" * 70)
    print("COMPARING SOURCE REVIEWS WITH CORPUS")
    print("-" * 70)

    missing_source_docs = []
    missing_model_preserved = []
    missing_model_lost = []

    model_preserved = []
    model_lost = []
    model_changed = []

    # We compare every source review against corpus doc_id.
    for review_id, source in source_reviews.items():

        source_model_missing = source["model_missing"]

        chunks = corpus_by_doc.get(review_id)

        # No chunks for this source review.
        if not chunks:
            missing_source_docs.append(review_id)
            continue

        corpus_models = [
            chunk.get("model")
            for chunk in chunks
        ]

        corpus_missing_flags = [
            is_missing(model)
            for model in corpus_models
        ]

        if source_model_missing:

            # Expected:
            # all chunks should have missing model.

            if all(corpus_missing_flags):
                missing_model_preserved.append(review_id)
            else:
                missing_model_lost.append(
                    {
                        "review_id": review_id,
                        "source_model": source["model"],
                        "corpus_models": corpus_models,
                        "brand": source["brand"],
                        "product_title": source["product_title"],
                    }
                )

        else:

            # Expected:
            # corpus should preserve the source model.

            source_model = normalise(source["model"])

            corpus_normalised = [
                normalise(model)
                for model in corpus_models
                if not is_missing(model)
            ]

            if not corpus_normalised:

                model_lost.append(
                    {
                        "review_id": review_id,
                        "source_model": source["model"],
                        "corpus_models": corpus_models,
                        "brand": source["brand"],
                        "product_title": source["product_title"],
                    }
                )

            elif all(
                model == source_model
                for model in corpus_normalised
            ):
                model_preserved.append(review_id)

            else:
                model_changed.append(
                    {
                        "review_id": review_id,
                        "source_model": source["model"],
                        "corpus_models": corpus_models,
                        "brand": source["brand"],
                        "product_title": source["product_title"],
                    }
                )

    # ------------------------------------------------------------------
    # RESULTS
    # ------------------------------------------------------------------

    print()
    print("SOURCE REVIEWS WITH MISSING MODEL")
    print("-" * 70)

    print(
        f"Source reviews with missing model: "
        f"{source_missing_model:,}"
    )

    print(
        f"Missing model correctly preserved: "
        f"{len(missing_model_preserved):,}"
    )

    print(
        f"Missing model incorrectly filled: "
        f"{len(missing_model_lost):,}"
    )

    print()

    if source_missing_model > 0:

        preserved_pct = (
            len(missing_model_preserved)
            / source_missing_model
            * 100
        )

        print(
            f"Preservation rate: "
            f"{preserved_pct:.2f}%"
        )

    # ------------------------------------------------------------------
    # MODELS THAT SHOULD EXIST
    # ------------------------------------------------------------------

    print()
    print("SOURCE REVIEWS WITH MODEL")
    print("-" * 70)

    print(
        f"Source reviews with model: "
        f"{source_has_model:,}"
    )

    print(
        f"Model correctly preserved: "
        f"{len(model_preserved):,}"
    )

    print(
        f"Model lost in corpus: "
        f"{len(model_lost):,}"
    )

    print(
        f"Model changed in corpus: "
        f"{len(model_changed):,}"
    )

    if source_has_model > 0:

        preservation_pct = (
            len(model_preserved)
            / source_has_model
            * 100
        )

        print(
            f"Model preservation rate: "
            f"{preservation_pct:.2f}%"
        )

    # ------------------------------------------------------------------
    # SOURCE REVIEWS WITHOUT CORPUS CHUNKS
    # ------------------------------------------------------------------

    print()
    print("SOURCE REVIEWS WITHOUT CORPUS CHUNKS")
    print("-" * 70)

    print(
        f"Source reviews without corpus chunks: "
        f"{len(missing_source_docs):,}"
    )

    # ------------------------------------------------------------------
    # PROBLEMATIC MISSING-MODEL CASES
    # ------------------------------------------------------------------

    if missing_model_lost:

        print()
        print("-" * 70)
        print("MISSING MODEL WAS INCORRECTLY FILLED")
        print("-" * 70)

        for item in missing_model_lost[:args.examples]:

            print()
            print(f"Review ID:     {item['review_id']}")
            print(f"Brand:         {item['brand']}")
            print(f"Product title: {item['product_title']}")
            print(f"Source model:  {item['source_model']}")
            print(f"Corpus models: {item['corpus_models']}")

    # ------------------------------------------------------------------
    # PROBLEMATIC MODEL-LOSS CASES
    # ------------------------------------------------------------------

    if model_lost:

        print()
        print("-" * 70)
        print("MODEL WAS LOST IN CORPUS")
        print("-" * 70)

        for item in model_lost[:args.examples]:

            print()
            print(f"Review ID:     {item['review_id']}")
            print(f"Brand:         {item['brand']}")
            print(f"Product title: {item['product_title']}")
            print(f"Source model:  {item['source_model']}")
            print(f"Corpus models: {item['corpus_models']}")

    # ------------------------------------------------------------------
    # PROBLEMATIC MODEL-CHANGE CASES
    # ------------------------------------------------------------------

    if model_changed:

        print()
        print("-" * 70)
        print("MODEL VALUE CHANGED IN CORPUS")
        print("-" * 70)

        for item in model_changed[:args.examples]:

            print()
            print(f"Review ID:     {item['review_id']}")
            print(f"Brand:         {item['brand']}")
            print(f"Product title: {item['product_title']}")
            print(f"Source model:  {item['source_model']}")
            print(f"Corpus models: {item['corpus_models']}")

    # ------------------------------------------------------------------
    # FINAL VERDICT
    # ------------------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)

    critical_failures = []

    if source_missing_model > 0 and missing_model_lost:
        critical_failures.append(
            "Some source reviews with missing model "
            "were assigned a model in the corpus."
        )

    if model_lost:
        critical_failures.append(
            "Some source models were lost in the corpus."
        )

    if model_changed:
        critical_failures.append(
            "Some source models were changed in the corpus."
        )

    if missing_source_docs:
        critical_failures.append(
            "Some source reviews have no corresponding corpus document."
        )

    if not critical_failures:

        print("PASS")
        print()
        print(
            "Model metadata is being preserved correctly "
            "from clean_reviews.jsonl to corpus.jsonl."
        )

        print()
        print(
            "The missing models are source-data limitations, "
            "not a chunking/metadata pipeline problem."
        )

    else:

        print("FAIL")
        print()

        for failure in critical_failures:
            print(f" - {failure}")

    print("=" * 70)


if __name__ == "__main__":
    main()
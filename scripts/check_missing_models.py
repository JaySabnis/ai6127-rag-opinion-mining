"""
Check whether missing model values already exist in the cleaned source data.

Usage:
    uv run python -m scripts.check_missing_models \
        --input ../clean_reviews.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check missing brand/model values in clean_reviews.jsonl"
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to cleaned reviews JSONL file",
    )

    parser.add_argument(
        "--examples",
        type=int,
        default=30,
        help="Number of missing-model examples to display",
    )

    return parser.parse_args()


def is_missing(value) -> bool:
    """Return True when a value is None or empty/whitespace."""
    if value is None:
        return True

    return not str(value).strip()


def main() -> None:
    args = parse_args()

    input_path = args.input

    if not input_path.exists():
        print(f"ERROR: File not found: {input_path}")
        return

    print("=" * 70)
    print("CHECKING MISSING MODEL VALUES")
    print("=" * 70)
    print(f"Input: {input_path}")
    print()

    total_reviews = 0
    missing_model = 0
    missing_brand = 0
    missing_both = 0

    missing_model_examples = []

    missing_model_by_brand: Counter[str] = Counter()

    model_values: Counter[str] = Counter()
    brand_values: Counter[str] = Counter()

    malformed_json = 0

    with input_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):

            if not line.strip():
                continue

            try:
                review = json.loads(line)
            except json.JSONDecodeError:
                malformed_json += 1
                continue

            total_reviews += 1

            brand = review.get("brand")
            model = review.get("model")

            brand_missing = is_missing(brand)
            model_missing = is_missing(model)

            if brand_missing:
                missing_brand += 1

            if model_missing:
                missing_model += 1

            if brand_missing and model_missing:
                missing_both += 1

            # Track existing values
            if not brand_missing:
                brand_values[str(brand).strip()] += 1

            if not model_missing:
                model_values[str(model).strip()] += 1

            # Store missing-model examples
            if model_missing:

                brand_name = (
                    str(brand).strip()
                    if not brand_missing
                    else "<missing>"
                )

                missing_model_by_brand[brand_name] += 1

                if len(missing_model_examples) < args.examples:
                    missing_model_examples.append(
                        {
                            "line": line_number,
                            "review_id": review.get("review_id"),
                            "parent_asin": review.get("parent_asin"),
                            "asin": review.get("asin"),
                            "brand": brand,
                            "model": model,
                            "product_title": review.get("product_title"),
                            "text": str(review.get("text", ""))[:300],
                        }
                    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    print("-" * 70)
    print("SUMMARY")
    print("-" * 70)

    print(f"Total reviews:        {total_reviews:,}")
    print(f"Missing model:        {missing_model:,}")
    print(f"Missing brand:        {missing_brand:,}")
    print(f"Missing both:         {missing_both:,}")
    print(f"Malformed JSON lines: {malformed_json:,}")
    print()

    if total_reviews > 0:
        print(
            f"Missing model %: "
            f"{missing_model / total_reviews * 100:.2f}%"
        )

        print(
            f"Missing brand %: "
            f"{missing_brand / total_reviews * 100:.2f}%"
        )

    # ------------------------------------------------------------------
    # Missing model by brand
    # ------------------------------------------------------------------

    print()
    print("-" * 70)
    print("MISSING MODEL BY BRAND")
    print("-" * 70)

    if missing_model_by_brand:
        for brand, count in missing_model_by_brand.most_common(50):
            print(f"{brand:40s} {count:>8,}")
    else:
        print("No missing models found.")

    # ------------------------------------------------------------------
    # Examples
    # ------------------------------------------------------------------

    print()
    print("-" * 70)
    print(f"EXAMPLES OF REVIEWS WITH MISSING MODEL (max {args.examples})")
    print("-" * 70)

    if not missing_model_examples:
        print("No missing-model reviews found.")
    else:
        for i, example in enumerate(missing_model_examples, start=1):

            print()
            print(f"Example {i}")
            print(f"  Source line:   {example['line']}")
            print(f"  Review ID:     {example['review_id']}")
            print(f"  Parent ASIN:   {example['parent_asin']}")
            print(f"  ASIN:          {example['asin']}")
            print(f"  Brand:         {example['brand']}")
            print(f"  Model:         {example['model']}")
            print(f"  Product title: {example['product_title']}")
            print(f"  Review text:   {example['text']}")

    # ------------------------------------------------------------------
    # Final conclusion
    # ------------------------------------------------------------------

    print()
    print("=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    if missing_model == 0:
        print("PASS: Every cleaned review has a model.")
        print()
        print(
            "Therefore, any missing model values in corpus.jsonl "
            "would indicate a pipeline problem."
        )

    elif missing_model > 0:
        print(
            f"FOUND: {missing_model:,} cleaned reviews already have "
            "a missing model."
        )

        print()
        print(
            "This means missing models may legitimately originate "
            "from the source/cleaned dataset."
        )

        print()
        print(
            "Next step: compare these reviews against corpus.jsonl "
            "to make sure the missing values are preserved correctly."
        )

    print("=" * 70)


if __name__ == "__main__":
    main()
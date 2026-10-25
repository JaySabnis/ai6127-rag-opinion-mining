from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path


REQUIRED_FIELDS = {
    "chunk_id",
    "doc_id",
    "chunk_index",
    "text",
    "chunk_word_count",
    "brand",
    "model",
    "rating",
    "timestamp",
    "verified",
    "helpful_votes",
    "is_renewed",
    "aspect_tags",
    "mentions_other_model",
    "source_url",
}

ALLOWED_ASPECTS = {
    "camera",
    "battery",
    "display",
    "performance",
    "service",
    "carrier",
}

BOOLEAN_FIELDS = {
    "verified",
    "is_renewed",
    "mentions_other_model",
}


def word_count(text: str) -> int:
    return len(text.split())


def validate_corpus(input_path: Path, sample_size: int = 20, seed: int = 42):

    print("=" * 70)
    print("CORPUS VALIDATION")
    print("=" * 70)
    print(f"Input: {input_path}")
    print()

    # Counters
    total_records = 0
    malformed_json = 0

    missing_fields = Counter()
    empty_text = 0
    very_short_text = 0

    duplicate_chunk_ids = 0
    seen_chunk_ids = set()

    word_count_mismatches = 0
    word_counts = []

    too_short_chunks = 0
    normal_chunks = 0
    over_150_chunks = 0

    invalid_ratings = 0
    rating_distribution = Counter()

    invalid_booleans = Counter()

    invalid_aspects = Counter()
    aspect_distribution = Counter()
    no_aspect_chunks = 0

    missing_brand = 0
    missing_model = 0

    brand_distribution = Counter()
    model_distribution = Counter()

    mentions_other_model_count = 0
    other_model_examples = []

    missing_source_url = 0

    documents = defaultdict(list)

    duplicate_texts = Counter()

    random_samples = []
    rng = random.Random(seed)

    errors = []

    # ---------------------------------------------------------
    # PASS 1: Read corpus
    # ---------------------------------------------------------

    print("[1/3] Reading corpus...")

    with input_path.open("r", encoding="utf-8") as f:

        for line_number, line in enumerate(f, start=1):

            line = line.rstrip("\n")

            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                malformed_json += 1

                if len(errors) < 20:
                    errors.append(
                        f"Line {line_number}: malformed JSON: {e}"
                    )

                continue

            total_records += 1

            # -------------------------------------------------
            # Required fields
            # -------------------------------------------------

            for field in REQUIRED_FIELDS:
                if field not in record:
                    missing_fields[field] += 1

            # -------------------------------------------------
            # Chunk ID
            # -------------------------------------------------

            chunk_id = record.get("chunk_id")

            if chunk_id in seen_chunk_ids:
                duplicate_chunk_ids += 1
            else:
                seen_chunk_ids.add(chunk_id)

            # -------------------------------------------------
            # Text
            # -------------------------------------------------

            text = record.get("text", "")

            if not isinstance(text, str) or not text.strip():
                empty_text += 1
                continue

            actual_words = word_count(text)
            reported_words = record.get("chunk_word_count")

            word_counts.append(actual_words)

            if actual_words < 5:
                very_short_text += 1

            if actual_words < 100:
                too_short_chunks += 1
            elif actual_words <= 150:
                normal_chunks += 1
            else:
                over_150_chunks += 1

            if reported_words != actual_words:
                word_count_mismatches += 1

            # -------------------------------------------------
            # Rating
            # -------------------------------------------------

            rating = record.get("rating")

            try:
                rating_float = float(rating)

                if rating_float < 1 or rating_float > 5:
                    invalid_ratings += 1
                else:
                    rating_distribution[rating_float] += 1

            except (TypeError, ValueError):
                invalid_ratings += 1

            # -------------------------------------------------
            # Boolean fields
            # -------------------------------------------------

            for field in BOOLEAN_FIELDS:

                value = record.get(field)

                if not isinstance(value, bool):
                    invalid_booleans[field] += 1

            # -------------------------------------------------
            # Aspects
            # -------------------------------------------------

            aspects = record.get("aspect_tags")

            if not isinstance(aspects, list):
                invalid_aspects["NOT_A_LIST"] += 1
                aspects = []

            if not aspects:
                no_aspect_chunks += 1

            for aspect in aspects:

                if aspect not in ALLOWED_ASPECTS:
                    invalid_aspects[aspect] += 1
                else:
                    aspect_distribution[aspect] += 1

            # -------------------------------------------------
            # Brand / model
            # -------------------------------------------------

            brand = record.get("brand")
            model = record.get("model")

            if not brand:
                missing_brand += 1
            else:
                brand_distribution[str(brand)] += 1

            if not model:
                missing_model += 1
            else:
                model_distribution[str(model)] += 1

            # -------------------------------------------------
            # Other model
            # -------------------------------------------------

            if record.get("mentions_other_model") is True:

                mentions_other_model_count += 1

                if len(other_model_examples) < 20:
                    other_model_examples.append(record)

            # -------------------------------------------------
            # Source URL
            # -------------------------------------------------

            if not record.get("source_url"):
                missing_source_url += 1

            # -------------------------------------------------
            # Document grouping
            # -------------------------------------------------

            doc_id = record.get("doc_id")

            if doc_id:
                documents[doc_id].append(
                    record.get("chunk_index")
                )

            # -------------------------------------------------
            # Duplicate text
            # -------------------------------------------------

            duplicate_texts[text] += 1

            # -------------------------------------------------
            # Reservoir sample
            # -------------------------------------------------

            if len(random_samples) < sample_size:
                random_samples.append(record)
            else:
                index = rng.randint(0, total_records - 1)

                if index < sample_size:
                    random_samples[index] = record

            # -------------------------------------------------
            # Progress
            # -------------------------------------------------

            if total_records % 50000 == 0:
                print(
                    f"  Processed {total_records:,} records...",
                    flush=True,
                )

    # ---------------------------------------------------------
    # DOCUMENT / CHUNK VALIDATION
    # ---------------------------------------------------------

    documents_with_bad_indexes = 0
    total_documents = len(documents)

    min_chunks = None
    max_chunks = None

    one_chunk_documents = 0
    multi_chunk_documents = 0

    for doc_id, indexes in documents.items():

        indexes_sorted = sorted(
            i for i in indexes
            if isinstance(i, int)
        )

        count = len(indexes_sorted)

        if min_chunks is None or count < min_chunks:
            min_chunks = count

        if max_chunks is None or count > max_chunks:
            max_chunks = count

        if count == 1:
            one_chunk_documents += 1
        elif count > 1:
            multi_chunk_documents += 1

        expected = list(range(count))

        if indexes_sorted != expected:
            documents_with_bad_indexes += 1

    # ---------------------------------------------------------
    # DUPLICATE TEXT
    # ---------------------------------------------------------

    duplicate_text_count = sum(
        1
        for count in duplicate_texts.values()
        if count > 1
    )

    # ---------------------------------------------------------
    # STATISTICS
    # ---------------------------------------------------------

    average_chunks = (
        total_records / total_documents
        if total_documents
        else 0
    )

    mean_words = (
        statistics.mean(word_counts)
        if word_counts
        else 0
    )

    median_words = (
        statistics.median(word_counts)
        if word_counts
        else 0
    )

    # ---------------------------------------------------------
    # PRINT RESULTS
    # ---------------------------------------------------------

    print()
    print("[2/3] Validation results")
    print()

    print("-" * 70)
    print("GENERAL")
    print("-" * 70)

    print(f"Documents:                 {total_documents:,}")
    print(f"Chunks:                    {total_records:,}")
    print(f"Malformed JSON lines:      {malformed_json:,}")
    print(f"Unique chunk IDs:          {len(seen_chunk_ids):,}")
    print(f"Duplicate chunk IDs:       {duplicate_chunk_ids:,}")

    print()
    print("-" * 70)
    print("CHUNK LENGTH")
    print("-" * 70)

    print(f"Mean words/chunk:          {mean_words:.2f}")
    print(f"Median words/chunk:        {median_words:.2f}")
    print(f"Minimum words:             {min(word_counts) if word_counts else 0}")
    print(f"Maximum words:             {max(word_counts) if word_counts else 0}")
    print(f"<100 words:                {too_short_chunks:,}")
    print(f"100-150 words:             {normal_chunks:,}")
    print(f">150 words:                {over_150_chunks:,}")
    print(f"Empty chunks:              {empty_text:,}")
    print(f"<5 words:                  {very_short_text:,}")
    print(f"Word-count mismatches:     {word_count_mismatches:,}")

    print()
    print("-" * 70)
    print("DOCUMENT / CHUNK STRUCTURE")
    print("-" * 70)

    print(f"Average chunks/document:   {average_chunks:.2f}")
    print(f"Min chunks/document:       {min_chunks}")
    print(f"Max chunks/document:       {max_chunks}")
    print(f"1-chunk documents:         {one_chunk_documents:,}")
    print(f"Multi-chunk documents:     {multi_chunk_documents:,}")
    print(
        f"Bad chunk-index documents: {documents_with_bad_indexes:,}"
    )

    print()
    print("-" * 70)
    print("RATINGS")
    print("-" * 70)

    for rating in [1, 2, 3, 4, 5]:
        count = rating_distribution.get(float(rating), 0)
        percentage = (
            count / total_records * 100
            if total_records
            else 0
        )
        print(
            f"{rating} stars:                  "
            f"{count:,} ({percentage:.2f}%)"
        )

    print(f"Invalid ratings:            {invalid_ratings:,}")

    print()
    print("-" * 70)
    print("ASPECT COVERAGE")
    print("-" * 70)

    for aspect in sorted(ALLOWED_ASPECTS):

        count = aspect_distribution.get(aspect, 0)

        percentage = (
            count / total_records * 100
            if total_records
            else 0
        )

        print(
            f"{aspect:20s} {count:10,} "
            f"({percentage:6.2f}%)"
        )

    print(
        f"{'No aspect':20s} "
        f"{no_aspect_chunks:10,} "
        f"({no_aspect_chunks / total_records * 100:.2f}%)"
        if total_records
        else "No aspect: 0"
    )

    print()
    print("-" * 70)
    print("METADATA")
    print("-" * 70)

    print(f"Missing brand:              {missing_brand:,}")
    print(f"Missing model:              {missing_model:,}")
    print(f"Missing source URL:         {missing_source_url:,}")

    for field in BOOLEAN_FIELDS:
        print(
            f"Invalid {field}:".ljust(29),
            f"{invalid_booleans.get(field, 0):,}"
        )

    print()
    print("-" * 70)
    print("OTHER MODEL")
    print("-" * 70)

    percentage = (
        mentions_other_model_count / total_records * 100
        if total_records
        else 0
    )

    print(
        f"mentions_other_model=true: "
        f"{mentions_other_model_count:,} "
        f"({percentage:.2f}%)"
    )

    print()
    print("-" * 70)
    print("TOP BRANDS")
    print("-" * 70)

    for brand, count in brand_distribution.most_common(20):
        print(f"{brand:30s} {count:,}")

    print()
    print("-" * 70)
    print("TOP MODELS")
    print("-" * 70)

    for model, count in model_distribution.most_common(20):
        print(f"{model:40s} {count:,}")

    print()
    print("-" * 70)
    print("DUPLICATE TEXT")
    print("-" * 70)

    print(
        f"Unique texts appearing more than once: "
        f"{duplicate_text_count:,}"
    )

    print()
    print("-" * 70)
    print("INVALID ASPECT VALUES")
    print("-" * 70)

    if invalid_aspects:
        for aspect, count in invalid_aspects.items():
            print(f"{aspect}: {count:,}")
    else:
        print("None")

    print()
    print("-" * 70)
    print("MISSING REQUIRED FIELDS")
    print("-" * 70)

    if missing_fields:
        for field, count in missing_fields.items():
            print(f"{field}: {count:,}")
    else:
        print("None")

    # ---------------------------------------------------------
    # RANDOM SAMPLE
    # ---------------------------------------------------------

    print()
    print("[3/3] Random sample inspection")
    print("=" * 70)

    for i, record in enumerate(random_samples, start=1):

        print()
        print(f"SAMPLE {i}")
        print(f"chunk_id:               {record.get('chunk_id')}")
        print(f"doc_id:                 {record.get('doc_id')}")
        print(f"chunk_index:            {record.get('chunk_index')}")
        print(f"brand:                  {record.get('brand')}")
        print(f"model:                  {record.get('model')}")
        print(f"rating:                 {record.get('rating')}")
        print(f"chunk_word_count:       {record.get('chunk_word_count')}")
        print(f"aspect_tags:            {record.get('aspect_tags')}")
        print(
            f"mentions_other_model:  "
            f"{record.get('mentions_other_model')}"
        )
        print(f"text: {record.get('text')}")

    # ---------------------------------------------------------
    # OTHER MODEL EXAMPLES
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("EXAMPLES: mentions_other_model = TRUE")
    print("=" * 70)

    if other_model_examples:

        for i, record in enumerate(other_model_examples, start=1):

            print()
            print(f"EXAMPLE {i}")
            print(f"brand:   {record.get('brand')}")
            print(f"model:   {record.get('model')}")
            print(f"text:    {record.get('text')}")

    else:
        print("No examples found.")

    # ---------------------------------------------------------
    # FINAL PASS / FAIL
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL VALIDATION SUMMARY")
    print("=" * 70)

    checks = {
        "JSONL validity": malformed_json == 0,
        "Required fields": not missing_fields,
        "Unique chunk IDs": duplicate_chunk_ids == 0,
        "Chunk indexes": documents_with_bad_indexes == 0,
        "Word counts": word_count_mismatches == 0,
        "No empty chunks": empty_text == 0,
        "Ratings": invalid_ratings == 0,
        "Boolean fields": not invalid_booleans,
        "Aspect tags": not invalid_aspects,
    }

    for name, passed in checks.items():

        status = "PASS" if passed else "FAIL"

        print(
            f"[{status}] {name}"
        )

    print()
    print("Warnings:")
    print(
        f"- {over_150_chunks:,} chunks are >150 words"
    )
    print(
        f"- {too_short_chunks:,} chunks are <100 words"
    )
    print(
        f"- {duplicate_text_count:,} duplicate text groups"
    )
    print(
        f"- {missing_source_url:,} chunks have no source URL"
    )

    critical_failure = not all(checks.values())

    print()

    if critical_failure:
        print("❌ VALIDATION FAILED")
        return 1

    print("✅ VALIDATION PASSED")
    return 0


def main():

    parser = argparse.ArgumentParser(
        description="Validate RAG corpus JSONL"
    )

    parser.add_argument(
        "--input",
        default="../corpus.jsonl",
        help="Path to corpus.jsonl",
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        default=20,
        help="Number of random samples",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    if not input_path.exists():
        print(f"ERROR: File not found: {input_path}")
        return 1

    return validate_corpus(
        input_path=input_path,
        sample_size=args.sample_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    raise SystemExit(main())
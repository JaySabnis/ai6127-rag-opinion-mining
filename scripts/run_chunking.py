"""
Run the complete Task 2 chunking + metadata pipeline.

Example:

uv run python -m scripts.run_chunking \
    --input /path/to/cleaned_reviews.jsonl \
    --output /path/to/corpus.jsonl \
    --stats /path/to/corpus_statistics.json
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

from chunking.chunk import create_chunk_records
from chunking.metadata import (
    build_metadata,
    build_model_catalog,
)


# ============================================================
# Configuration
# ============================================================

PROGRESS_INTERVAL = 5_000
# PROGRESS_INTERVAL = 10


# ============================================================
# Arguments
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Chunk cleaned smartphone reviews and generate corpus.jsonl."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to cleaned review JSONL.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path for generated corpus.jsonl.",
    )

    parser.add_argument(
        "--stats",
        required=True,
        help="Path for corpus_statistics.jsonl.",
    )

    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Optional limit for testing. Example: --max-records 1000",
    )

    return parser.parse_args()


# ============================================================
# JSONL reader
# ============================================================

def read_jsonl(path: Path):
    """
    Stream JSONL one record at a time.
    """
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                yield json.loads(line)

            except json.JSONDecodeError as exc:
                print(
                    f"[WARNING] Invalid JSON at line {line_number}: {exc}",
                    flush=True,
                )


# ============================================================
# Statistics helpers
# ============================================================

def count_words(text: str) -> int:
    return len(text.split())


def tokenise_for_types(text: str) -> list[str]:
    """
    Simple normalized word tokenizer for vocabulary/type statistics.
    """
    text = text.lower()

    return re.findall(
        r"\b[\w'-]+\b",
        text,
    )


def format_elapsed(seconds: float) -> str:
    """
    Convert seconds into a readable duration.
    """
    if seconds < 60:
        return f"{seconds:.1f}s"

    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)

    if minutes < 60:
        return f"{minutes}m {remaining_seconds}s"

    hours = minutes // 60
    remaining_minutes = minutes % 60

    return f"{hours}h {remaining_minutes}m {remaining_seconds}s"


def print_progress(
    stage: str,
    index: int,
    chunks: int,
    start_time: float,
):
    """
    Print progress information.
    """
    elapsed = time.time() - start_time

    rate = (
        index / elapsed
        if elapsed > 0
        else 0
    )

    print(
        f"      [{stage}] "
        f"Processed {index:,} reviews | "
        f"{chunks:,} chunks | "
        f"{rate:,.0f} reviews/sec | "
        f"elapsed {format_elapsed(elapsed)}",
        flush=True, 
    )


# ============================================================
# Main pipeline
# ============================================================

def main():

    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    stats_path = Path(args.stats)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file does not exist: {input_path}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    stats_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 70)
    print("AI6127 Task 2 — Chunking + Metadata Pipeline")
    print("=" * 70)

    print(f"Input : {input_path}")
    print(f"Output: {output_path}")
    print(f"Stats : {stats_path}")

    if args.max_records:
        print(f"Limit : {args.max_records:,} reviews")

    print("=" * 70)
    print(flush=True)

    pipeline_start = time.time()

    # ---------------------------------------------------------
    # First pass:
    # Build model catalog.
    #
    # We need this for mentions_other_model.
    # ---------------------------------------------------------

    print(
        "[1/3] Building phone model catalog...",
        flush=True,
    )

    catalog_start = time.time()

    reviews_for_catalog = []

    for index, review in enumerate(
        read_jsonl(input_path),
        start=1,
    ):

        reviews_for_catalog.append(review)

        if index % PROGRESS_INTERVAL == 0:
            print_progress(
                stage="Catalog",
                index=index,
                chunks=0,
                start_time=catalog_start,
            )

        if (
            args.max_records is not None
            and index >= args.max_records
        ):
            break

    print(
        f"      Finished catalog pass: "
        f"{len(reviews_for_catalog):,} reviews",
        flush=True,
    )

    model_catalog = build_model_catalog(
        reviews_for_catalog
    )

    print(
        f"      Found {len(model_catalog):,} "
        f"brand/model combinations.",
        flush=True,
    )

    print(
        f"      Catalog pass completed in "
        f"{format_elapsed(time.time() - catalog_start)}",
        flush=True,
    )

    # ---------------------------------------------------------
    # Second pass:
    # Generate chunks.
    # ---------------------------------------------------------

    print()
    print(
        "[2/3] Generating corpus.jsonl...",
        flush=True,
    )

    chunking_start = time.time()

    documents = 0
    chunks = 0

    raw_words = 0
    chunk_words = 0

    vocabulary = set()

    rating_distribution = Counter()
    brand_distribution = Counter()
    year_distribution = Counter()
    aspect_distribution = Counter()

    chunk_count_distribution = Counter()

    reviews_under_150 = 0
    reviews_over_150 = 0

    chunks_with_other_model = 0
    chunks_with_multiple_aspects = 0

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:

        for index, review in enumerate(
            read_jsonl(input_path),
            start=1,
        ):

            if (
                args.max_records is not None
                and index > args.max_records
            ):
                break

            documents += 1

            source_text = (
                review.get("text_norm")
                or review.get("text")
                or ""
            )

            source_word_count = count_words(
                source_text
            )

            raw_words += source_word_count

            # -------------------------------------------------
            # Vocabulary statistics should use cleaned
            # review text, not repeated overlap.
            # -------------------------------------------------

            vocabulary.update(
                tokenise_for_types(source_text)
            )

            if source_word_count <= 150:
                reviews_under_150 += 1
            else:
                reviews_over_150 += 1

            # -------------------------------------------------
            # Review-level distributions
            # -------------------------------------------------

            rating = review.get("rating")

            if rating is not None:
                rating_distribution[
                    str(rating)
                ] += 1

            brand = str(
                review.get("brand") or "UNKNOWN"
            ).strip()

            brand_distribution[brand] += 1

            date = str(
                review.get("date")
                or review.get("timestamp")
                or ""
            )

            year_match = re.match(
                r"^(\d{4})",
                date,
            )

            if year_match:
                year_distribution[
                    year_match.group(1)
                ] += 1

            # -------------------------------------------------
            # Create chunks
            # -------------------------------------------------

            chunk_records = create_chunk_records(
                review
            )

            chunk_count_distribution[
                len(chunk_records)
            ] += 1

            for chunk in chunk_records:

                record = build_metadata(
                    review=review,
                    chunk=chunk,
                    model_catalog=model_catalog,
                )

                output_file.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                chunks += 1

                current_chunk_words = record[
                    "chunk_word_count"
                ]

                chunk_words += current_chunk_words

                # -------------------------------------------------
                # Aspect statistics
                # -------------------------------------------------

                aspects = record["aspect_tags"]

                for aspect in aspects:
                    aspect_distribution[aspect] += 1

                if len(aspects) > 1:
                    chunks_with_multiple_aspects += 1

                if record[
                    "mentions_other_model"
                ]:
                    chunks_with_other_model += 1

            # -------------------------------------------------
            # Progress
            # -------------------------------------------------

            if index % PROGRESS_INTERVAL == 0:
                print_progress(
                    stage="Chunking",
                    index=index,
                    chunks=chunks,
                    start_time=chunking_start,
                )

    print(
        f"      Finished chunking: "
        f"{documents:,} reviews → {chunks:,} chunks",
        flush=True,
    )

    print(
        f"      Chunking pass completed in "
        f"{format_elapsed(time.time() - chunking_start)}",
        flush=True,
    )

    # ---------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------

    print()
    print(
        "[3/3] Writing statistics...",
        flush=True,
    )

    statistics = {

        "documents": documents,

        "chunks": chunks,

        # Cleaned-review word count.
        "words": raw_words,

        # Chunk word count includes overlap.
        "chunk_words_including_overlap": chunk_words,

        "types": len(vocabulary),

        "chunking": {

            "max_words": 150,

            "target_min_words": 100,

            "overlap": "one_sentence",

            "reviews_under_or_equal_150": (
                reviews_under_150
            ),

            "reviews_over_150": (
                reviews_over_150
            ),

            "average_chunks_per_document": (
                chunks / documents
                if documents
                else 0
            ),

            "chunks_with_multiple_aspects": (
                chunks_with_multiple_aspects
            ),

            "chunks_mentioning_other_model": (
                chunks_with_other_model
            ),
        },

        "distributions": {

            "rating": dict(
                sorted(
                    rating_distribution.items()
                )
            ),

            "brand": dict(
                sorted(
                    brand_distribution.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
            ),

            "year": dict(
                sorted(
                    year_distribution.items()
                )
            ),

            "aspect": dict(
                sorted(
                    aspect_distribution.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
            ),

            "chunks_per_document": dict(
                sorted(
                    chunk_count_distribution.items(),
                    key=lambda item: int(item[0]),
                )
            ),
        },
    }

    with stats_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            statistics,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    total_time = time.time() - pipeline_start

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)

    print(f"Documents : {documents:,}")
    print(f"Chunks    : {chunks:,}")
    print(f"Words     : {raw_words:,}")
    print(f"Types     : {len(vocabulary):,}")

    print()
    print(
        f"Average chunks/document: "
        f"{chunks / documents:.2f}"
        if documents
        else "Average chunks/document: 0"
    )

    print()
    print(f"Total runtime: {format_elapsed(total_time)}")

    print()
    print(f"Corpus: {output_path}")
    print(f"Stats : {stats_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
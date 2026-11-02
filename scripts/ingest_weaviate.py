"""
Push corpus.jsonl chunk data into Weaviate.

This script imports:
- chunk text
- chunk metadata

It does NOT generate or upload embeddings.
Embeddings can be handled separately by Subgroup B.

Example smoke test:

uv run python scripts/ingest_weaviate.py \
    --input ../corpus.jsonl \
    --batch-size 100 \
    --max-records 1000

Full import:

uv run python scripts/ingest_weaviate.py \
    --input ../corpus.jsonl \
    --batch-size 100
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import weaviate
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.init import Auth


COLLECTION_NAME = "ReviewChunk"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Import corpus.jsonl chunks into Weaviate."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to corpus.jsonl",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of objects per batch.",
    )

    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Maximum number of records to import. "
             "Useful for testing. Omit for full import.",
    )

    return parser.parse_args()


def create_collection(client):
    """
    Create the ReviewChunk collection.

    Self-provided vectors are configured because embeddings
    will be handled separately later.
    """

    if client.collections.exists(COLLECTION_NAME):
        print(
            f"Collection '{COLLECTION_NAME}' already exists."
        )
        return client.collections.get(COLLECTION_NAME)

    collection = client.collections.create(
        name=COLLECTION_NAME,

        vector_config=Configure.Vectors.self_provided(),

        properties=[
            Property(
                name="chunk_id",
                data_type=DataType.TEXT,
            ),
            Property(
                name="doc_id",
                data_type=DataType.TEXT,
            ),
            Property(
                name="chunk_index",
                data_type=DataType.INT,
            ),
            Property(
                name="text",
                data_type=DataType.TEXT,
            ),
            Property(
                name="chunk_word_count",
                data_type=DataType.INT,
            ),
            Property(
                name="brand",
                data_type=DataType.TEXT,
            ),
            Property(
                name="model",
                data_type=DataType.TEXT,
            ),
            Property(
                name="rating",
                data_type=DataType.INT,
            ),
            Property(
                name="timestamp",
                data_type=DataType.DATE,
            ),
            Property(
                name="verified",
                data_type=DataType.BOOL,
            ),
            Property(
                name="helpful_votes",
                data_type=DataType.INT,
            ),
            Property(
                name="is_renewed",
                data_type=DataType.BOOL,
            ),
            Property(
                name="aspect_tags",
                data_type=DataType.TEXT_ARRAY,
            ),
            Property(
                name="mentions_other_model",
                data_type=DataType.BOOL,
            ),
            Property(
                name="source_url",
                data_type=DataType.TEXT,
            ),
        ],
    )

    print(f"Created collection '{COLLECTION_NAME}'.")
    return collection


def clean_properties(record: dict) -> dict:
    """
    Convert one corpus.jsonl record into Weaviate properties.
    """

    properties = {
        "chunk_id": record["chunk_id"],
        "doc_id": record["doc_id"],
        "chunk_index": record["chunk_index"],
        "text": record["text"],
        "chunk_word_count": record["chunk_word_count"],
        "brand": record["brand"],
        "model": record["model"],
        "rating": record["rating"],
        "verified": record["verified"],
        "helpful_votes": record["helpful_votes"],
        "is_renewed": record["is_renewed"],
        "aspect_tags": record["aspect_tags"],
        "mentions_other_model": record["mentions_other_model"],
        "source_url": record["source_url"],
    }

    # Convert YYYY-MM-DD into an ISO timestamp
    # accepted by Weaviate's DATE type.
    if record.get("timestamp"):
        properties["timestamp"] = (
            f"{record['timestamp']}T00:00:00Z"
        )

    return properties


def main():
    args = parse_args()

    input_path = Path(args.input)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Corpus does not exist: {input_path}"
        )

    if args.max_records is not None and args.max_records <= 0:
        raise ValueError(
            "--max-records must be greater than 0."
        )

    if args.batch_size <= 0:
        raise ValueError(
            "--batch-size must be greater than 0."
        )

    # ---------------------------------------------------------
    # Environment
    # ---------------------------------------------------------

    weaviate_url = os.environ.get("WEAVIATE_URL")
    weaviate_api_key = os.environ.get("WEAVIATE_API_KEY")

    if not weaviate_url:
        raise RuntimeError(
            "WEAVIATE_URL is not set."
        )

    if not weaviate_api_key:
        raise RuntimeError(
            "WEAVIATE_API_KEY is not set."
        )

    # ---------------------------------------------------------
    # Connect
    # ---------------------------------------------------------

    print("Connecting to Weaviate...")

    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=weaviate_url,
        auth_credentials=Auth.api_key(
            weaviate_api_key
        ),
    )

    try:
        print(
            f"Weaviate ready: {client.is_ready()}"
        )

        if not client.is_ready():
            raise RuntimeError(
                "Weaviate is not ready."
            )

        # -----------------------------------------------------
        # Collection
        # -----------------------------------------------------

        collection = create_collection(client)

        imported = 0
        skipped = 0

        # -----------------------------------------------------
        # Import
        # -----------------------------------------------------

        print()
        print("Starting import...")
        print(f"Input: {input_path}")
        print(f"Batch size: {args.batch_size}")

        if args.max_records is None:
            print("Maximum records: ALL")
        else:
            print(
                f"Maximum records: {args.max_records:,}"
            )

        print()

        with input_path.open(
            "r",
            encoding="utf-8",
        ) as f:

            with collection.batch.fixed_size(
                batch_size=args.batch_size,
                concurrent_requests=4,
            ) as batch:

                for line in f:

                    # Stop after requested number
                    # of successfully processed records.
                    if (
                        args.max_records is not None
                        and imported >= args.max_records
                    ):
                        break

                    line = line.strip()

                    if not line:
                        skipped += 1
                        continue

                    try:
                        record = json.loads(line)

                        properties = clean_properties(
                            record
                        )

                        # IMPORTANT:
                        # No vector/embedding is supplied here.
                        batch.add_object(
                            properties=properties
                        )

                        imported += 1

                        if (
                            imported % 1000 == 0
                            or imported == args.max_records
                        ):
                            print(
                                f"\rImported "
                                f"{imported:,} chunks...",
                                end="",
                                flush=True,
                            )

                    except Exception as exc:
                        print()
                        print(
                            f"\nERROR processing record "
                            f"{imported + 1:,}:"
                        )
                        print(exc)
                        raise

        print()
        print()

        # -----------------------------------------------------
        # Results
        # -----------------------------------------------------

        failed = collection.batch.failed_objects

        print("=" * 60)
        print("IMPORT COMPLETE")
        print("=" * 60)
        print(
            f"Imported: {imported:,}"
        )
        print(
            f"Skipped empty lines: {skipped:,}"
        )
        print(
            f"Failed: {len(failed):,}"
        )

        if args.max_records is not None:
            print(
                f"Test limit: {args.max_records:,}"
            )

        if failed:
            print()
            print("First failed object:")
            print(failed[0])

        print("=" * 60)

    finally:
        client.close()


if __name__ == "__main__":
    main()
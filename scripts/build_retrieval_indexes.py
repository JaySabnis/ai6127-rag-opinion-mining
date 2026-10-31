"""Build the retrieval indexes from corpus.jsonl.

Stages (run all, or one at a time):

    meta    corpus.jsonl  -> indexes/meta.pkl          (~1 min)
    bm25    meta.pkl      -> indexes/bm25/             (a few minutes)
    encode  meta.pkl      -> indexes/dense/shards/     (GPU/MPS recommended; resumable)
    faiss   shards        -> indexes/dense/faiss.index (~1 min)

Examples (from the repository root):

    uv run python -m scripts.build_retrieval_indexes --stage all
    uv run python -m scripts.build_retrieval_indexes --stage encode --device mps --batch-size 256
    # quick end-to-end check on the first 5,000 chunks:
    uv run python -m scripts.build_retrieval_indexes --stage all --max-records 5000 --index-dir indexes_small
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from retrieval.config import DEFAULT_CORPUS, DEFAULT_EMBED_MODEL, DEFAULT_INDEX_DIR, IndexPaths

STAGES = ("meta", "bm25", "encode", "faiss")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=STAGES + ("dense", "all"), default="all")
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    ap.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    ap.add_argument("--max-records", type=int, default=None, help="only index the first N chunks (for testing)")
    ap.add_argument("--stemmer", default="auto", choices=["auto", "pystemmer", "nltk", "none"])
    ap.add_argument("--model", default=DEFAULT_EMBED_MODEL)
    ap.add_argument("--device", default=None, help="cuda / mps / cpu (default: auto)")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--shard-size", type=int, default=50_000)
    args = ap.parse_args()

    if args.stage == "all":
        stages = STAGES
    elif args.stage == "dense":
        stages = ("encode", "faiss")
    else:
        stages = (args.stage,)

    paths = IndexPaths(args.index_dir)
    from retrieval.corpus import build_meta, load_meta

    meta = None
    for stage in stages:
        t0 = time.time()
        print(f"== stage: {stage}")
        if stage == "meta":
            meta = build_meta(args.corpus, paths.meta, args.max_records)
            print(f"  {len(meta):,} chunks, {meta['doc_id'].nunique():,} reviews -> {paths.meta}")
        else:
            if meta is None:
                meta = load_meta(paths.meta)
            texts = meta["text"].tolist()
            if stage == "bm25":
                from retrieval.bm25 import BM25Index

                idx = BM25Index.build(texts, stemmer=args.stemmer)
                idx.save(paths.bm25_dir)
            elif stage == "encode":
                from retrieval.dense import SentenceTransformerEncoder, encode_corpus

                enc = SentenceTransformerEncoder(args.model, device=args.device, batch_size=args.batch_size)
                print(f"  encoder {enc.name} on {enc.device}, dim={enc.dim}")
                encode_corpus(texts, enc, paths.dense_dir, shard_size=args.shard_size)
            elif stage == "faiss":
                from retrieval.dense import DenseIndex

                DenseIndex.build(paths.dense_dir)
        print(f"   done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

"""Paths and default settings for the retrieval module.

Everything the retrieval stage writes lives under one index directory
(default: ``indexes/``), which is git-ignored:

    indexes/
    ├── meta.pkl            row-aligned chunk metadata (row i == FAISS id i == BM25 doc i)
    ├── bm25/               sparse BM25 index
    └── dense/              embedding shards + faiss.index
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_CORPUS = Path("data/corpus.jsonl")
DEFAULT_INDEX_DIR = Path("indexes")

# Dense retrieval -----------------------------------------------------------
DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
# BGE v1.5 recommends this instruction on the *query* side only.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# BM25 (Lucene-style defaults, same as Elasticsearch/Pyserini) --------------
BM25_K1 = 1.2
BM25_B = 0.75

# Hybrid --------------------------------------------------------------------
RRF_K = 60  # Cormack et al. (2009) default


@dataclass
class IndexPaths:
    root: Path = DEFAULT_INDEX_DIR
    meta: Path = field(init=False)
    bm25_dir: Path = field(init=False)
    dense_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.meta = self.root / "meta.pkl"
        self.bm25_dir = self.root / "bm25"
        self.dense_dir = self.root / "dense"

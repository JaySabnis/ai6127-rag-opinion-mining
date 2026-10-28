"""Retrieval components for smartphone review evidence.

    retriever.py      BM25 + dense (BGE/FAISS) + RRF hybrid -> candidate pool
    query_rewrite.py  DeepSeek query rewrite with product/aspect extraction
    rerank.py         MiniLM cross-encoder reranking of a candidate pool

``from retrieval import Retriever`` works, but the import is lazy so that
using only ``retrieval.rerank`` or ``retrieval.query_rewrite`` does not load
FAISS, pandas or the indexes.
"""

__all__ = ["Hit", "RetrievalResult", "Retriever"]


def __getattr__(name):
    if name in __all__:
        from retrieval import retriever

        return getattr(retriever, name)
    raise AttributeError(f"module 'retrieval' has no attribute {name!r}")

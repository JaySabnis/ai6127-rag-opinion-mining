"""Retrieval stage: BM25 + dense (BGE/FAISS) + RRF hybrid.

Public interface: ``Retriever`` (see retriever.py and retrieval/README.md).
"""

from retrieval.retriever import Hit, RetrievalResult, Retriever

__all__ = ["Hit", "RetrievalResult", "Retriever"]

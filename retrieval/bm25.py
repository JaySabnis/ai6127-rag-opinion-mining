"""Sparse retrieval: BM25 implemented on a precomputed sparse weight matrix.

For document d and query term t (Lucene / Robertson-Sparck Jones form):

    idf(t)    = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
    w(d, t)   = idf(t) * tf(d,t) * (k1 + 1) / (tf(d,t) + k1 * (1 - b + b * |d| / avgdl))
    score(d,q) = sum over query terms t of qtf(t) * w(d, t)

Because w(d, t) does not depend on the query, it is computed once for every
(document, term) pair and stored as a CSC sparse matrix W (documents x vocab).
A query then only reads the columns of its own terms:

    scores = W[:, query_term_ids] @ query_term_counts

which takes milliseconds even for ~460k documents.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import scipy.sparse as sp

from retrieval.config import BM25_B, BM25_K1
from retrieval.text import Tokenizer


class BM25Index:
    def __init__(self, weights: sp.csc_matrix, vocab: dict[str, int], tokenizer: Tokenizer, params: dict):
        self.W = weights
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.params = params

    @property
    def n_docs(self) -> int:
        return self.W.shape[0]

    # ------------------------------------------------------------------ build
    @classmethod
    def build(
        cls,
        texts: Iterable[str],
        k1: float = BM25_K1,
        b: float = BM25_B,
        stemmer: str = "auto",
        log_every: int = 50_000,
    ) -> "BM25Index":
        tok = Tokenizer(stemmer)
        vocab: dict[str, int] = {}
        indptr = [0]
        indices: list[int] = []
        counts: list[int] = []
        t0 = time.time()
        for i, text in enumerate(texts):
            c = Counter(tok(text))
            for term, n in c.items():
                tid = vocab.setdefault(term, len(vocab))
                indices.append(tid)
                counts.append(n)
            indptr.append(len(indices))
            if log_every and (i + 1) % log_every == 0:
                print(f"  tokenised {i + 1:,} docs ({time.time() - t0:.0f}s)")

        n_docs = len(indptr) - 1
        tf = sp.csr_matrix(
            (np.asarray(counts, dtype=np.float32), np.asarray(indices, dtype=np.int32), np.asarray(indptr, dtype=np.int64)),
            shape=(n_docs, len(vocab)),
        )
        doc_len = np.asarray(tf.sum(axis=1)).ravel()
        avgdl = float(doc_len.mean()) if n_docs else 0.0
        df = np.bincount(tf.indices, minlength=len(vocab)).astype(np.float64)
        idf = np.log1p((n_docs - df + 0.5) / (df + 0.5))

        # Per-nonzero BM25 weight. Row i of the CSR data spans indptr[i]:indptr[i+1].
        row_of_nnz = np.repeat(np.arange(n_docs), np.diff(tf.indptr))
        norm = k1 * (1.0 - b + b * doc_len[row_of_nnz] / max(avgdl, 1e-9))
        tfv = tf.data.astype(np.float64)
        w = idf[tf.indices] * tfv * (k1 + 1.0) / (tfv + norm)
        W = sp.csr_matrix((w.astype(np.float32), tf.indices, tf.indptr), shape=tf.shape).tocsc()

        params = {
            "k1": k1,
            "b": b,
            "n_docs": n_docs,
            "vocab_size": len(vocab),
            "avgdl": avgdl,
            "stemmer": tok.stemmer_name,
            "nnz": int(W.nnz),
        }
        print(f"  BM25 built: {n_docs:,} docs, {len(vocab):,} terms, {W.nnz:,} postings ({time.time() - t0:.0f}s)")
        return cls(W, vocab, tok, params)

    # ------------------------------------------------------------- persist
    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        sp.save_npz(d / "weights.npz", self.W, compressed=False)
        (d / "vocab.json").write_text(json.dumps(self.vocab), encoding="utf-8")
        (d / "params.json").write_text(json.dumps(self.params, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path) -> "BM25Index":
        d = Path(directory)
        params = json.loads((d / "params.json").read_text(encoding="utf-8"))
        vocab = json.loads((d / "vocab.json").read_text(encoding="utf-8"))
        W = sp.load_npz(d / "weights.npz").tocsc()
        stem = params.get("stemmer", "none")
        tok = Tokenizer({"pystemmer-english": "pystemmer", "nltk-snowball-english": "nltk"}.get(stem, "none"))
        if tok.stemmer_name != stem:
            raise RuntimeError(f"BM25 index was built with stemmer '{stem}' but only '{tok.stemmer_name}' is available")
        return cls(W, vocab, tok, params)

    # -------------------------------------------------------------- search
    def score_all(self, query: str) -> np.ndarray:
        """BM25 score of every document for this query (zeros where no term matches)."""
        counts = Counter(t for t in self.tokenizer(query) if t in self.vocab)
        if not counts:
            return np.zeros(self.n_docs, dtype=np.float32)
        ids = np.fromiter((self.vocab[t] for t in counts), dtype=np.int64, count=len(counts))
        qtf = np.fromiter(counts.values(), dtype=np.float32, count=len(counts))
        return np.asarray(self.W[:, ids] @ qtf).ravel()

    def search(self, query: str, k: int, rows: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Top-k (row_ids, scores), best first. ``rows`` restricts the search to a subset."""
        scores = self.score_all(query)
        if rows is not None:
            cand, cand_scores = rows, scores[rows]
        else:
            cand, cand_scores = None, scores
        positive = np.flatnonzero(cand_scores > 0)
        if positive.size == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        k = min(k, positive.size)
        top = positive[np.argpartition(-cand_scores[positive], k - 1)[:k]]
        top = top[np.argsort(-cand_scores[top], kind="stable")]
        row_ids = cand[top] if cand is not None else top
        return row_ids.astype(np.int64), cand_scores[top].astype(np.float32)

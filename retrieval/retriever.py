"""Unified retrieval interface used by the reranker, the UI and generation.

    from retrieval import Retriever

    r = Retriever.load("indexes")
    res = r.retrieve("How is the camera on the Galaxy S10?", mode="hybrid", k=50)
    for hit in res.hits:
        print(hit.rank, hit.chunk_id, hit.text)

Modes
-----
``bm25``    sparse only          (baseline)
``dense``   dense only           (baseline)
``hybrid``  RRF(bm25, dense) + optional aspect boost

Each ranker retrieves ``candidate_k`` rows; the fused list is cut to ``k``.
A reranker should call ``retrieve(..., k=50)`` (or similar) and re-order the
returned hits.

Filters
-------
Hard metadata filters (``model``, ``brand``, ``min_rating``, ``max_rating``,
``exclude_renewed``) are applied inside both rankers. With ``auto_filter=True``
the model detected in the query is used as a filter unless the caller passed an
explicit model/brand filter. Aspects are never a hard filter (see hybrid.py).
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from retrieval.bm25 import BM25Index
from retrieval.config import RRF_K, IndexPaths
from retrieval.corpus import load_meta, rows_for
from retrieval.dense import DenseIndex, Encoder
from retrieval.hybrid import add_aspect_boost, rank_fused, ranks_of, rrf_fuse
from retrieval.query_parser import ParsedQuery, QueryParser

MODES = ("bm25", "dense", "hybrid")
FILTER_KEYS = {"model", "brand", "min_rating", "max_rating", "exclude_renewed"}


@dataclass
class Hit:
    rank: int
    row_id: int
    chunk_id: str
    doc_id: str
    text: str
    score: float  # score of the requested mode (RRF score for hybrid)
    brand: str | None
    model: str | None
    rating: int | None
    timestamp: str | None
    aspect_tags: list[str]
    helpful_votes: int | None
    verified: bool | None
    is_renewed: bool | None
    source_url: str | None
    bm25_score: float | None = None
    bm25_rank: int | None = None
    dense_score: float | None = None
    dense_rank: int | None = None
    aspect_boosted: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievalResult:
    query: str
    mode: str
    search_text: str  # text actually sent to BM25 / the encoder
    parsed: ParsedQuery
    filters: dict[str, Any]
    hits: list[Hit]
    n_candidates: int | None  # rows allowed by the filters (None = whole corpus)
    timings_ms: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "mode": self.mode,
            "search_text": self.search_text,
            "parsed": self.parsed.to_dict(),
            "filters": self.filters,
            "n_candidates": self.n_candidates,
            "timings_ms": self.timings_ms,
            "hits": [h.to_dict() for h in self.hits],
        }


def _none_if_nan(v):
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v


class Retriever:
    def __init__(self, meta: pd.DataFrame, bm25: BM25Index | None, dense: DenseIndex | None):
        if bm25 is not None and bm25.n_docs != len(meta):
            raise RuntimeError(f"BM25 has {bm25.n_docs} docs but meta has {len(meta)} rows; rebuild the indexes")
        if dense is not None and dense.backend.ntotal != len(meta):
            raise RuntimeError(f"dense index has {dense.backend.ntotal} vectors but meta has {len(meta)} rows")
        self.meta = meta
        self.bm25 = bm25
        self.dense = dense
        self.parser = QueryParser(meta)
        self._aspects = meta["aspect_tags"].tolist()
        # model -> row ids, so the common "model only" filter is a dict lookup.
        self._model_rows = {m: np.sort(ix) for m, ix in meta.groupby("model").indices.items()}

    @classmethod
    def load(
        cls,
        index_dir: str | Path = "indexes",
        encoder: Encoder | None = None,
        use_bm25: bool = True,
        use_dense: bool = True,
    ) -> "Retriever":
        paths = IndexPaths(index_dir)
        meta = load_meta(paths.meta)
        bm25 = BM25Index.load(paths.bm25_dir) if use_bm25 else None
        dense = None
        if use_dense:
            if encoder is None:
                from retrieval.dense import SentenceTransformerEncoder
                import json

                info = json.loads((paths.dense_dir / "info.json").read_text())
                encoder = SentenceTransformerEncoder(info["model"])
            dense = DenseIndex.load(paths.dense_dir, encoder)
            # The first encode on a GPU/MPS is slow (~300 ms vs ~10 ms); pay it at
            # load time instead of on the user's first query.
            dense.encode_query("warm up")
        return cls(meta, bm25, dense)

    # ------------------------------------------------------------------ helpers
    def _resolve_filters(self, parsed: ParsedQuery, filters: dict | None, auto_filter: bool) -> dict:
        filters = dict(filters or {})
        unknown = set(filters) - FILTER_KEYS
        if unknown:
            raise ValueError(f"unknown filter keys {unknown}; allowed: {sorted(FILTER_KEYS)}")
        if auto_filter and parsed.model and "model" not in filters and "brand" not in filters:
            filters["model"] = parsed.model
        return {k: v for k, v in filters.items() if v is not None and v is not False}

    def _make_hit(self, rank: int, row: int, score: float, extra: dict) -> Hit:
        r = self.meta.iloc[row]
        rating = _none_if_nan(r["rating"])
        votes = _none_if_nan(r["helpful_votes"])
        verified = _none_if_nan(r["verified"])
        renewed = _none_if_nan(r["is_renewed"])
        return Hit(
            rank=rank,
            row_id=int(row),
            chunk_id=r["chunk_id"],
            doc_id=r["doc_id"],
            text=r["text"],
            score=float(score),
            brand=_none_if_nan(r["brand"]),
            model=_none_if_nan(r["model"]),
            rating=int(rating) if rating is not None else None,
            timestamp=_none_if_nan(r["timestamp"]),
            aspect_tags=list(r["aspect_tags"]),
            # plain Python types so to_dict() is JSON-serialisable
            helpful_votes=int(votes) if votes is not None else None,
            verified=bool(verified) if verified is not None else None,
            is_renewed=bool(renewed) if renewed is not None else None,
            source_url=_none_if_nan(r["source_url"]),
            **extra,
        )

    # --------------------------------------------------------------- retrieve
    def retrieve(
        self,
        query: str,
        mode: str = "hybrid",
        k: int = 10,
        candidate_k: int = 100,
        filters: dict | None = None,
        auto_filter: bool = True,
        aspect_weight: float = 0.5,
        max_per_doc: int | None = 1,
        rrf_k: int = RRF_K,
        strip_model: bool = True,
    ) -> RetrievalResult:
        """Retrieve the top-k chunks for a query.

        Args:
            mode: "bm25", "dense" or "hybrid".
            k: number of hits returned.
            candidate_k: depth retrieved from each ranker before fusion.
            filters: hard metadata filters, e.g. {"model": "Galaxy S10", "min_rating": 4}.
            auto_filter: use the model detected in the query as a filter.
            aspect_weight: hybrid aspect boost (0 disables it).
            max_per_doc: keep at most this many chunks per original review
                (overlapping chunks of one review are not independent evidence);
                None keeps all.
            strip_model: when results are already filtered to the model named in
                the query, remove the model name from the text sent to BM25 and
                the encoder. Otherwise every candidate shares the model, and
                chunks that merely *repeat* the name ("works for Galaxy S10")
                outrank chunks about the asked aspect.
        """
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if mode in ("bm25", "hybrid") and self.bm25 is None:
            raise RuntimeError("BM25 index not loaded")
        if mode in ("dense", "hybrid") and self.dense is None:
            raise RuntimeError("dense index not loaded")

        timings: dict[str, float] = {}
        t_all = time.perf_counter()

        t = time.perf_counter()
        parsed = self.parser.parse(query)
        applied = self._resolve_filters(parsed, filters, auto_filter)
        if set(applied) == {"model"}:
            rows = self._model_rows.get(applied["model"], np.empty(0, dtype=np.int64))
        else:
            rows = rows_for(self.meta, **applied)

        search_text = query
        if strip_model and parsed.model and applied.get("model") == parsed.model and parsed.without_model:
            # Keep the original if nothing searchable would remain ("Galaxy S10?").
            if self.bm25 is None or self.bm25.tokenizer(parsed.without_model):
                search_text = parsed.without_model
        timings["parse_filter"] = (time.perf_counter() - t) * 1000

        # Retrieve deeper than k so per-review de-duplication still leaves k hits.
        depth = max(candidate_k, k * (3 if max_per_doc else 1))

        bm_rows = bm_scores = de_rows = de_scores = None
        if mode in ("bm25", "hybrid"):
            t = time.perf_counter()
            bm_rows, bm_scores = self.bm25.search(search_text, depth, rows)
            timings["bm25"] = (time.perf_counter() - t) * 1000
        if mode in ("dense", "hybrid"):
            t = time.perf_counter()
            qv = self.dense.encode_query(search_text)
            timings["dense_encode"] = (time.perf_counter() - t) * 1000
            t = time.perf_counter()
            de_rows, de_scores = self.dense.search(k=depth, rows=rows, query_vector=qv)
            timings["dense_search"] = (time.perf_counter() - t) * 1000

        bm_rank = ranks_of(bm_rows) if bm_rows is not None else {}
        de_rank = ranks_of(de_rows) if de_rows is not None else {}
        bm_score = dict(zip(map(int, bm_rows), map(float, bm_scores))) if bm_rows is not None else {}
        de_score = dict(zip(map(int, de_rows), map(float, de_scores))) if de_rows is not None else {}

        t = time.perf_counter()
        boosted: set[int] = set()
        if mode == "bm25":
            ranked = list(zip(map(int, bm_rows), map(float, bm_scores)))
        elif mode == "dense":
            ranked = list(zip(map(int, de_rows), map(float, de_scores)))
        else:
            fused = rrf_fuse({"bm25": bm_rows, "dense": de_rows}, k=rrf_k)
            boosted = add_aspect_boost(fused, self._aspects, parsed.aspects, aspect_weight, k=rrf_k)
            ranked = rank_fused(fused)

        hits: list[Hit] = []
        per_doc: dict[str, int] = {}
        for row, score in ranked:
            doc = self.meta["doc_id"].iat[row]
            if max_per_doc:
                if per_doc.get(doc, 0) >= max_per_doc:
                    continue
                per_doc[doc] = per_doc.get(doc, 0) + 1
            hits.append(
                self._make_hit(
                    len(hits) + 1,
                    row,
                    score,
                    {
                        "bm25_score": bm_score.get(row),
                        "bm25_rank": bm_rank.get(row),
                        "dense_score": de_score.get(row),
                        "dense_rank": de_rank.get(row),
                        "aspect_boosted": row in boosted,
                    },
                )
            )
            if len(hits) >= k:
                break
        timings["fuse_format"] = (time.perf_counter() - t) * 1000
        timings["total"] = (time.perf_counter() - t_all) * 1000

        return RetrievalResult(
            query=query,
            mode=mode,
            search_text=search_text,
            parsed=parsed,
            filters=applied,
            hits=hits,
            n_candidates=None if rows is None else int(len(rows)),
            timings_ms={k_: round(v, 2) for k_, v in timings.items()},
        )

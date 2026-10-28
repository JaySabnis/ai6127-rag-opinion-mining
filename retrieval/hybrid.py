"""Hybrid fusion of the sparse and dense rankings.

Reciprocal Rank Fusion (Cormack, Clarke & Buettcher, SIGIR 2009):

    RRF(d) = sum over rankers r of  w_r / (k + rank_r(d))

RRF uses ranks, not raw scores, so BM25 scores (unbounded) and cosine
similarities (in [-1, 1]) never need to be calibrated against each other.
k = 60 damps the advantage of the very top ranks.

Optional aspect boost: a chunk whose ``aspect_tags`` contain an aspect named in
the query receives an extra  w_aspect / (k + 1), i.e. as if an "aspect ranker"
had put it at rank 1. It is a soft signal because the tags are keyword based.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from retrieval.config import RRF_K


def rrf_fuse(
    rankings: Mapping[str, Sequence[int]],
    k: int = RRF_K,
    weights: Mapping[str, float] | None = None,
) -> dict[int, float]:
    """Fuse ranked lists of row ids (best first) into {row_id: rrf_score}."""
    fused: dict[int, float] = {}
    for name, ranked in rankings.items():
        w = 1.0 if weights is None else weights.get(name, 1.0)
        for rank, row in enumerate(ranked, start=1):
            row = int(row)
            fused[row] = fused.get(row, 0.0) + w / (k + rank)
    return fused


def add_aspect_boost(
    fused: dict[int, float],
    aspect_tags: Sequence[Sequence[str]],
    query_aspects: Sequence[str],
    weight: float,
    k: int = RRF_K,
) -> set[int]:
    """Add the aspect bonus in place; returns the set of boosted rows."""
    if not query_aspects or weight <= 0:
        return set()
    wanted = set(query_aspects)
    bonus = weight / (k + 1)
    boosted = set()
    for row in fused:
        if wanted.intersection(aspect_tags[row]):
            fused[row] += bonus
            boosted.add(row)
    return boosted


def rank_fused(fused: Mapping[int, float]) -> list[tuple[int, float]]:
    """Sort by score desc, ties broken by row id for determinism."""
    return sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))


def ranks_of(row_ids: np.ndarray) -> dict[int, int]:
    return {int(r): i for i, r in enumerate(row_ids, start=1)}

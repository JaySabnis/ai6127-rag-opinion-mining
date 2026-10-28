"""Row-aligned corpus metadata store.

Both indexes refer to chunks by their *row number* in ``corpus.jsonl``:

    row i  ==  BM25 document i  ==  FAISS vector id i

This module reads the corpus once and saves the metadata (including text)
as a pandas pickle, so later stages never have to re-parse 459k JSON lines.
The ``chunk_id`` / ``doc_id`` values from the knowledge-base group are kept
unchanged, as their handoff document requires.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

META_COLUMNS = [
    "chunk_id",
    "doc_id",
    "chunk_index",
    "text",
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
]


def iter_corpus(path: str | Path) -> Iterator[dict]:
    """Stream records from a JSONL corpus (one chunk per line)."""
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}: malformed JSON on line {line_no}") from exc


def build_meta(corpus_path: str | Path, out_path: str | Path, max_records: int | None = None) -> pd.DataFrame:
    """Parse corpus.jsonl into a row-aligned DataFrame and save it."""
    rows = []
    for i, rec in enumerate(iter_corpus(corpus_path)):
        if max_records is not None and i >= max_records:
            break
        rows.append({col: rec.get(col) for col in META_COLUMNS})
    meta = pd.DataFrame(rows, columns=META_COLUMNS)

    if meta["chunk_id"].duplicated().any():
        raise ValueError("chunk_id values are not unique; the row-id mapping would be ambiguous")

    # Normalise types the retrieval code relies on.
    meta["text"] = meta["text"].fillna("").astype(str)
    meta["aspect_tags"] = meta["aspect_tags"].apply(lambda v: list(v) if isinstance(v, list) else [])
    for col in ("brand", "model"):
        meta[col] = meta[col].where(meta[col].notna() & (meta[col].astype(str).str.strip() != ""), None)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta.to_pickle(out_path)
    return meta


def load_meta(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; build it with: python -m scripts.build_retrieval_indexes --stage meta")
    return pd.read_pickle(path)


def rows_for(
    meta: pd.DataFrame,
    model: str | None = None,
    brand: str | None = None,
    min_rating: int | None = None,
    max_rating: int | None = None,
    exclude_renewed: bool = False,
) -> np.ndarray | None:
    """Return the row ids matching hard metadata filters, or None for 'no filter'.

    Aspect is deliberately NOT a hard filter here: the aspect tags are keyword
    based and noisy (e.g. "fast delivery" is tagged as performance), so the
    hybrid stage uses them as a soft boost instead.
    """
    mask = np.ones(len(meta), dtype=bool)
    active = False
    if model is not None:
        mask &= (meta["model"] == model).to_numpy()
        active = True
    if brand is not None:
        mask &= (meta["brand"] == brand).to_numpy()
        active = True
    if min_rating is not None:
        mask &= (meta["rating"] >= min_rating).to_numpy()
        active = True
    if max_rating is not None:
        mask &= (meta["rating"] <= max_rating).to_numpy()
        active = True
    if exclude_renewed:
        mask &= ~meta["is_renewed"].fillna(False).astype(bool).to_numpy()
        active = True
    return np.flatnonzero(mask) if active else None

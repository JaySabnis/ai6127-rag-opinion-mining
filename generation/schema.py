"""Input boundary shared with Group B's Hit.to_dict() and UI candidates."""
from __future__ import annotations

import re
import unicodedata

ASPECTS = ("camera", "battery", "display", "performance", "service", "carrier")
LABELS = ("positive", "negative", "mixed", "neutral")


def normal_text(text):
    return " ".join(unicodedata.normalize("NFC", text).split())


def quote_in(quote, text):
    return isinstance(quote, str) and bool(normal_text(quote)) and normal_text(quote) in normal_text(text)


def normalize_chunks(rows):
    if not isinstance(rows, list):
        raise ValueError("Evidence must be a list of Group B chunk dictionaries")
    out, seen = [], {}
    for raw in rows:
        if hasattr(raw, "to_dict"):
            raw = raw.to_dict()
        if not isinstance(raw, dict):
            raise ValueError("Each chunk must be an object")
        metadata = raw.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be an object")
        row = {**metadata, **raw}
        for key in ("chunk_id", "doc_id", "text"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"Chunk requires nonempty {key}")
        if not re.fullmatch(r"[A-Za-z0-9_.:-]+", row["chunk_id"]):
            raise ValueError("chunk_id must be safe for unambiguous bracket citations")
        for key in ("brand", "model"):
            if row.get(key) is not None and not isinstance(row[key], str):
                raise ValueError(f"{key} must be text or null")
        cid = row["chunk_id"]
        if cid in seen:
            if any(row.get(k) != seen[cid].get(k) for k in ("doc_id", "text", "brand", "model")):
                raise ValueError("Conflicting duplicate chunk_id: " + cid)
            continue
        seen[cid] = row
        out.append(row)
    return out


def product_key(row):
    return tuple((row.get(k) or "").strip().casefold() for k in ("brand", "model"))

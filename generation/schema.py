"""Input boundary shared with Group B's Hit.to_dict() and UI candidates."""
from __future__ import annotations

import re
import unicodedata
from retrieval.product_identity import make_product_id, variant_key

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
        row["product_id"] = product_metadata(row)["product_id"]
        cid = row["chunk_id"]
        if cid in seen:
            if any(row.get(k) != seen[cid].get(k) for k in ("doc_id", "text", "brand", "model", "product_id")):
                raise ValueError("Conflicting duplicate chunk_id: " + cid)
            old_canonical, new_canonical = seen[cid].get("canonical_model"), row.get("canonical_model")
            if old_canonical and new_canonical and variant_key(old_canonical, row.get("brand")) != variant_key(new_canonical, row.get("brand")):
                raise ValueError("Conflicting canonical model for duplicate chunk_id: " + cid)
            if new_canonical and not old_canonical:
                seen[cid]["canonical_model"] = new_canonical
            continue
        seen[cid] = row
        out.append(row)
    return out


def product_metadata(row):
    """Prefer supplied catalogue IDs; infer old-file IDs with the same rules.

    The original model is never overwritten. An explicit canonical_model is
    display metadata from the catalogue, while product_id is the grouping key.
    """
    for key in ("brand", "model", "canonical_model", "product_id"):
        if row.get(key) is not None and not isinstance(row[key], str):
            raise ValueError(f"{key} must be text or null")
    for key in ("canonical_model", "product_id"):
        if row.get(key) is not None and not row[key].strip():
            raise ValueError(f"{key} must be nonempty when provided")
    canonical_model = row.get("canonical_model")
    identity = row.get("product_id") or make_product_id(
        row.get("brand"), canonical_model or row.get("model"), doc_id=row.get("doc_id"))
    return {"product_id": identity, "canonical_model": canonical_model}


def product_key(row):
    return (product_metadata(row)["product_id"],)

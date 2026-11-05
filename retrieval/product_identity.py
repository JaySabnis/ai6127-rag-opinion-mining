"""Catalogue identity rules shared with generation; no model/data dependencies.

Keep model tokens, especially Plus, FE, Ultra and 5G. This is the catalogue's
existing spelling normalization, not fuzzy matching or suffix removal.
"""
from __future__ import annotations

import hashlib
import json
import re

_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_LETTER_DIGIT = re.compile(r"(?<=[a-z])(?=\d)|(?<=\d)(?=[a-z])")


def model_tokens(text: str) -> tuple[str, ...]:
    """Tokenize spellings while keeping S10, S10 Plus and S10 5G distinct."""
    text = _NON_ALNUM.sub(" ", text.lower().replace("+", " plus "))
    return tuple(_LETTER_DIGIT.sub(" ", text).split())


def variant_key(model: str, brand: str | None) -> tuple:
    """Brand-scoped identity of catalogue spellings (Note9 / Note 9, etc.)."""
    brand_tokens = model_tokens(brand) if brand else ()
    tokens = model_tokens(model)
    if brand_tokens and tokens[:len(brand_tokens)] == brand_tokens and len(tokens) > len(brand_tokens):
        tokens = tokens[len(brand_tokens):]
    # Galaxy is a Samsung family prefix, not a removable word for every brand.
    if brand_tokens == ("samsung",) and tokens[:1] == ("galaxy",) and len(tokens) > 1:
        tokens = tokens[1:]
    return brand_tokens, tokens


def make_product_id(brand: str | None, model: str | None, *, doc_id: str | None = None) -> str:
    """Stable ID independent of which spelling is most frequent in a corpus.

    An unknown model cannot establish that two source reviews refer to the same
    phone. Scope that fallback to the source review, even when brands match.
    """
    key = variant_key(model or "", brand)
    if not key[1]:
        if not isinstance(doc_id, str) or not doc_id.strip():
            raise ValueError("Unknown product model requires doc_id or an explicit product_id")
        key = ("unknown_model", key[0], doc_id)
    raw = json.dumps(key, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return "phone-v1:" + hashlib.sha256(raw).hexdigest()

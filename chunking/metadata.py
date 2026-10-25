"""
Metadata extraction for review chunks.

Task 2 metadata:
- chunk_id
- doc_id
- brand
- model
- rating
- timestamp
- verified
- helpful_votes
- is_renewed
- aspect_tags
- mentions_other_model
- source_url
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any


# ---------------------------------------------------------------------------
# Aspect keywords
# ---------------------------------------------------------------------------

ASPECT_KEYWORDS = {
    "camera": [
        "camera",
        "cameras",
        "photo",
        "photos",
        "picture",
        "pictures",
        "photography",
        "selfie",
        "selfies",
        "video",
        "videos",
        "lens",
        "lenses",
        "megapixel",
        "megapixels",
        "flash",
    ],
    "battery": [
        "battery",
        "battery life",
        "charging",
        "charge",
        "charger",
        "power",
        "drain",
        "drains",
        "battery life",
    ],
    "display": [
        "display",
        "screen",
        "screens",
        "resolution",
        "brightness",
        "amoled",
        "oled",
        "lcd",
        "touchscreen",
        "touch screen",
    ],
    "performance": [
        "performance",
        "processor",
        "cpu",
        "gpu",
        "ram",
        "speed",
        "fast",
        "faster",
        "slow",
        "slower",
        "lag",
        "laggy",
        "gaming",
        "games",
        "multitasking",
    ],
    "service": [
        "service",
        "customer service",
        "customer support",
        "support",
        "repair",
        "repairs",
        "warranty",
        "replacement",
    ],
    "carrier": [
        "carrier",
        "network",
        "signal",
        "reception",
        "at&t",
        "verizon",
        "t-mobile",
        "sprint",
        "5g",
        "4g",
        "lte",
        "gsm",
        "cdma",
    ],
}


# ---------------------------------------------------------------------------
# Precompiled regexes
# ---------------------------------------------------------------------------

_WORD_BOUNDARY = r"(?<!\w)(?:{term})(?!\w)"

_DATE_ISO_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}"
)


# Compile aspect patterns ONCE.
#
# Previously this was done inside:
#
#   detect_aspects()
#
# which means every chunk repeatedly constructed regex strings.
_ASPECT_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    aspect: [
        re.compile(
            _WORD_BOUNDARY.format(term=re.escape(keyword)),
            re.IGNORECASE,
        )
        for keyword in keywords
    ]
    for aspect, keywords in ASPECT_KEYWORDS.items()
}


# ---------------------------------------------------------------------------
# Model detector cache
# ---------------------------------------------------------------------------

# The model catalog is created once by the pipeline and then reused for
# hundreds of thousands of chunks.
#
# We cache using id(model_catalog) so we do NOT construct a giant tuple/hash
# on every chunk.
_MODEL_DETECTOR_CACHE: dict[int, tuple[
    re.Pattern[str],
    dict[str, set[str]],
]] = {}


def _normalise(value: Any) -> str:
    if value is None:
        return ""

    return str(value).strip()


def parse_date(value: Any) -> str | None:
    """
    Convert common date formats to YYYY-MM-DD.

    Returns None when parsing fails.
    """
    if value is None:
        return None

    value = str(value).strip()

    if not value:
        return None

    # Already ISO-like.
    match = _DATE_ISO_RE.match(value)

    if match:
        return match.group(0)

    formats = (
        "%B %d, %Y",
        "%b %d, %Y",
        "%Y/%m/%d",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%Y-%m-%d",
    )

    for fmt in formats:
        try:
            return datetime.strptime(value, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue

    return value


def detect_aspects(text: str) -> list[str]:
    """
    Detect one or more aspects discussed by a chunk.

    Returns aspects in a stable order.
    """
    if not text:
        return []

    detected: list[str] = []

    # All regexes were compiled once at module load time.
    for aspect, patterns in _ASPECT_PATTERNS.items():
        for pattern in patterns:
            if pattern.search(text):
                detected.append(aspect)
                break

    return detected


def build_model_catalog(
    reviews: list[dict[str, Any]],
) -> list[tuple[str, str]]:
    """
    Build a list of (brand, model) combinations from the cleaned corpus.

    This catalog is used to detect references to another phone model.
    """
    catalog: set[tuple[str, str]] = set()

    for review in reviews:
        brand = _normalise(review.get("brand"))
        model = _normalise(review.get("model"))

        if model:
            catalog.add((brand, model))

    # Longest model names first to reduce partial matching problems.
    return sorted(
        catalog,
        key=lambda item: len(item[1]),
        reverse=True,
    )


def _build_model_detector(
    model_catalog: list[tuple[str, str]],
) -> tuple[
    re.Pattern[str],
    dict[str, set[str]],
]:
    """
    Build one compiled regex for all phone models.

    Returns:
        pattern:
            One compiled regex containing all known models.

        model_brands:
            Mapping:
                normalized model -> set of brands

            This lets us preserve the original behavior when the same
            model name exists under multiple brands.
    """
    model_brands: dict[str, set[str]] = {}

    for brand, model in model_catalog:
        brand_lower = brand.strip().lower()
        model_lower = model.strip().lower()

        if not model_lower:
            continue

        model_brands.setdefault(model_lower, set()).add(
            brand_lower
        )

    models = sorted(
        model_brands.keys(),
        key=len,
        reverse=True,
    )

    if not models:
        # Never matches.
        pattern = re.compile(r"(?!)")
    else:
        alternatives = "|".join(
            re.escape(model)
            for model in models
        )

        pattern = re.compile(
            rf"(?<!\w)(?:{alternatives})(?!\w)",
            re.IGNORECASE,
        )

    return pattern, model_brands


def _get_model_detector(
    model_catalog: list[tuple[str, str]],
) -> tuple[
    re.Pattern[str],
    dict[str, set[str]],
]:
    """
    Return the cached model detector.

    The detector is built only once for the model catalog.
    """
    cache_key = id(model_catalog)

    detector = _MODEL_DETECTOR_CACHE.get(cache_key)

    if detector is None:
        detector = _build_model_detector(model_catalog)
        _MODEL_DETECTOR_CACHE[cache_key] = detector

    return detector


def detect_other_model(
    text: str,
    current_brand: str,
    current_model: str,
    model_catalog: list[tuple[str, str]],
) -> bool:
    """
    Detect whether a chunk mentions a phone model other than
    its own product model.

    This is optimized for large corpora:
    - One regex is compiled for the entire model catalog.
    - The regex is cached and reused across all chunks.
    - finditer() identifies all model mentions in the chunk.
    """
    if not text:
        return False

    current_model_lower = current_model.strip().lower()
    current_brand_lower = current_brand.strip().lower()

    pattern, model_brands = _get_model_detector(
        model_catalog
    )

    for match in pattern.finditer(text):
        mentioned_model = match.group(0).strip().lower()

        # If the mentioned model name is different, it is definitely
        # another model.
        if mentioned_model != current_model_lower:
            return True

        # Same model name:
        #
        # If that model exists under another brand, the original logic
        # would consider it an "other model" because the brand/model
        # combination is different.
        brands = model_brands.get(
            mentioned_model,
            set(),
        )

        if any(
            brand != current_brand_lower
            for brand in brands
        ):
            return True

    return False


def build_metadata(
    review: dict[str, Any],
    chunk: dict[str, Any],
    model_catalog: list[tuple[str, str]],
) -> dict[str, Any]:
    """
    Create the final metadata object for one chunk.
    """
    brand = _normalise(review.get("brand"))
    model = _normalise(review.get("model"))

    chunk_text = chunk["text"]

    metadata = {
        "chunk_id": chunk["chunk_id"],
        "doc_id": chunk["doc_id"],
        "chunk_index": chunk["chunk_index"],
        "text": chunk_text,
        "chunk_word_count": chunk["chunk_word_count"],

        "brand": brand,
        "model": model,

        "rating": review.get("rating"),

        "timestamp": parse_date(
            review.get("date")
            or review.get("timestamp")
        ),

        "verified": bool(
            review.get("verified_purchase", False)
        ),

        "helpful_votes": int(
            review.get("helpful_vote")
            or review.get("helpful_votes")
            or 0
        ),

        "is_renewed": bool(
            review.get("is_renewed", False)
        ),

        "aspect_tags": detect_aspects(
            chunk_text
        ),

        "mentions_other_model": detect_other_model(
            text=chunk_text,
            current_brand=brand,
            current_model=model,
            model_catalog=model_catalog,
        ),

        "source_url": _normalise(
            review.get("source_url")
        ),
    }

    return metadata
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
from dataclasses import dataclass
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
# Phone model detection
# ---------------------------------------------------------------------------

_Tokens = tuple[str, ...]

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# "(Gen 4)", "(2020)", "(Unlocked)" are listing details, not part of the
# name reviewers write.
_PARENTHESES_RE = re.compile(r"\([^)]*\)")

# Reviewers call "Moto G (Gen 4)" the "Moto G4".
_GENERATION_RE = re.compile(
    r"\(\s*gen\s*(\d+)\s*\)",
    re.IGNORECASE,
)

# Tokens that contain a digit but are not a model:
# 4g, 5g, 3x, 4k, 3d, 1080p, 64gb, 12mp, 5000mah, 18w, 2nd, mp3, ip68 ...
_SPEC_TOKEN_RE = re.compile(
    r"^(?:"
    r"[2-5]g"
    r"|\d+(?:x|k|d|p|gb|mb|tb|mp|mah|hz|ghz|mm|w|st|nd|rd|th)"
    r"|mp[34]"
    r"|ip6\d"
    r")$"
)

# Brand values that are placeholders rather than manufacturers.
_PLACEHOLDER_BRANDS = frozenset({
    "",
    "other",
    "generic",
    "unknown",
    "unbranded",
    "artist unknown",
})

# Product-title words that say nothing about which model it is.
_GENERIC_TOKENS = frozenset({
    "phone",
    "phones",
    "cell",
    "cellphone",
    "mobile",
    "smartphone",
    "tablet",
    "unlocked",
    "dual",
    "sim",
    "lte",
    "band",
    "inch",
    "inches",
    "year",
    "years",
    "net10",
})


@dataclass(frozen=True)
class ModelCatalog:
    """
    Index of known phone models, built once and reused for every chunk.

    alias_brands:
        Model name as tokens -> brands that sell a model with that name.

    distinctive:
        Aliases specific enough to match without a brand in front of them.
        Every other alias only counts when written as "<brand> <alias>".
    """

    alias_brands: dict[_Tokens, set[_Tokens]]
    distinctive: frozenset[_Tokens]
    brands: frozenset[_Tokens]
    max_alias_length: int
    max_brand_length: int

    def __len__(self) -> int:
        return len(self.alias_brands)


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


def _tokens(value: Any) -> _Tokens:
    return tuple(
        _TOKEN_RE.findall(_normalise(value).lower())
    )


def _brand_tokens(brand: Any) -> _Tokens:
    """
    Brand as tokens, or () when the brand is only a placeholder.
    """
    if _normalise(brand).lower() in _PLACEHOLDER_BRANDS:
        return ()

    return _tokens(brand)


def _model_aliases(
    brand: _Tokens,
    model: Any,
) -> list[_Tokens]:
    """
    Token sequences reviewers use to refer to a catalog model.

    Examples:
        Motorola / "Motorola Razr"       -> ("razr",)
        Motorola / "Moto G Plus (Gen 5)" -> ("moto", "g", "plus"),
                                            ("moto", "g5", "plus")
    """
    model = _normalise(model)

    alias = _tokens(
        _PARENTHESES_RE.sub(" ", model)
    )

    # The brand is matched on its own, so drop it when the model repeats it.
    if (
        brand
        and len(alias) > len(brand)
        and alias[:len(brand)] == brand
    ):
        alias = alias[len(brand):]

    if not alias:
        return []

    aliases = [alias]

    generation = _GENERATION_RE.search(model)

    if generation:
        letters = [
            index
            for index, token in enumerate(alias)
            if len(token) == 1 and token.isalpha()
        ]

        if letters:
            index = letters[-1]
            aliases.append(
                alias[:index]
                + (alias[index] + generation.group(1),)
                + alias[index + 1:]
            )

    return aliases


def _is_distinctive(alias: _Tokens) -> bool:
    """
    Whether an alias can be matched without a brand in front of it.

    Model identifiers mix letters and digits ("iPhone 6", "Galaxy S5",
    "G7", "T68i"). Bare words ("Phone", "One", "Go", "Razr") and bare
    numbers ("2", "3310") are ordinary review vocabulary, so those only
    count when brand-qualified ("Razer Phone", "HTC One", "Nokia 3310").
    """
    if not any(
        char.isdigit()
        for token in alias
        for char in token
    ):
        return False

    return any(
        token not in _GENERIC_TOKENS
        and not _SPEC_TOKEN_RE.match(token)
        and any(char.isalpha() for char in token)
        for token in alias
    )


def _contains(
    sequence: _Tokens,
    part: _Tokens,
) -> bool:
    length = len(part)

    return any(
        sequence[index:index + length] == part
        for index in range(len(sequence) - length + 1)
    )


def build_model_catalog(
    reviews: list[dict[str, Any]],
) -> ModelCatalog:
    """
    Build the phone model index from the cleaned corpus.

    This catalog is used to detect references to another phone model.
    """
    alias_brands: dict[_Tokens, set[_Tokens]] = {}

    for review in reviews:
        brand = _brand_tokens(review.get("brand"))

        # A placeholder brand cannot qualify a mention, and its "models"
        # are mostly product-title fragments.
        if not brand:
            continue

        for alias in _model_aliases(brand, review.get("model")):
            alias_brands.setdefault(alias, set()).add(brand)

            # Reviewers shorten "Galaxy S3" to "S3" and "Moto G7 Power"
            # to "G7", so model codes are aliases on their own too.
            if len(alias) > 1:
                for token in alias:
                    if _is_distinctive((token,)):
                        alias_brands.setdefault((token,), set()).add(brand)

    brands = frozenset(
        brand
        for alias_brand_set in alias_brands.values()
        for brand in alias_brand_set
    )

    return ModelCatalog(
        alias_brands=alias_brands,
        distinctive=frozenset(
            alias
            for alias in alias_brands
            if _is_distinctive(alias)
        ),
        brands=brands,
        max_alias_length=max(map(len, alias_brands), default=0),
        max_brand_length=max(map(len, brands), default=0),
    )


def _preceding_brand(
    tokens: list[str],
    end: int,
    model_catalog: ModelCatalog,
) -> _Tokens | None:
    """
    Return the known brand written right before tokens[end], if any.
    """
    for length in range(
        min(model_catalog.max_brand_length, end),
        0,
        -1,
    ):
        brand = tuple(tokens[end - length:end])

        if brand in model_catalog.brands:
            return brand

    return None


def _find_model_mentions(
    tokens: list[str],
    model_catalog: ModelCatalog,
):
    """
    Yield (alias, brand) for each catalog model mentioned in the tokens.

    brand is the brand written right before the alias ("LG G6"), or None.

    The longest alias wins, so "Galaxy S10 Plus" is not also read as
    "Galaxy S10". Non-distinctive aliases are only yielded when one of
    their own brands precedes them.
    """
    index = 0

    while index < len(tokens):
        for length in range(
            min(model_catalog.max_alias_length, len(tokens) - index),
            0,
            -1,
        ):
            alias = tuple(tokens[index:index + length])

            alias_brands = model_catalog.alias_brands.get(alias)

            if alias_brands is None:
                continue

            brand = _preceding_brand(tokens, index, model_catalog)

            if (
                alias in model_catalog.distinctive
                or brand in alias_brands
            ):
                yield alias, brand
                index += length
                break
        else:
            index += 1


def detect_other_model(
    text: str,
    current_brand: str,
    current_model: str,
    model_catalog: ModelCatalog,
) -> bool:
    """
    Detect whether a chunk mentions a phone model other than
    its own product model.

    A mention counts as "other" when:
    - it is preceded by a different brand ("LG G6" in a Moto G6 review), or
    - it is a distinctive model name that is not part of the reviewed
      model ("Galaxy S4" in a Galaxy S5 review).

    "S5" or "Galaxy" in a Galaxy S5 review, and same-brand nicknames such
    as "Essential Phone" in a PH-1 review, are treated as the same phone.
    """
    if not text:
        return False

    brand = _brand_tokens(current_brand)
    current_aliases = _model_aliases(brand, current_model)

    tokens = _TOKEN_RE.findall(text.lower())

    for alias, mentioned_brand in _find_model_mentions(
        tokens,
        model_catalog,
    ):
        # "LG G6" in a Moto G6 review. A carrier written before the model
        # ("AT&T iPhone 6") does not sell that model, so it falls through.
        if (
            brand
            and mentioned_brand
            and mentioned_brand != brand
            and mentioned_brand in model_catalog.alias_brands[alias]
        ):
            return True

        if alias not in model_catalog.distinctive:
            continue

        if current_aliases:
            if not any(
                _contains(current_alias, alias)
                for current_alias in current_aliases
            ):
                return True

        # Unknown reviewed model: only a model this brand does not sell
        # is certainly a different phone.
        elif brand and brand not in model_catalog.alias_brands[alias]:
            return True

    return False


def build_metadata(
    review: dict[str, Any],
    chunk: dict[str, Any],
    model_catalog: ModelCatalog,
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
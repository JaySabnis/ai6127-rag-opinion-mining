"""Extract the product and aspect signals from a user query.

    "How is the camera on the Galaxy S10?"  ->  model="Galaxy S10", aspects=["camera"]

* Aspects use the knowledge-base group's own ``detect_aspects`` so the query
  side and the chunk side share one keyword dictionary.
* Models are matched against the catalogue of ``model`` values that actually
  occur in the corpus, on token boundaries, preferring the longest match
  ("iPhone 11 Pro" wins over "iPhone 11"). "+" is read as "Plus" ("S10+").
* ``resolve_model`` / ``resolve_brand`` map product names produced elsewhere
  (e.g. the DeepSeek query rewrite) onto names that exist in the corpus, so a
  filter never silently matches nothing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from chunking.metadata import detect_aspects

_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_LETTER_DIGIT = re.compile(r"(?<=[a-z])(?=\d)|(?<=\d)(?=[a-z])")


def model_tokens(text: str) -> tuple[str, ...]:
    """Lowercase, drop punctuation, split letter/digit runs: 'iPhone15-Pro' -> (iphone, 15, pro).

    '+' becomes the word 'plus', so "S10+" and "S10 Plus" give the same tokens.
    """
    t = _NON_ALNUM.sub(" ", text.lower().replace("+", " plus "))
    t = _LETTER_DIGIT.sub(" ", t)
    return tuple(t.split())


def variant_key(model: str, brand: str | None) -> tuple:
    """Key shared by spellings of the same phone within one brand.

    Ignores case, spacing and punctuation ("Galaxy Note9" = "Galaxy Note 9",
    "G8X Thinq™" = "G8X Thinq"), a leading brand name ("Motorola Razr" = "Razr")
    and Samsung's leading "Galaxy" ("Galaxy S20 FE" = "S20 FE"). Checked by hand
    on all 85 groups this forms in the corpus catalogue.
    """
    btoks = model_tokens(brand) if brand else ()
    toks = model_tokens(model)
    if btoks and toks[: len(btoks)] == btoks and len(toks) > len(btoks):
        toks = toks[len(btoks):]
    if toks and toks[0] == "galaxy" and len(toks) > 1:
        toks = toks[1:]
    return (btoks, toks)


@dataclass
class ParsedQuery:
    text: str
    model: str | None = None
    brand: str | None = None
    aspects: list[str] = field(default_factory=list)
    without_model: str | None = None  # query text with the model mention removed

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "model": self.model,
            "brand": self.brand,
            "aspects": self.aspects,
            "without_model": self.without_model,
        }


# Spec units ("128GB", "48MP", "5000mAh", "120Hz", "6.5 inch") never count as model words.
_UNITS = frozenset({"gb", "tb", "mb", "mp", "mah", "hz", "ghz", "inch", "mm"})
# Single letters are a unit after a number ("5G", "4K") but a series name
# before one ("G2", "K30", "X").
_UNIT_LETTERS = frozenset({"g", "k", "w", "x"})


def _model_words(toks: tuple[str, ...]) -> list[str]:
    out = []
    for i, t in enumerate(toks):
        if not t.isalpha() or t in _UNITS:
            continue
        if t in _UNIT_LETTERS and i > 0 and toks[i - 1].isdigit():
            continue
        out.append(t)
    return out


def _distinctive(toks: tuple[str, ...]) -> bool:
    """Can these tokens identify a model on their own, without the brand name?

    Yes for "galaxy s 10", "iphone 11", "g 6", "motorola razr" (a letter token
    plus a number, or at least two word tokens). No for single words or bare
    numbers, because the real catalogue contains names such as "Pre", "Rugged",
    "6.1" and "3310" that would otherwise fire on ordinary queries
    ("pre-installed apps", "6.1 inch screen").
    """
    words = _model_words(toks)  # "5 g", "128 gb", "mah battery" -> spec, not a model
    digit = sum(t.isdigit() for t in toks)
    return (len(words) >= 1 and digit >= 1) or len(words) >= 2


class QueryParser:
    """Match model names in queries.

    Patterns per model (best one wins on a clash):
      1. brand + model  ("nokia 3310", "palm pre", "lg g 6")  - always allowed
      2. model alone    ("galaxy s 10", "g 6")                - only if distinctive
      3. model without its family word ("s 10" for "Galaxy S10") - only if
         distinctive, >= 2 tokens, and not claimed by another model.
    On a clash a full name beats a shortened alias, then the more frequent
    model wins. Matching prefers the longest pattern found in the query.
    """

    def __init__(self, meta: pd.DataFrame):
        models = meta.loc[meta["model"].notna(), ["model", "brand"]]
        counts = models["model"].value_counts().to_dict()
        # Most frequent brand per model (models are brand-specific in practice).
        self.model_brand: dict[str, str | None] = (
            models.groupby("model")["brand"].agg(lambda s: s.mode().iat[0] if not s.mode().empty else None).to_dict()
        )
        # The corpus spells some phones several ways ("Galaxy Note 9" / "Galaxy Note9",
        # "Motorola Razr" / "Razr", "Galaxy S20 FE" / "S20 FE"). Group them so a
        # filter covers all of a phone's reviews; the most frequent spelling is
        # the canonical name returned by detect_model() / resolve_model().
        groups: dict[tuple, list[str]] = {}
        for model, brand in self.model_brand.items():
            groups.setdefault(variant_key(model, brand), []).append(model)
        self.canonical: dict[str, str] = {}
        self._variants: dict[str, list[str]] = {}
        for names in groups.values():
            names.sort(key=lambda m: (-counts.get(m, 0), m))
            self._variants[names[0]] = names
            for m in names:
                self.canonical[m] = names[0]

        best: dict[tuple[str, ...], tuple[tuple[int, int], str]] = {}

        def add(toks: tuple[str, ...], model: str, is_full: bool) -> None:
            prio = (int(is_full), counts.get(model, 0))
            if toks and (toks not in best or prio > best[toks][0]):
                best[toks] = (prio, model)

        for model, brand in self.model_brand.items():
            toks = model_tokens(model)
            if not toks:
                continue
            btoks = model_tokens(brand) if brand else ()
            if btoks and toks[: len(btoks)] != btoks:
                add(btoks + toks, model, True)
            if _distinctive(toks):
                add(toks, model, True)
            rest = toks[1:]
            if len(rest) >= 2 and _distinctive(rest):
                add(rest, model, False)

        # Index patterns by their first token for fast lookup, and keep each
        # model's own patterns so its mentions can be removed from a query.
        self._by_first: dict[str, list[tuple[tuple[str, ...], str]]] = {}
        self._patterns_of: dict[str, list[tuple[str, ...]]] = {}
        for toks, (_, model) in best.items():
            canon = self.canonical[model]
            self._by_first.setdefault(toks[0], []).append((toks, canon))
            self._patterns_of.setdefault(canon, []).append(toks)
        for pats in self._by_first.values():
            pats.sort(key=lambda p: len(p[0]), reverse=True)
        for pats in self._patterns_of.values():
            pats.sort(key=len, reverse=True)

        # Exact lookup for names coming from elsewhere (LLM output): the full
        # model name, ignoring case, spacing and punctuation.
        self._exact: dict[tuple[str, ...], str] = {}
        for model in sorted(self.model_brand, key=lambda m: -counts.get(m, 0)):
            self._exact.setdefault(model_tokens(model), self.canonical[model])
        brands = meta.loc[meta["brand"].notna(), "brand"].value_counts()
        self._brands: dict[tuple[str, ...], str] = {}
        for brand in brands.index:  # most frequent spelling first
            self._brands.setdefault(model_tokens(str(brand)), brand)

    # ------------------------------------------------------- external names
    def resolve_model(self, brand: str | None, model: str | None) -> str | None:
        """Map a product named elsewhere (e.g. by the LLM) to a model in the corpus.

        1. exact name, ignoring case/spacing/punctuation ("galaxy s10" -> "Galaxy S10")
        2. otherwise the same pattern matching used for queries, on "brand model"
           ("Samsung", "Galaxy S10+") -> "Galaxy S10 Plus"
        3. otherwise None: the caller should not filter by model.
        """
        if not model:
            return None
        hit = self._exact.get(model_tokens(model))
        if hit:
            return hit
        return self.detect_model(" ".join(x for x in (brand, model) if x))

    def resolve_brand(self, brand: str | None) -> str | None:
        """Map a brand named elsewhere to the corpus spelling ("samsung" -> "Samsung"), else None."""
        return self._brands.get(model_tokens(brand)) if brand else None

    def canonical_of(self, model: str) -> str:
        """Canonical spelling of a corpus model ("Razr" -> "Motorola Razr"); unknown names unchanged."""
        return self.canonical.get(model, model)

    def variants_of(self, model: str) -> list[str]:
        """All corpus spellings of the same phone, canonical first ("Razr" -> ["Motorola Razr", "Razr"])."""
        return list(self._variants.get(self.canonical_of(model), [model]))

    def remove_model(self, query: str, model: str) -> str:
        """Remove every mention of ``model`` (any spelling: full name, brand + name, short alias)."""
        out = query
        for toks in self._patterns_of.get(self.canonical_of(model), ()):
            out = self.remove_mention(out, toks)
        return out

    def _match(self, query: str) -> tuple[str, tuple[str, ...]] | None:
        q = model_tokens(query)
        found: tuple[tuple[str, ...], str] | None = None
        for i, tok in enumerate(q):
            for toks, model in self._by_first.get(tok, ()):
                if q[i : i + len(toks)] == toks:
                    if found is None or len(toks) > len(found[0]):
                        found = (toks, model)
                    break  # patterns are sorted longest first
        return (found[1], found[0]) if found else None

    def detect_model(self, query: str) -> str | None:
        m = self._match(query)
        return m[0] if m else None

    @staticmethod
    def remove_mention(query: str, toks: tuple[str, ...]) -> str:
        """Delete the matched model mention from the query text.

        ("galaxy","s","10") also matches "Galaxy S10", "galaxy-s10", "GALAXY S 10";
        the token "plus" also matches "+".
        """
        body = r"[\W_]*".join(r"(?:plus|\+)" if t == "plus" else re.escape(t) for t in toks)
        pattern = rf"(?<![a-z0-9]){body}(?![a-z0-9])"
        out = re.sub(pattern, " ", query, flags=re.IGNORECASE)
        return re.sub(r"\s+", " ", out).strip()

    def parse(self, query: str) -> ParsedQuery:
        m = self._match(query)
        model = m[0] if m else None
        without_model = self.remove_mention(query, m[1]) if m else None
        return ParsedQuery(
            text=query,
            model=model,
            brand=self.model_brand.get(model) if model else None,
            # Detect aspects without the model name: "Moto G *Power*" is not about the battery.
            aspects=detect_aspects(without_model if m else query),
            without_model=without_model,
        )

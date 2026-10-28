"""Tokenisation shared by BM25 indexing and querying.

Lowercase -> alphanumeric tokens -> drop English stopwords -> optional stemming.
Model names survive intact ("s10", "5g", "11"), which matters for exact
product matching.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Callable

_TOKEN = re.compile(r"[a-z0-9]+")

# Standard English stopword list (subset of the NLTK / Lucene lists).
STOPWORDS = frozenset(
    """a about above after again against all am an and any are as at be because been before being below
    between both but by can could did do does doing down during each few for from further had has have
    having he her here hers herself him himself his how i if in into is it its itself just me more most
    my myself nor of off on once only or other our ours ourselves out over own same she should so some
    such than that the their theirs them themselves then there these they this those through to too
    under until up very was we were what when where which while who whom why will with would you your
    yours yourself yourselves s t d ll m o re ve y""".split()
)


def make_stemmer(name: str = "auto") -> tuple[str, Callable[[str], str] | None]:
    """Return (stemmer_name, stem_fn). Tries PyStemmer, then NLTK, else no stemming."""
    if name in ("none", None):
        return "none", None
    if name in ("auto", "pystemmer"):
        try:
            import Stemmer  # PyStemmer

            s = Stemmer.Stemmer("english")
            return "pystemmer-english", s.stemWord
        except ImportError:
            if name == "pystemmer":
                raise
    if name in ("auto", "nltk"):
        try:
            from nltk.stem.snowball import SnowballStemmer

            return "nltk-snowball-english", SnowballStemmer("english").stem
        except ImportError:
            if name == "nltk":
                raise
    return "none", None


class Tokenizer:
    def __init__(self, stemmer: str = "auto"):
        self.stemmer_name, stem = make_stemmer(stemmer)
        # ~100k distinct word types in the corpus -> cache every stem once.
        self._stem = lru_cache(maxsize=1_000_000)(stem) if stem else None

    def __call__(self, text: str) -> list[str]:
        toks = [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]
        if self._stem is not None:
            toks = [self._stem(t) for t in toks]
        return toks

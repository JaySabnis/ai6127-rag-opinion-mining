"""Unit tests for the retrieval module (no GPU, no model download needed).

The dense tests use a tiny hashing encoder and the NumPy backend, so they test
the plumbing (sharding, ids, filters, fusion), not embedding quality.
Real FAISS + BGE are covered by scripts/smoke_test_retrieval.py.

    uv run python -m unittest discover -s tests -v   (all tests, incl. reranker/UI)
    uv run python tests/test_retrieval.py            (only these)
"""

from __future__ import annotations

import hashlib
import math
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from retrieval import Retriever  # noqa: E402
from retrieval.bm25 import BM25Index  # noqa: E402
from retrieval.config import IndexPaths  # noqa: E402
from retrieval.corpus import build_meta, rows_for  # noqa: E402
from retrieval.dense import DenseIndex, encode_corpus  # noqa: E402
from retrieval.hybrid import add_aspect_boost, rrf_fuse  # noqa: E402
from retrieval.query_parser import QueryParser  # noqa: E402
from retrieval.text import Tokenizer  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "mini_corpus.jsonl"


class HashEncoder:
    """Deterministic bag-of-words hashing encoder (test double for BGE)."""

    name = "test-hash-encoder"
    dim = 64

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in Tokenizer("none")(text):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1
        n = np.linalg.norm(v)
        return v / n if n else v

    def encode_documents(self, texts):
        return np.stack([self._vec(t) for t in texts])

    def encode_queries(self, texts):
        return self.encode_documents(texts)


def _build(tmp: str, shard_size: int = 5) -> Retriever:
    paths = IndexPaths(tmp)
    meta = build_meta(FIXTURE, paths.meta)
    BM25Index.build(meta["text"].tolist(), stemmer="none", log_every=0).save(paths.bm25_dir)
    enc = HashEncoder()
    encode_corpus(meta["text"].tolist(), enc, paths.dense_dir, shard_size=shard_size)
    DenseIndex.build(paths.dense_dir, use_faiss=False)
    return Retriever.load(tmp, encoder=enc)


# ------------------------------------------------------------------ BM25
def test_bm25_matches_textbook_formula():
    meta = build_meta(FIXTURE, Path(tempfile.mkdtemp()) / "m.pkl")
    texts = meta["text"].tolist()
    idx = BM25Index.build(texts, stemmer="none", log_every=0)
    tok = Tokenizer("none")
    docs = [tok(t) for t in texts]
    n, avgdl = len(docs), sum(map(len, docs)) / len(docs)
    for q in ["battery drains fast", "camera photos night", "screen resolution low price"]:
        expected = []
        for d in docs:
            c, s = Counter(d), 0.0
            for t, qn in Counter(tok(q)).items():
                df = sum(t in x for x in docs)
                if df:
                    idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                    s += qn * idf * c[t] * 2.2 / (c[t] + 1.2 * (0.25 + 0.75 * len(d) / avgdl))
            expected.append(s)
        assert np.allclose(idx.score_all(q), expected, atol=1e-4), q


def test_bm25_subset_and_unknown_terms():
    meta = build_meta(FIXTURE, Path(tempfile.mkdtemp()) / "m.pkl")
    idx = BM25Index.build(meta["text"].tolist(), stemmer="none", log_every=0)
    rows = rows_for(meta, model="Galaxy S10")
    got, _ = idx.search("battery charge", 10, rows)
    assert set(got) <= set(rows) and len(got) > 0
    got, scores = idx.search("qwertyuiop", 10)
    assert got.size == 0 and scores.size == 0


# ------------------------------------------------------------ query parser
def test_query_parser():
    meta = build_meta(FIXTURE, Path(tempfile.mkdtemp()) / "m.pkl")
    p = QueryParser(meta)
    assert p.parse("How is the camera on the Galaxy S10?").model == "Galaxy S10"
    assert p.parse("s10 gaming lag").model == "Galaxy S10"
    assert p.parse("iphone 11 pro battery").model == "iPhone 11 Pro"  # longest match wins
    assert p.parse("iPhone11 screen").model == "iPhone 11"
    assert p.parse("best phone overall").model is None
    q = p.parse("Moto G Power signal on verizon")
    assert q.model == "Moto G Power" and q.aspects == ["carrier"]  # 'power' in the name is not 'battery'


def test_query_parser_ignores_junk_model_names():
    """Names like these occur in the real corpus catalogue."""
    import pandas as pd

    rows = [
        ("Pre", "Palm", 470), ("6.1", "Nokia", 800), ("Rugged", "Kyocera", 300), ("3310", "Nokia", 630),
        ("G6", "LG", 527), ("Moto G6", "Motorola", 2983), ("Galaxy S10", "Samsung", 3383),
        ("Galaxy S10e", "Samsung", 3234), ("Galaxy S10 Plus", "Samsung", 3453), ("Ultra 5G", "Samsung", 100),
        ("7700mah Battery", "Generic", 50), ("G2", "LG", 2020),
    ]
    meta = pd.DataFrame([(m, b) for m, b, n in rows for _ in range(n)], columns=["model", "brand"])
    p = QueryParser(meta)
    assert p.detect_model("are the pre-installed apps annoying") is None
    assert p.detect_model("is the 6.1 inch screen too small") is None
    assert p.detect_model("rugged phone with good battery") is None
    assert p.detect_model("Palm Pre battery") == "Pre"
    assert p.detect_model("nokia 3310 battery") == "3310"
    assert p.detect_model("LG G6 camera") == "G6"
    assert p.detect_model("moto g6 camera") == "Moto G6"
    assert p.detect_model("Galaxy S10e screen") == "Galaxy S10e"
    assert p.detect_model("galaxy s10 plus battery") == "Galaxy S10 Plus"
    assert p.detect_model("galaxy s10 battery") == "Galaxy S10"
    assert p.detect_model("s10e screen") == "Galaxy S10e"
    assert p.detect_model("Is the 5G signal good?") is None  # spec, not the model "Ultra 5G"
    assert p.detect_model("samsung ultra 5g camera") == "Ultra 5G"
    assert p.detect_model("5000mah battery") is None  # not the accessory "7700mah Battery"
    assert p.detect_model("128gb storage") is None
    assert p.detect_model("g2 camera") == "G2"  # letter before number = series name
    # "+" means "Plus"
    assert p.detect_model("s10+ battery") == "Galaxy S10 Plus"
    assert p.parse("Galaxy S10+ camera").without_model == "camera"


def test_resolve_names_from_llm():
    """Product names returned by the query rewrite are mapped onto corpus names."""
    import pandas as pd

    rows = [("Galaxy S10", "Samsung", 30), ("Galaxy S10e", "Samsung", 20), ("Galaxy S10 Plus", "Samsung", 25),
            ("iPhone 11", "Apple", 30), ("3310", "Nokia", 5), ("Oneplus 7 Pro", "OnePlus", 5)]
    meta = pd.DataFrame([(m, b) for m, b, n in rows for _ in range(n)], columns=["model", "brand"])
    p = QueryParser(meta)
    assert p.resolve_model("Samsung", "Galaxy S10") == "Galaxy S10"
    assert p.resolve_model("Samsung", "galaxy s10e") == "Galaxy S10e"  # case/spacing
    assert p.resolve_model("Samsung", "Galaxy S10+") == "Galaxy S10 Plus"  # "+" -> Plus
    assert p.resolve_model("Samsung", "S10 Plus") == "Galaxy S10 Plus"  # family word missing
    assert p.resolve_model("OnePlus", "OnePlus 7 Pro") == "Oneplus 7 Pro"  # corpus spelling
    assert p.resolve_model("Nokia", "3310") == "3310"
    assert p.resolve_model("Apple", "iPhone 99") is None  # not in the corpus -> no filter
    assert p.resolve_model("Samsung", None) is None
    assert p.resolve_brand("samsung") == "Samsung" and p.resolve_brand("Nokiaa") is None
    q = "How does battery compare between Samsung Galaxy S10+ and Apple iPhone 11?"
    assert p.remove_model(q, "Galaxy S10 Plus") == "How does battery compare between and Apple iPhone 11?"


# ------------------------------------------------------------------ fusion
def test_rrf_and_aspect_boost():
    fused = rrf_fuse({"a": [1, 2, 3], "b": [3, 1]}, k=60)
    assert math.isclose(fused[1], 1 / 61 + 1 / 62)
    assert math.isclose(fused[3], 1 / 63 + 1 / 61)
    assert math.isclose(fused[2], 1 / 62)
    tags = {1: ["camera"], 2: [], 3: ["battery"]}
    boosted = add_aspect_boost(fused, tags, ["battery"], weight=0.5, k=60)
    assert boosted == {3} and math.isclose(fused[3], 1 / 63 + 1 / 61 + 0.5 / 61)


# -------------------------------------------------------------- retriever
def test_retriever_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        r = _build(tmp)
        res = r.retrieve("How is the camera on the Galaxy S10?", mode="hybrid", k=5)
        assert res.filters == {"model": "Galaxy S10"}
        assert res.n_candidates == 8  # 7 x "Galaxy S10" + 1 x "Galaxy S 10" (same phone)
        assert res.model_variants == ["Galaxy S10", "Galaxy S 10"]
        assert all(h.model in res.model_variants for h in res.hits)
        assert res.hits[0].doc_id in {"s10a", "s10e"}
        assert [h.rank for h in res.hits] == list(range(1, len(res.hits) + 1))
        assert {"parse_filter", "bm25", "dense_encode", "dense_search", "total"} <= set(res.timings_ms)

        for mode in ("bm25", "dense"):
            res = r.retrieve("battery charge", mode=mode, k=3, auto_filter=False)
            assert len(res.hits) == 3
            assert all(h.score >= res.hits[-1].score for h in res.hits)

        # explicit filter overrides auto-detected model
        res = r.retrieve("Galaxy S10 battery", k=10, filters={"model": "iPhone 11"})
        assert all(h.model == "iPhone 11" for h in res.hits)

        # rating filter
        res = r.retrieve("battery", k=10, auto_filter=False, filters={"max_rating": 2})
        assert res.hits and all(h.rating <= 2 for h in res.hits)


def test_model_mention_removed_from_search_text():
    meta = build_meta(FIXTURE, Path(tempfile.mkdtemp()) / "m.pkl")
    p = QueryParser(meta)
    assert p.parse("How is the camera on the Galaxy S10?").without_model == "How is the camera on the ?"
    assert p.parse("galaxy-s10 battery").without_model == "battery"
    assert p.parse("iPhone11 screen").without_model == "screen"
    assert p.parse("best phone").without_model is None
    with tempfile.TemporaryDirectory() as tmp:
        r = _build(tmp)
        res = r.retrieve("How is the camera on the Galaxy S10?", k=3)
        assert res.search_text == "How is the camera on the ?"
        # explicit filter for a different model: keep the query as typed
        res = r.retrieve("Galaxy S10 camera", k=3, filters={"model": "iPhone 11"})
        assert res.search_text == "Galaxy S10 camera"
        # nothing left after removal -> keep the query as typed
        assert r.retrieve("Galaxy S10", k=3).search_text == "Galaxy S10"
        assert r.retrieve("Galaxy S10 camera", k=3, strip_model=False).search_text == "Galaxy S10 camera"
        # explicit filter for the model in the query (LLM path): also removed
        res = r.retrieve("How is the camera on the Samsung Galaxy S10?", k=3,
                         filters={"model": "Galaxy S10"}, auto_filter=False)
        assert res.search_text == "How is the camera on the ?"
        assert all(h.model in ("Galaxy S10", "Galaxy S 10") for h in res.hits)


def test_spelling_variants_are_one_phone():
    """Spellings found in the real catalogue are grouped; different phones are not."""
    import pandas as pd

    rows = [("Motorola Razr", "Motorola", 35), ("Razr", "Motorola", 23), ("Galaxy S20 FE", "Samsung", 32),
            ("S20 FE", "Samsung", 1), ("Galaxy Note 9", "Samsung", 30), ("Galaxy Note9", "Samsung", 2),
            ("G8X Thinq™", "LG", 5), ("G8X Thinq", "LG", 4), ("Moto G Power", "Motorola", 3),
            ("Moto G Power (2020)", "Motorola", 2), ("Galaxy Note 10", "Samsung", 13), ("Note 10", "Xiaomi", 4)]
    meta = pd.DataFrame([(m, b) for m, b, n in rows for _ in range(n)], columns=["model", "brand"])
    p = QueryParser(meta)
    assert p.variants_of("Razr") == ["Motorola Razr", "Razr"]
    assert p.canonical_of("S20 FE") == "Galaxy S20 FE"
    assert p.variants_of("Galaxy Note 9") == ["Galaxy Note 9", "Galaxy Note9"]
    assert p.variants_of("G8X Thinq") == ["G8X Thinq™", "G8X Thinq"]
    assert p.variants_of("Moto G Power") == ["Moto G Power"]  # the 2020 model is a different phone
    assert p.variants_of("Note 10") == ["Note 10"]  # Xiaomi, not Samsung's Galaxy Note 10
    assert p.resolve_model("Motorola", "Razr") == "Motorola Razr"  # canonical spelling
    assert p.detect_model("galaxy note9 battery") == "Galaxy Note 9"
    assert p.remove_model("Is the Galaxy Note9 good?", "Galaxy Note 9") == "Is the good?"


def test_variant_filter_in_retrieve():
    with tempfile.TemporaryDirectory() as tmp:
        r = _build(tmp)
        # an explicit filter with the rarer spelling covers the whole phone too
        res = r.retrieve("battery", k=10, auto_filter=False, filters={"model": "Galaxy S 10", "min_rating": 1})
        assert res.filters["model"] == "Galaxy S10" and res.n_candidates == 8
        res = r.retrieve("Galaxy S10 battery lasts all day", k=10)
        assert "s10g_0" in {h.chunk_id for h in res.hits}  # the "Galaxy S 10" review is found


def test_aspects_override_from_llm():
    with tempfile.TemporaryDirectory() as tmp:
        r = _build(tmp)
        # "lasts all day" has no aspect keyword; the LLM says it is about the battery
        res = r.retrieve("does it last all day", k=10, auto_filter=False, aspects=["battery", "battery"])
        assert res.parsed.aspects == ["battery"]
        assert all(("battery" in h.aspect_tags) == h.aspect_boosted for h in res.hits)
        assert any(h.aspect_boosted for h in res.hits)
        assert r.retrieve("does it last all day", k=10, auto_filter=False).parsed.aspects == []


def test_dedupe_by_review():
    with tempfile.TemporaryDirectory() as tmp:
        r = _build(tmp)
        res = r.retrieve("screen gorgeous", mode="bm25", k=10, auto_filter=False)
        assert len({h.doc_id for h in res.hits}) == len(res.hits)
        res = r.retrieve("screen gorgeous", mode="bm25", k=10, auto_filter=False, max_per_doc=None)
        assert {"s10c_0", "s10c_1"} <= {h.chunk_id for h in res.hits}  # both overlapping chunks kept


def test_dense_shards_resume_and_alignment():
    with tempfile.TemporaryDirectory() as tmp:
        paths = IndexPaths(tmp)
        meta = build_meta(FIXTURE, paths.meta)
        enc = HashEncoder()
        texts = meta["text"].tolist()
        encode_corpus(texts, enc, paths.dense_dir, shard_size=5)
        shards = sorted((paths.dense_dir / "shards").glob("part_*.npy"))
        assert len(shards) == 4
        shards[-1].unlink()  # simulate an interrupted run
        encode_corpus(texts, enc, paths.dense_dir, shard_size=5)  # resumes
        DenseIndex.build(paths.dense_dir, use_faiss=False)
        d = DenseIndex.load(paths.dense_dir, enc)
        # every chunk's own text must retrieve itself first (row ids aligned)
        for i, t in enumerate(texts):
            rows, _ = d.search(t, k=1)
            assert np.isclose(enc.encode_queries([t])[0] @ enc.encode_documents([texts[rows[0]]])[0], 1.0), i


def test_mismatched_index_is_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        r = _build(tmp)
        try:
            Retriever(r.meta.iloc[:5], r.bm25, None)
        except RuntimeError:
            return
        raise AssertionError("expected a size-mismatch error")


class RetrievalCoreTests(unittest.TestCase):
    """Lets `python -m unittest discover -s tests` run the function-style tests above."""


for _name, _fn in sorted(dict(globals()).items()):
    if _name.startswith("test_") and callable(_fn):
        setattr(RetrievalCoreTests, _name, staticmethod(_fn))


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(tests)} tests passed")

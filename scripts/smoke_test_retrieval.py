"""End-to-end smoke test with the REAL libraries (FAISS + BGE + stemmer).

Builds every index on the small synthetic fixture in a temporary folder and
checks that each mode returns sensible results. Run it once after `uv sync`:

    uv run python -m scripts.smoke_test_retrieval

It downloads BAAI/bge-small-en-v1.5 (~130 MB) on first use.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "mini_corpus.jsonl"


def main() -> int:
    from retrieval import Retriever
    from retrieval.bm25 import BM25Index
    from retrieval.config import IndexPaths
    from retrieval.corpus import build_meta
    from retrieval.dense import HAVE_FAISS, DenseIndex, SentenceTransformerEncoder, encode_corpus

    print(f"faiss available: {HAVE_FAISS}")
    if not HAVE_FAISS:
        print("FAIL: faiss is not installed (uv add faiss-cpu)")
        return 1

    with tempfile.TemporaryDirectory() as tmp:
        paths = IndexPaths(tmp)
        meta = build_meta(FIXTURE, paths.meta)
        bm25 = BM25Index.build(meta["text"].tolist(), log_every=0)
        bm25.save(paths.bm25_dir)
        print(f"stemmer: {bm25.params['stemmer']}")
        enc = SentenceTransformerEncoder()
        print(f"encoder device: {enc.device}")
        encode_corpus(meta["text"].tolist(), enc, paths.dense_dir, shard_size=8)
        DenseIndex.build(paths.dense_dir)

        r = Retriever.load(tmp, encoder=enc)
        assert r.dense.backend_name == "faiss.IndexFlatIP", r.dense.backend_name
        failures = 0

        def check(name, cond):
            nonlocal failures
            print(("  ok   " if cond else "  FAIL ") + name)
            failures += 0 if cond else 1

        # Vocabulary mismatch: no query word appears in the relevant reviews.
        res = r.retrieve("phone runs out of juice before the day ends", mode="dense", k=3, auto_filter=False)
        top = [h.doc_id for h in res.hits]
        print("  dense top3:", top)
        check("dense finds battery complaints without shared words", bool({"s10b", "s10d"} & set(top)))

        res = r.retrieve("How is the camera on the Galaxy S10?", mode="hybrid", k=5)
        check("model auto-filter -> only Galaxy S10 hits", all(h.model == "Galaxy S10" for h in res.hits))
        check("camera chunks ranked first", res.hits[0].doc_id in {"s10a", "s10e"})
        check("aspect boost applied", any(h.aspect_boosted for h in res.hits))

        res = r.retrieve("iPhone 11 Pro camera lens", mode="dense", k=5)
        check("FAISS IDSelector subset search", all(h.model == "iPhone 11 Pro" for h in res.hits) and len(res.hits) == 2)

        res = r.retrieve("screen gorgeous", mode="bm25", k=5, auto_filter=False)
        check("max_per_doc=1 de-duplicates overlapping chunks", len({h.doc_id for h in res.hits}) == len(res.hits))
        print(f"  timings (ms): {res.timings_ms}")

        print("\nALL OK" if failures == 0 else f"\n{failures} check(s) failed")
        return 1 if failures else 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    sys.exit(main())

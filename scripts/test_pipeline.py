"""End-to-end check of the integrated pipeline on the REAL indexes.

    question -> query rewrite (DeepSeek) -> resolve products to corpus models
             -> hybrid retrieval (k=50 per product) -> MiniLM rerank (top 5)

This mirrors the reference ``search()`` in the integration plan. It checks
that every stage runs and that the filters are right, and prints per-stage
latency for the report.

Rewrite modes (``--rewrite``):
    live    call DeepSeek (needs DEEPSEEK_API_KEY in .env); on failure the
            pipeline falls back to the original question, as the UI will
    canned  hand-written rewrite outputs in DeepSeek's format, so the
            resolve -> retrieve -> rerank path can be tested without an API key
    off     no rewrite: the retriever's own parser handles the question
    auto    live if DEEPSEEK_API_KEY is set, else canned (default)

Usage (repository root, after the indexes are built):

    uv run python -m scripts.test_pipeline
    uv run python -m scripts.test_pipeline --rewrite live
    uv run python -m scripts.test_pipeline --rewrite off --no-rerank
    uv run python -m scripts.test_pipeline --json results/pipeline_check.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# --------------------------------------------------------------------------- cases
@dataclass
class Case:
    name: str
    question: str
    # Hand-written stand-in for DeepSeek's output (same JSON shape). NOT a real
    # model response; used only in --rewrite canned mode.
    canned: dict
    # Expected filters per retrieval group when the rewrite gives these products.
    expect_filters: list[dict]
    # Expected filters when there is no rewrite (parser only).
    expect_filters_off: list[dict]


CASES = [
    Case(
        "plain question",
        "How is the camera on the Galaxy S10?",
        {"query": "How is the camera on the Samsung Galaxy S10?",
         "product": [{"brand": "Samsung", "model": "Galaxy S10"}], "aspects": ["camera"]},
        [{"model": "Galaxy S10"}],
        [{"model": "Galaxy S10"}],
    ),
    Case(
        "Chinese question",
        "Galaxy S10 的电池续航怎么样？",
        {"query": "How is the battery life of the Samsung Galaxy S10?",
         "product": [{"brand": "Samsung", "model": "Galaxy S10"}], "aspects": ["battery"]},
        [{"model": "Galaxy S10"}],
        [{"model": "Galaxy S10"}],
    ),
    Case(
        "S10+ spelling",
        "Is the S10+ battery any good?",
        {"query": "Is the battery of the Samsung Galaxy S10+ good?",
         "product": [{"brand": "Samsung", "model": "Galaxy S10+"}], "aspects": ["battery"]},
        [{"model": "Galaxy S10 Plus"}],
        [{"model": "Galaxy S10 Plus"}],
    ),
    Case(
        "comparison",
        "Which has better battery life, Galaxy S10 or iPhone 11?",
        {"query": "How does battery life compare between Samsung Galaxy S10 and Apple iPhone 11?",
         "product": [{"brand": "Samsung", "model": "Galaxy S10"}, {"brand": "Apple", "model": "iPhone 11"}],
         "aspects": ["battery"]},
        [{"model": "Galaxy S10"}, {"model": "iPhone 11"}],
        [{"model": "Galaxy S10"}],  # the parser alone keeps one model
    ),
    Case(
        "brand only",
        "Are Motorola phones good for battery life?",
        {"query": "How is the battery life of Motorola phones?",
         "product": [{"brand": "Motorola", "model": None}], "aspects": ["battery"]},
        [{"brand": "Motorola"}],
        [{}],  # the parser does not filter on brand words alone
    ),
    Case(
        "no product",
        "My phone dies before lunch, which one lasts longer?",
        {"query": "Which smartphone has long battery life that lasts a full day?",
         "product": [], "aspects": ["battery"]},
        [{}],
        [{}],
    ),
    Case(
        "model not in corpus",
        "Is the iPhone 99 camera good?",
        {"query": "How is the camera on the Apple iPhone 99?",
         "product": [{"brand": "Apple", "model": "iPhone 99"}], "aspects": ["camera"]},
        [{"brand": "Apple"}],  # unknown model -> fall back to the brand
        [{}],
    ),
]


# ------------------------------------------------------------------------ pipeline
def product_filter(r, product: dict) -> dict | None:
    """Same as the integration plan: model if it maps to the corpus, else brand, else none."""
    model = r.parser.resolve_model(product.get("brand"), product.get("model"))
    if model:
        return {"model": model}
    brand = r.parser.resolve_brand(product.get("brand"))
    return {"brand": brand} if brand else None


@dataclass
class Outcome:
    case: str
    question: str
    rewrite_source: str  # live / canned / off / fallback (live call failed)
    rewrite: dict | None
    groups: list[dict] = field(default_factory=list)
    timing_ms: dict = field(default_factory=dict)
    checks: list[tuple[str, bool, str]] = field(default_factory=list)  # (name, ok, detail)
    error: str | None = None

    @property
    def passed(self) -> bool:
        return self.error is None and all(ok for _, ok, _ in self.checks)


def run_case(r, reranker, case: Case, mode: str, pool_k: int, top_k: int) -> Outcome:
    out = Outcome(case.name, case.question, mode, None)
    started = perf_counter()

    # 1. rewrite
    rewritten = None
    if mode == "canned":
        rewritten = case.canned
    elif mode == "live":
        from retrieval.query_rewrite import rewrite_query

        t = perf_counter()
        try:
            rewritten = rewrite_query(case.question)
        except (RuntimeError, ValueError) as exc:
            out.rewrite_source = f"fallback ({exc})"
        out.timing_ms["rewrite"] = (perf_counter() - t) * 1000
    out.rewrite = rewritten

    # 2. resolve + retrieve
    t = perf_counter()
    if rewritten:
        query = rewritten["query"]
        filters = [product_filter(r, p) for p in rewritten["product"]] or [None]
        results = [r.retrieve(query, k=pool_k, auto_filter=False, aspects=rewritten["aspects"], filters=f)
                   for f in filters]
    else:
        query = case.question
        results = [r.retrieve(query, k=pool_k)]
    out.timing_ms["retrieve"] = (perf_counter() - t) * 1000

    # 3. rerank
    t = perf_counter()
    for res in results:
        candidates = [h.to_dict() for h in res.hits]
        if reranker is not None and candidates:
            top = reranker.rerank(query, candidates, query_aspects=res.parsed.aspects, top_k=top_k)
        else:
            top = candidates[:top_k]
        out.groups.append({
            "filters": res.filters,
            "search_text": res.search_text,
            "aspects": res.parsed.aspects,
            "n_candidates": res.n_candidates,
            "n_hits": len(res.hits),
            "retrieval_ms": res.timings_ms,
            "top": [{k: c.get(k) for k in ("chunk_id", "model", "brand", "rating", "aspect_tags",
                                           "rerank_score", "aspect_bonus", "text")} for c in top],
        })
    if reranker is not None:
        out.timing_ms["rerank"] = (perf_counter() - t) * 1000
    out.timing_ms["total"] = (perf_counter() - started) * 1000

    # 4. checks
    expected = case.expect_filters if rewritten else case.expect_filters_off
    got = [g["filters"] for g in out.groups]
    strict = out.rewrite_source in ("canned", "off")  # a live LLM may name products differently
    out.checks.append(("filters", got == expected or not strict,
                       f"got {got}, expected {expected}" + ("" if strict else " (live: not enforced)")))
    for i, g in enumerate(out.groups):
        out.checks.append((f"group {i + 1} has hits", g["n_hits"] > 0, f"{g['n_hits']} hits"))
        f = g["filters"]
        ok = all((not f.get("model") or c["model"] == f["model"]) and
                 (not f.get("brand") or c["brand"] == f["brand"]) for c in g["top"])
        out.checks.append((f"group {i + 1} respects filter", ok, str(f) if f else "no filter"))
        if reranker is not None:
            scores = [c["rerank_score"] for c in g["top"]]
            out.checks.append((f"group {i + 1} reranked", all(s is not None for s in scores)
                               and scores == sorted(scores, reverse=True), f"{len(scores)} kept"))
    return out


# -------------------------------------------------------------------------- report
def show(o: Outcome, width: int) -> None:
    mark = "PASS" if o.passed else "FAIL"
    print(f"\n[{mark}] {o.case}: {o.question}")
    print(f"   rewrite: {o.rewrite_source}" + (f" -> {o.rewrite['query']!r}" if o.rewrite else ""))
    if o.rewrite:
        print(f"            products={o.rewrite['product']} aspects={o.rewrite['aspects']}")
    for i, g in enumerate(o.groups, 1):
        print(f"   group {i}: filters={g['filters'] or 'none'}  candidates={g['n_candidates'] or 'all'}"
              f"  hits={g['n_hits']}  aspects={g['aspects']}")
        print(f"            searched: {g['search_text']!r}")
        for j, c in enumerate(g["top"][:3], 1):
            score = f"{c['rerank_score']:.2f}" if c.get("rerank_score") is not None else "-"
            print(f"      {j}. [{score}] {c['model']} | {c['rating']}* | "
                  + textwrap.shorten(c["text"], width))
    for name, ok, detail in o.checks:
        if not ok:
            print(f"   ! {name}: {detail}")
    print("   timing (ms): " + ", ".join(f"{k}={v:.0f}" for k, v in o.timing_ms.items()))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index-dir", default="indexes")
    ap.add_argument("--rewrite", choices=["auto", "live", "canned", "off"], default="auto")
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--reranker-device", default="auto",
                    help="auto (cuda > mps > cpu) / cpu / mps / cuda. On macOS, CPU torch in the same "
                         "process as FAISS crashed with a segfault in testing, so auto prefers MPS.")
    ap.add_argument("--pool-k", type=int, default=50)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--only", nargs="*", help="run only cases whose name contains one of these words")
    ap.add_argument("--json", type=Path, help="also save all outcomes as JSON")
    ap.add_argument("--width", type=int, default=110)
    args = ap.parse_args()

    mode = args.rewrite
    if mode == "auto":
        try:
            from dotenv import load_dotenv

            load_dotenv(ROOT / ".env", override=False)
        except ImportError:
            pass
        mode = "live" if os.getenv("DEEPSEEK_API_KEY", "").strip() else "canned"

    from retrieval import Retriever

    print(f"Loading indexes from {args.index_dir} ...")
    t = perf_counter()
    r = Retriever.load(args.index_dir)
    print(f"  loaded {len(r.meta):,} chunks in {perf_counter() - t:.1f}s; rewrite mode: {mode}")

    reranker = None
    if not args.no_rerank:
        from retrieval.dense import pick_device
        from retrieval.rerank import CrossEncoderReranker, RerankerConfig

        if args.reranker_device == "auto":
            args.reranker_device = pick_device()
        reranker = CrossEncoderReranker(RerankerConfig(device=args.reranker_device))
        t = perf_counter()
        reranker.rerank("warm up", [{"chunk_id": "w", "doc_id": "w", "text": "warm up"}], top_k=1)
        print(f"  reranker ready on {args.reranker_device} in {perf_counter() - t:.1f}s")

    cases = [c for c in CASES if not args.only or any(w.lower() in c.name.lower() for w in args.only)]
    outcomes = []
    for case in cases:
        try:
            o = run_case(r, reranker, case, mode, args.pool_k, args.top_k)
        except Exception as exc:  # report and keep going
            o = Outcome(case.name, case.question, mode, None, error=f"{type(exc).__name__}: {exc}")
        outcomes.append(o)
        if o.error:
            print(f"\n[FAIL] {o.case}: {o.question}\n   ! error: {o.error}")
        else:
            show(o, args.width)

    print("\n" + "=" * 70)
    passed = sum(o.passed for o in outcomes)
    print(f"{passed}/{len(outcomes)} cases passed  (rewrite={mode}, rerank={'off' if reranker is None else args.reranker_device})")
    stages = ["rewrite", "retrieve", "rerank", "total"]
    for s in stages:
        vals = [o.timing_ms[s] for o in outcomes if s in o.timing_ms]
        if vals:
            print(f"  {s:<9} median {statistics.median(vals):7.0f} ms   max {max(vals):7.0f} ms   (n={len(vals)})")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps([o.__dict__ | {"passed": o.passed} for o in outcomes],
                                        ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(f"  saved {args.json}")
    return 0 if passed == len(outcomes) else 1


if __name__ == "__main__":
    sys.exit(main())

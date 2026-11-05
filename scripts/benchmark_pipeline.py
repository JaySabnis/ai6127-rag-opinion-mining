"""Measure the real UI search + answer pipeline on at least five distinct queries.

This command makes live DeepSeek calls unless --check-only is set. It never
substitutes sample candidates or canned API outputs. Run from the repository:

    python -m scripts.benchmark_pipeline --queries questions.jsonl \
        --env-file .env --output-dir results/end_to_end --expected-chunks 459418

Input is a JSON array or JSONL of strings or {"query_id": "q1", "query": "..."}.
Use a NEW output directory. Index/model loading and one local search warmup are
separate from query totals. All calls are serial, with no application response
cache by default. Provider-side and OS caches are not controlled.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.summarize_latency import summarize
from generation.client import DeepSeekClient, load_env, write_json


def read_queries(path):
    path = Path(path)
    raw = path.read_text(encoding="utf-8-sig")
    try:
        rows = json.loads(raw) if path.suffix.lower() == ".json" else [json.loads(line) for line in raw.splitlines() if line.strip()]
    except ValueError as exc:
        raise ValueError(f"Invalid query file JSON: {exc}") from None
    if not isinstance(rows, list):
        raise ValueError("Query JSON must contain an array")
    result, ids, questions = [], set(), set()
    for i, row in enumerate(rows, 1):
        if isinstance(row, str):
            row = {"query": row}
        if not isinstance(row, dict):
            raise ValueError(f"Query {i} must be a string or object")
        query = row.get("query", row.get("question"))
        query_id = row.get("query_id", f"q{i:03}")
        if not isinstance(query, str) or not query.strip() or len(query.strip()) > 2000:
            raise ValueError(f"Query {i} must contain 1–2000 characters")
        if not isinstance(query_id, str) or not query_id.strip() or query_id in ids:
            raise ValueError(f"Query {i} must have a unique, nonempty string query_id")
        normalized = " ".join(query.casefold().split())
        if normalized in questions:
            raise ValueError(f"Query {i} duplicates an earlier question")
        ids.add(query_id)
        questions.add(normalized)
        result.append({"query_id": query_id, "query": query.strip()})
    if len(result) < 5:
        raise ValueError("A benchmark needs at least five distinct queries")
    return result


def prerequisites(index_dir, *, rewrite=True, rerank=True):
    """Check files/import availability without loading models or making calls."""
    index_dir = Path(index_dir)
    problems = []
    needed = ["numpy", "scipy", "pandas", "torch", "sentence_transformers"]
    if rewrite:
        needed += ["httpx", "dotenv"]
    if (index_dir / "dense/faiss.index").is_file() and not (index_dir / "dense/vectors.npy").is_file():
        needed.append("faiss")
    params = index_dir / "bm25/params.json"
    if params.is_file():
        config = json.loads(params.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("Invalid index bm25/params.json: expected an object")
        stemmer = config.get("stemmer")
        needed += {"nltk-snowball-english": ["nltk"], "pystemmer-english": ["Stemmer"]}.get(stemmer, [])
    missing = [name for name in needed if importlib.util.find_spec(name) is None]
    if missing:
        problems.append("Missing Python packages: " + ", ".join(missing) + ". Install this repository's dependencies in the selected Python environment.")
    required = ["meta.pkl", "bm25/weights.npz", "bm25/vocab.json", "bm25/params.json", "dense/info.json"]
    absent = [str(index_dir / name) for name in required if not (index_dir / name).is_file()]
    if not any((index_dir / "dense" / name).is_file() for name in ("faiss.index", "vectors.npy")):
        absent.append(str(index_dir / "dense") + "/{faiss.index or vectors.npy}")
    if absent:
        problems.append("Missing retrieval index files: " + ", ".join(absent) + ". Build or restore the real indexes; no fixture fallback is available.")
    if rerank and not (ROOT / "retrieval/models/review_ranker_v2.json").is_file():
        problems.append("Missing trained reranker artifact: retrieval/models/review_ranker_v2.json")
    return problems


def timing_row(query, searched, reply, elapsed, *, position):
    """Keep the outer total authoritative; expose only disjoint stages."""
    row = {**query, "timing_scope": "end_to_end", "query_state": "after_local_search_warmup",
           "query_position": position, "n_chunks": len(reply.get("chunks", [])), "total_s": elapsed}
    for key in ("rewrite", "retrieve", "rerank"):
        if key in searched.get("timing_ms", {}):
            row[key + "_s"] = searched["timing_ms"][key] / 1000
    for key in ("classification_s", "generation_s", "verification_s"):
        if key in reply.get("timing", {}):
            row[key] = reply["timing"][key]
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, default=ROOT / "indexes")
    parser.add_argument("--expected-chunks", type=int, help="Fail if loaded index count differs; use 459418 for the audited corpus")
    parser.add_argument("--model", help="Use this model for both rewrite and downstream requests")
    parser.add_argument("--cache-dir", type=Path, help="Opt in to downstream response cache; rewrite is always uncached")
    parser.add_argument("--max-api-calls", type=int, default=32, help="Downstream API call budget PER query, excluding one rewrite call")
    parser.add_argument("--no-rewrite", action="store_true")
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--pool-k", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--reranker-device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--check-only", action="store_true", help="Check input/files/packages only: no models, API calls or output files")
    args = parser.parse_args(argv)
    manifest = None
    try:
        queries = read_queries(args.queries)
        if args.env_file and not args.env_file.is_file():
            raise ValueError("The explicit --env-file does not exist")
        if min(args.pool_k, args.top_k, args.max_api_calls) < 1 or args.top_k > args.pool_k:
            raise ValueError("pool-k, top-k and max-api-calls must be positive; top-k must not exceed pool-k")
        if args.expected_chunks is not None and args.expected_chunks < 1:
            raise ValueError("expected-chunks must be positive")
        if args.output_dir.exists():
            raise ValueError("Use a new --output-dir; existing benchmark results will not be overwritten")
        load_env(args.env_file)
        if args.model:
            os.environ["DEEPSEEK_MODEL"] = args.model
        problems = prerequisites(args.index_dir, rewrite=not args.no_rewrite, rerank=not args.no_rerank)
        if not os.environ.get("DEEPSEEK_API_KEY", "").strip():
            problems.append("Set DEEPSEEK_API_KEY in --env-file or the process environment.")
        if problems:
            raise ValueError("\n".join(problems))
        if args.check_only:
            print(f"Prerequisite check passed for {len(queries)} queries. Model/index loading, API access and index completeness are not yet verified.")
            return 0
        from app.pipeline import search_full_corpus
        from generation.pipeline import enrich_reply
        from retrieval import Retriever
        from retrieval.dense import pick_device
        from retrieval.rerank import CrossEncoderReranker, RerankerConfig
        rewrite_fn = None
        if not args.no_rewrite:
            from retrieval.query_rewrite import rewrite_query
            rewrite_fn = rewrite_query
        # Constructing a client validates settings and sends no requests.
        DeepSeekClient(env_file=args.env_file, max_calls=args.max_api_calls)
        args.output_dir.mkdir(parents=True)
        manifest = {"status": "loading", "started_at": datetime.now(timezone.utc).isoformat(),
                    "n_queries_requested": len(queries), "index_dir": str(args.index_dir.resolve()),
                    "query_file": str(args.queries.resolve()), "expected_index_chunks": args.expected_chunks,
                    "corpus_scope": "loaded_index; completeness requires an externally verified expected count",
                    "rewrite": not args.no_rewrite, "rerank": not args.no_rerank, "verify": not args.no_verify,
                    "top_k_per_product": args.top_k, "pool_k_per_product": args.pool_k,
                    "downstream_model": os.getenv("DEEPSEEK_MODEL", "deepseek-flash"),
                    "rewrite_model": os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro") if rewrite_fn else None,
                    "downstream_cache": str(args.cache_dir.resolve()) if args.cache_dir else "disabled",
                    "rewrite_cache": "disabled", "provider_and_os_cache_state": "uncontrolled",
                    "api_warmup": "none; API warm/cold state is unknown", "concurrency": 1,
                    "max_downstream_api_calls_per_query": args.max_api_calls,
                    "query_total_scope": "Outer wall time around search_full_corpus + enrich_reply; excludes imports, client construction, index/model loading, local warmup and file writes",
                    "startup_in_query_total": False, "output_serialization_in_query_total": False}
        write_json(args.output_dir / "manifest.json", manifest)
        started = perf_counter()
        retriever = Retriever.load(args.index_dir)
        manifest["index_and_encoder_load_s"] = perf_counter() - started
        manifest["retriever_load_note"] = "Retriever.load includes one local dense encoder warmup; downloads may occur for absent weights. OS caches are not cleared."
        manifest["loaded_index_chunks"] = len(retriever.meta)
        if args.expected_chunks is not None and len(retriever.meta) != args.expected_chunks:
            raise ValueError(f"Loaded index has {len(retriever.meta)} chunks, expected {args.expected_chunks}; no queries were sent to the API")
        reranker = None
        if not args.no_rerank:
            device = pick_device() if args.reranker_device == "auto" else args.reranker_device
            reranker = CrossEncoderReranker(RerankerConfig(device=device))
            manifest["reranker_device"] = device
        search_options = {"top_k": args.top_k, "pool_k": args.pool_k, "rerank": not args.no_rerank, "reranker": reranker}
        started = perf_counter()
        # One real-index local search: no generated answer or paid warmup request.
        warmup = search_full_corpus(queries[0]["query"], retriever, **search_options, use_rewrite=False)
        manifest["local_search_and_reranker_warmup_s"] = perf_counter() - started
        manifest["local_warmup_retrieved_chunks"] = len(warmup.get("chunks", []))
        manifest["reranker_warmup_completed"] = bool(reranker and warmup.get("chunks"))
        manifest["local_warmup_note"] = "First input question searched once without rewrite/generation, excluded from timings. Uses the same real index and reranker."
        manifest["status"] = "running"
        write_json(args.output_dir / "manifest.json", manifest)
        timings, degraded = [], 0
        batch_started = perf_counter()
        for i, query in enumerate(queries, 1):
            client = DeepSeekClient(env_file=args.env_file, cache_dir=args.cache_dir, max_calls=args.max_api_calls)
            started = perf_counter()
            searched = search_full_corpus(query["query"], retriever, **search_options,
                                         use_rewrite=not args.no_rewrite, rewrite_fn=rewrite_fn)
            reply = enrich_reply(searched, query["query"], client=client,
                                 language="zh" if re.search(r"[\u4e00-\u9fff]", query["query"]) else "en",
                                 verify=not args.no_verify)
            elapsed = perf_counter() - started
            degraded_query = bool(searched.get("rewrite_error") or reply.get("checker_error"))
            degraded += degraded_query
            row = timing_row(query, searched, reply, elapsed, position=i)
            row["degraded"] = degraded_query
            timings.append(row)
            write_json(args.output_dir / f"query_{i:03}.json", {**query, "result": reply, "benchmark_timing": row})
            with (args.output_dir / "timings.jsonl").open("a", encoding="utf-8") as file:
                file.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            print(f"Completed query {i}/{len(queries)}: {elapsed:.3f} s" + (" (degraded)" if degraded_query else ""), flush=True)
        batch_wall = perf_counter() - batch_started
        summary = summarize(timings, run=args.output_dir.name)
        summary.update({"n_degraded_queries": degraded, "sequential_batch_wall_s": round(batch_wall, 3),
                        "sequential_batch_wall_includes_output_writes": True,
                        "observed_serial_batch_queries_per_minute": round(60 * len(timings) / batch_wall, 2),
                        "api_usage_note": "Per-query result.api covers downstream calls only; rewrite request usage/cost is not captured."})
        write_json(args.output_dir / "summary.json", summary)
        manifest.update(status="completed_with_degraded_queries" if degraded else "completed",
                        n_queries_completed=len(timings), finished_at=datetime.now(timezone.utc).isoformat())
        write_json(args.output_dir / "manifest.json", manifest)
        print(f"Saved measured results to {args.output_dir}")
        return 1 if degraded else 0
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        if manifest is not None:
            manifest.update(status="failed", failure_type=type(exc).__name__, failure_message=str(exc))
            write_json(args.output_dir / "manifest.json", manifest)
        parser.exit(1, f"Benchmark not completed: {exc}\n")


if __name__ == "__main__":
    sys.exit(main())

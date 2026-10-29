from __future__ import annotations

import json
import os
from pathlib import Path
from time import perf_counter

import torch

from app.pipeline import product_filter
from retrieval import Retriever
from retrieval.query_rewrite import validate_result
from retrieval.rerank import CrossEncoderReranker, RerankerConfig

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "evaluation" / "retrieval_ablation_40"
OUT = ROOT / "results" / "ablation_8"


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def main():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    OUT.mkdir(parents=True, exist_ok=True)
    checkpoints = OUT / "collection"
    checkpoints.mkdir(exist_ok=True)
    source = DATA / "queries.jsonl"
    queries = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    rewrites = {r["query_id"]: validate_result({k: r[k] for k in ("query", "product", "aspects")})
                for r in map(json.loads, (DATA / "rewrites.jsonl").read_text(encoding="utf-8").splitlines())}
    retriever = Retriever.load(ROOT / "indexes")
    # Current default reranker; original seven-system results remain in ablation_40.
    reranker = CrossEncoderReranker(RerankerConfig(local_files_only=True, strategy="legacy"))
    learned = CrossEncoderReranker(RerankerConfig(local_files_only=True))
    for q in queries:
        qid, text = q["query_id"], q["query"]
        path = checkpoints / f"{qid}.json"
        if path.exists() and "08_learned" in json.loads(path.read_text(encoding="utf8"))["runs"]:
            print(f"{qid} cached", flush=True)
            continue
        started = perf_counter()
        runs, details = {}, {}
        for name, mode, strip, boost in [
            ("01_bm25", "bm25", False, 0),
            ("02_dense", "dense", False, 0),
            ("03_rrf", "hybrid", False, 0),
            ("04_strip_model", "hybrid", True, 0),
            ("05_aspect", "hybrid", True, 0.5),
        ]:
            res = retriever.retrieve(text, mode=mode, k=100, candidate_k=150,
                                     strip_model=strip, aspect_weight=boost,
                                     max_per_doc=1)
            hits = [h.to_dict() for h in res.hits]
            runs[name] = hits[:100]
            details[name] = {"filters": res.filters, "search_text": res.search_text,
                             "aspects": res.parsed.aspects,
                             "candidate_count": len(hits)}
            if name == "05_aspect":
                runs["06_rerank"] = reranker.rerank(
                    text, hits, query_aspects=res.parsed.aspects, top_k=100)
                details["06_rerank"] = {**details[name], "config": vars(reranker.config)}
        rewritten = rewrites[qid]
        # Same parser-independent entity resolution as the application. A
        # comparison has up to 100 candidates per product, then one global Top-100.
        filters = [product_filter(retriever, p) for p in rewritten["product"]] or [None]
        merged, groups = {}, []
        for f in filters:
            res = retriever.retrieve(rewritten["query"], k=100, candidate_k=150,
                                     filters=f, auto_filter=False, strip_model=True,
                                     aspects=rewritten["aspects"], aspect_weight=0.5,
                                     max_per_doc=1)
            groups.append({"filters": res.filters, "search_text": res.search_text,
                           "candidate_count": len(res.hits)})
            for h in res.hits:
                hit = h.to_dict()
                if hit["chunk_id"] not in merged or hit["score"] > merged[hit["chunk_id"]]["score"]:
                    merged[hit["chunk_id"]] = hit
        merged_hits = sorted(merged.values(), key=lambda h: h["score"], reverse=True)
        runs["07_rewrite"] = reranker.rerank(
            rewritten["query"], merged_hits[:100],
            query_aspects=rewritten["aspects"], top_k=100)
        details["07_rewrite"] = {"rewrite": rewritten, "groups": groups,
                                 "config": vars(reranker.config)}
        # Same rewritten retrieval pool, latest learned ranking instead of legacy.
        # Strip every resolved product from the semantic query for comparisons.
        scoring_query = rewritten["query"]
        for f in filters:
            if f and f.get("model"):
                scoring_query = retriever.parser.remove_model(scoring_query, f["model"])
        runs["08_learned"] = learned.rerank(
            rewritten["query"], merged_hits[:100],
            query_aspects=rewritten["aspects"], scoring_query=scoring_query, top_k=100)
        details["08_learned"] = {"rewrite": rewritten, "groups": groups,
                                  "scoring_query": scoring_query, "config": vars(learned.config)}
        # All variants keep at most one chunk per source review in final output.
        for name, hits in runs.items():
            assert len({h["doc_id"] for h in hits}) == len(hits), name
        write_json(path, {"query_id": qid, "runs": runs, "details": details,
                          "elapsed_seconds": perf_counter() - started})
        count = len({h["chunk_id"] for hits in runs.values() for h in hits})
        print(f"{qid} pool={count} seconds={perf_counter()-started:.1f}", flush=True)


if __name__ == "__main__":
    main()

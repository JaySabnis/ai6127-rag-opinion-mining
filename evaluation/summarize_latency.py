"""
Summarize end-to-end latency and throughput of the pipeline (Q4
performance metrics).

Input: evaluation/timings/<run_name>.jsonl, one line per query, in the
order the queries were run. Every field ending in "_s" is a stage time in
seconds; n_chunks is the number of chunks classified for that query:

    {"query": "...", "retrieval_s": 0.21, "classification_s": 0.35, "generation_s": 2.8, "n_chunks": 20}

The pipeline can record these with time.perf_counter() around each stage.

Run (from the repository root):

    uv run python evaluation/summarize_latency.py evaluation/timings/<run_name>.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median, quantiles

HERE = Path(__file__).parent
RESULTS = HERE / "results"


def stats(values: list[float]) -> dict:
    # "inclusive" keeps p95 within the observed range when there are few queries.
    p95 = quantiles(values, n=20, method="inclusive")[-1] if len(values) > 1 else values[0]
    return {"mean": round(mean(values), 3), "median": round(median(values), 3),
            "p95": round(p95, 3), "max": round(max(values), 3)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("timings", help="JSONL with per-query stage times")
    args = parser.parse_args()

    with open(args.timings, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if not rows:
        raise SystemExit("No timing rows")
    stages = sorted({k for r in rows for k in r if k.endswith("_s")})
    for r in rows:
        r["total_s"] = sum(r.get(s, 0.0) for s in stages)

    summary = {"run": Path(args.timings).stem, "n_queries": len(rows)}
    for stage in stages + ["total_s"]:
        summary[stage] = stats([r[stage] for r in rows if stage in r])
    # The first query pays for loading models and indexes.
    summary["cold_total_s"] = round(rows[0]["total_s"], 3)
    if len(rows) > 1:
        summary["warm_total_mean_s"] = round(mean(r["total_s"] for r in rows[1:]), 3)
    summary["queries_per_minute"] = round(60 * len(rows) / sum(r["total_s"] for r in rows), 2)

    chunks = sum(r.get("n_chunks", 0) for r in rows)
    clf_time = sum(r.get("classification_s", 0.0) for r in rows)
    if chunks and clf_time:
        summary["chunks_classified_per_second"] = round(chunks / clf_time, 1)

    print(json.dumps(summary, indent=2))
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"latency_{summary['run']}.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

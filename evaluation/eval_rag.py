"""
Answer-level RAG metrics with RAGAS 0.4 (Q4): faithfulness, answer
relevancy, context precision and context relevance.

Input: evaluation/rag_outputs/<run_name>.jsonl, one line per query:

    {"question": "...", "contexts": ["retrieved chunk text", ...], "answer": "..."}

LLM judge: copy evaluation/llm_config.env.example to
evaluation/llm_config.env and fill it in (that file is git-ignored).

Run (from the repository root):

    uv run python evaluation/eval_rag.py evaluation/rag_outputs/<run_name>.jsonl
    uv run python evaluation/eval_rag.py evaluation/rag_outputs/<run_name>.jsonl --limit 2
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
from pathlib import Path
from statistics import mean

from openai import AsyncOpenAI
from ragas.embeddings import embedding_factory
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecisionWithoutReference,
    ContextRelevance,
    Faithfulness,
)

HERE = Path(__file__).parent
CONFIG = HERE / "llm_config.env"
RESULTS = HERE / "results"
CONCURRENCY = 4  # parallel judge calls; lower it if the API rate-limits


def load_config() -> dict[str, str]:
    if not CONFIG.exists():
        raise SystemExit(f"Copy {CONFIG.name}.example to {CONFIG.name} and fill it in.")
    cfg = {}
    for line in CONFIG.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            cfg[key.strip()] = value.strip()
    missing = [k for k in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL") if not cfg.get(k)]
    if missing:
        raise SystemExit(f"Fill in {', '.join(missing)} in {CONFIG}")
    return cfg


def build_metrics(cfg: dict[str, str]) -> dict:
    """metric name -> (metric, function mapping an input row to ascore() kwargs)."""
    client = AsyncOpenAI(base_url=cfg["LLM_BASE_URL"], api_key=cfg["LLM_API_KEY"])
    llm = llm_factory(cfg["LLM_MODEL"], client=client)
    metrics = {
        "faithfulness": (Faithfulness(llm=llm), lambda r: dict(
            user_input=r["question"], response=r["answer"], retrieved_contexts=r["contexts"])),
        "context_precision": (ContextPrecisionWithoutReference(llm=llm), lambda r: dict(
            user_input=r["question"], response=r["answer"], retrieved_contexts=r["contexts"])),
        "context_relevance": (ContextRelevance(llm=llm), lambda r: dict(
            user_input=r["question"], retrieved_contexts=r["contexts"])),
    }
    if cfg.get("EMBEDDING_MODEL"):
        embeddings = embedding_factory("openai", model=cfg["EMBEDDING_MODEL"], client=client)
        metrics["answer_relevancy"] = (AnswerRelevancy(llm=llm, embeddings=embeddings), lambda r: dict(
            user_input=r["question"], response=r["answer"]))
    else:
        print("EMBEDDING_MODEL is empty: skipping answer_relevancy (it needs embeddings).")
    return metrics


async def score_all(rows: list[dict], metrics: dict) -> list[dict]:
    gate = asyncio.Semaphore(CONCURRENCY)

    async def one(i: int, name: str):
        metric, to_kwargs = metrics[name]
        async with gate:
            try:
                result = await metric.ascore(**to_kwargs(rows[i]))
                # Some metrics (e.g. context_relevance) return NaN instead of raising on API errors.
                if result.value is None or math.isnan(result.value):
                    return i, name, None, "metric returned NaN (judge call failed?)"
                return i, name, result.value, ""
            except Exception as e:  # keep going; failed calls are reported, not scored
                return i, name, None, f"{type(e).__name__}: {e}"

    scored = [{"question": r["question"]} for r in rows]
    tasks = [one(i, name) for i in range(len(rows)) for name in metrics]
    for i, name, value, error in await asyncio.gather(*tasks):
        scored[i][name] = value
        if error:
            scored[i][f"{name}_error"] = error
            print(f"  row {i + 1} {name} failed: {error[:200]}")
    return scored


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("outputs", help="JSONL with question, contexts, answer")
    parser.add_argument("--limit", type=int, help="only score the first N queries (cheap trial run)")
    args = parser.parse_args()

    with open(args.outputs, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    for i, r in enumerate(rows, start=1):
        if not (isinstance(r.get("question"), str) and isinstance(r.get("answer"), str)
                and isinstance(r.get("contexts"), list) and r["contexts"]):
            raise SystemExit(f"Line {i}: need question (str), answer (str), contexts (non-empty list)")
    rows = rows[: args.limit] if args.limit else rows

    cfg = load_config()
    metrics = build_metrics(cfg)
    print(f"Scoring {len(rows)} queries x {len(metrics)} metrics with {cfg['LLM_MODEL']}")
    scored = asyncio.run(score_all(rows, metrics))

    summary = {"run": Path(args.outputs).stem, "judge_model": cfg["LLM_MODEL"], "n": len(rows)}
    for name in metrics:
        values = [s[name] for s in scored if s[name] is not None]
        summary[name] = round(mean(values), 4) if values else None
        summary[f"{name}_failed"] = len(rows) - len(values)
        print(f"  {name}: {summary[name]} ({len(values)}/{len(rows)} scored)")

    RESULTS.mkdir(exist_ok=True)
    stem = RESULTS / f"rag_{summary['run']}"
    with open(f"{stem}.csv", "w", encoding="utf-8-sig", newline="") as f:
        fields = sorted({k for s in scored for k in s}, key=lambda k: (k != "question", k))
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(scored)
    Path(f"{stem}.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote {stem}.csv and {stem}.json")


if __name__ == "__main__":
    main()

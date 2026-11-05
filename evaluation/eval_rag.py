"""RAGAS 0.4 answer/context metrics with input provenance and abstention coverage.

python evaluation/eval_rag.py results/paired_checker/rag_outputs/checker_new.jsonl \
    --env-file /absolute/path/local.env --output-dir results/paired_ragas

The original positional input, --limit and --metrics remain supported. Partial
metric updates require identical input bytes, row order and judge settings.
Answer-dependent metrics exclude abstentions; context_relevance still scores them.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
from statistics import mean
import tempfile
from urllib.parse import urlsplit

HERE = Path(__file__).parent
CONFIG = HERE / "llm_config.env"
RESULTS = HERE / "results"
CONCURRENCY = 4
METRIC_NAMES = ("faithfulness", "answer_relevancy", "context_precision", "context_relevance")
# Without-reference precision judges contexts against the supplied answer.
ANSWER_DEPENDENT = frozenset(("faithfulness", "answer_relevancy", "context_precision"))
SCORE_SCOPE = {
    "answer_metrics": "mean over non-abstained rows with finite successful scores; failures excluded and counted",
    "context_precision": "without-reference variant uses the answer; abstentions excluded",
    "context_relevance": "question and contexts only; includes abstentions",
    "comparison": "report answer coverage and metric valid_n; answer means may use different subsets across arms",
    "judge": "automated LLM evaluation, not human ground truth",
}
JUDGE_SETTINGS = {"max_tokens": 4096, "temperature": 0, "thinking": "disabled",
                  "timeout_seconds": 120, "client_max_retries": 2, "concurrency": CONCURRENCY}


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical_hash(value) -> str:
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":"), allow_nan=False).encode())


def load_config(env_file=None, *, environ=None) -> dict[str, str]:
    """Read dotenv as data. Explicit file values override environment aliases."""
    path = Path(env_file) if env_file is not None else CONFIG
    if env_file is not None and not path.is_file():
        raise ValueError(f"Environment file not found: {path}")
    file_cfg = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key, value = key.strip().removeprefix("export "), value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            if value:
                file_cfg[key] = value
    env = os.environ if environ is None else environ
    def pick(*keys):
        return next((source[k] for source in (file_cfg, env) for k in keys if source.get(k)), "")
    deepseek = any(pick(k) for k in ("DEEPSEEK_API_KEY", "DEEPSEEK_MODEL", "DEEPSEEK_BASE_URL"))
    cfg = {"LLM_API_KEY": pick("LLM_API_KEY", "DEEPSEEK_API_KEY"),
           "LLM_BASE_URL": pick("LLM_BASE_URL", "DEEPSEEK_BASE_URL") or ("https://api.deepseek.com" if deepseek else ""),
           "LLM_MODEL": pick("LLM_MODEL", "DEEPSEEK_MODEL") or ("deepseek-flash" if deepseek else ""),
           "EMBEDDING_MODEL": pick("EMBEDDING_MODEL"),
           "EMBEDDING_PROVIDER": pick("EMBEDDING_PROVIDER") or "openai"}
    if any(not cfg[k] for k in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL")):
        raise ValueError("Provide LLM_BASE_URL/LLM_API_KEY/LLM_MODEL or DEEPSEEK_API_KEY with optional DEEPSEEK_BASE_URL/DEEPSEEK_MODEL")
    url = urlsplit(cfg["LLM_BASE_URL"])
    if url.scheme not in ("http", "https") or not url.netloc or url.username or url.password or url.query or url.fragment:
        raise ValueError("Judge endpoint must be an HTTP(S) base URL without credentials, query or fragment")
    cfg["LLM_BASE_URL"] = cfg["LLM_BASE_URL"].rstrip("/")
    if cfg["EMBEDDING_PROVIDER"] not in ("openai", "huggingface"):
        raise ValueError("EMBEDDING_PROVIDER must be openai or huggingface")
    return cfg


def available_metrics(cfg):
    return [name for name in METRIC_NAMES if name != "answer_relevancy" or cfg.get("EMBEDDING_MODEL")]


def build_metrics(cfg: dict[str, str], names=None) -> dict:
    """Heavy dependencies are loaded only after input/merge preflight succeeds."""
    from openai import AsyncOpenAI
    from ragas.embeddings import HuggingFaceEmbeddings, embedding_factory
    from ragas.llms import llm_factory
    from ragas.metrics.collections import (AnswerRelevancy, ContextPrecisionWithoutReference,
                                           ContextRelevance, Faithfulness)
    names = names or available_metrics(cfg)
    client = AsyncOpenAI(base_url=cfg["LLM_BASE_URL"], api_key=cfg["LLM_API_KEY"],
                         timeout=JUDGE_SETTINGS["timeout_seconds"], max_retries=JUDGE_SETTINGS["client_max_retries"])
    llm = llm_factory(cfg["LLM_MODEL"], client=client, max_tokens=JUDGE_SETTINGS["max_tokens"],
                      temperature=JUDGE_SETTINGS["temperature"], extra_body={"thinking": {"type": "disabled"}})
    metrics = {}
    if "faithfulness" in names:
        metrics["faithfulness"] = (Faithfulness(llm=llm), lambda r: dict(
            user_input=r["question"], response=r["answer"], retrieved_contexts=r["contexts"]))
    if "context_precision" in names:
        metrics["context_precision"] = (ContextPrecisionWithoutReference(llm=llm), lambda r: dict(
            user_input=r["question"], response=r["answer"], retrieved_contexts=r["contexts"]))
    if "context_relevance" in names:
        metrics["context_relevance"] = (ContextRelevance(llm=llm), lambda r: dict(
            user_input=r["question"], retrieved_contexts=r["contexts"]))
    if "answer_relevancy" in names:
        embeddings = (HuggingFaceEmbeddings(model=cfg["EMBEDDING_MODEL"])
                      if cfg["EMBEDDING_PROVIDER"] == "huggingface" else
                      embedding_factory("openai", model=cfg["EMBEDDING_MODEL"], client=client))
        metrics["answer_relevancy"] = (AnswerRelevancy(llm=llm, embeddings=embeddings), lambda r: dict(
            user_input=r["question"], response=r["answer"]))
    return metrics


def read_rows(path, limit=None):
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
        raise ValueError("--limit must be a positive integer")
    raw = Path(path).read_bytes()
    rows = []
    for line_number, line in enumerate(raw.decode("utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if (not isinstance(row, dict) or not isinstance(row.get("question"), str) or not row["question"].strip()
                or not isinstance(row.get("answer"), str) or not isinstance(row.get("contexts"), list)
                or not row["contexts"] or any(not isinstance(c, str) or not c.strip() for c in row["contexts"])):
            raise ValueError(f"Line {line_number}: need nonempty question, string answer and nonempty list of context strings")
        if "abstained" in row and not isinstance(row["abstained"], bool):
            raise ValueError(f"Line {line_number}: abstained must be boolean")
        if row.get("status") in ("failed", "in_progress"):
            raise ValueError(f"Line {line_number}: failed/in-progress generation must not be scored as an answer")
        rid = row.get("query_id")
        if rid is not None and (not isinstance(rid, str) or not rid.strip()):
            raise ValueError(f"Line {line_number}: query_id must be a nonempty string")
        row = dict(row, row_id=rid or f"row-{len(rows)+1:05d}", row_sha256=canonical_hash(row),
                   abstained=row.get("abstained", False) or not row["answer"].strip())
        rows.append(row)
    if not rows:
        raise ValueError("Input contains no rows")
    if len({r["row_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate query_id/row_id in input")
    return rows[:limit], {"input_sha256": sha256(raw), "input_total_rows": len(rows)}


def provenance(rows, source, cfg):
    versions = {}
    for package in ("ragas", "openai", "sentence-transformers"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return {"schema_version": 2, **source, "row_ids": [r["row_id"] for r in rows],
            "row_sha256": [r["row_sha256"] for r in rows],
            "judge_config": {k: cfg[k] for k in ("LLM_BASE_URL", "LLM_MODEL", "EMBEDDING_MODEL", "EMBEDDING_PROVIDER")},
            "judge_settings": JUDGE_SETTINGS, "score_scope": SCORE_SCOPE,
            "package_versions": versions, "evaluator_sha256": sha256(Path(__file__).read_bytes())}


async def score_all(rows: list[dict], metrics: dict) -> list[dict]:
    gate = asyncio.Semaphore(CONCURRENCY)
    async def one(i, name):
        if rows[i]["abstained"] and name in ANSWER_DEPENDENT:
            return i, name, None, "skipped_abstention", ""
        metric, to_kwargs = metrics[name]
        async with gate:
            try:
                result = await metric.ascore(**to_kwargs(rows[i]))
                value = result.value
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    return i, name, None, "failed", "Metric returned a missing or non-finite score"
                return i, name, float(value), "scored", ""
            except Exception as exc:
                # Provider exception bodies/URLs may contain credentials. Do not persist them.
                return i, name, None, "failed", f"{type(exc).__name__}: judge call failed (provider details omitted)"
    scored = [{"row_id": r["row_id"], "row_sha256": r["row_sha256"], "question": r["question"],
               "abstained": r["abstained"]} for r in rows]
    for i, name, value, status, error in await asyncio.gather(*(one(i, name) for i in range(len(rows)) for name in metrics)):
        scored[i][name] = value
        scored[i][f"{name}_status"] = status
        scored[i][f"{name}_error"] = error
    return scored


def load_prior(stem, expected):
    summary_path, csv_path = Path(f"{stem}.json"), Path(f"{stem}.csv")
    if not summary_path.exists() and not csv_path.exists():
        return None
    if not summary_path.exists() or not csv_path.exists():
        raise ValueError("Partial results are incomplete; choose a new output directory")
    old = json.loads(summary_path.read_text())
    if old.get("provenance") != expected:
        raise ValueError("Partial merge refused: input hash, row ordering, judge settings or evaluator version differ; use a new output directory")
    if old.get("csv_sha256") != sha256(csv_path.read_bytes()):
        raise ValueError("Partial merge refused: saved score CSV changed or has no integrity hash")
    with csv_path.open(encoding="utf-8-sig", newline="") as file:
        earlier = list(csv.DictReader(file))
    if ([r.get("row_id") for r in earlier] != expected["row_ids"] or
            [r.get("row_sha256") for r in earlier] != expected["row_sha256"]):
        raise ValueError("Partial merge refused: score row ordering differs")
    return earlier


def merge_scores(scored, earlier, selected):
    if earlier is None:
        return scored
    for old, new in zip(earlier, scored):
        for name in METRIC_NAMES:
            if name in selected or name not in old:
                continue
            new[name] = float(old[name]) if old[name] else None
            new[f"{name}_status"] = old.get(f"{name}_status", "failed")
            new[f"{name}_error"] = old.get(f"{name}_error", "")
    return scored


def summarize(scored, run, cfg, identity):
    names = [name for name in METRIC_NAMES if any(name in row for row in scored)]
    abstained = sum(row["abstained"] for row in scored)
    summary = {"run": run, "judge_model": cfg["LLM_MODEL"], "n": len(scored),
               "n_answered": len(scored)-abstained, "n_abstained": abstained,
               "answer_coverage": (len(scored)-abstained)/len(scored), "metrics": names,
               "provenance": identity, "score_scope": SCORE_SCOPE}
    for name in names:
        valid = [r[name] for r in scored if r.get(f"{name}_status") == "scored" and r.get(name) is not None]
        skipped = sum(r.get(f"{name}_status") == "skipped_abstention" for r in scored)
        eligible = len(scored)-skipped
        summary[name] = round(mean(valid), 4) if valid else None
        summary[f"{name}_valid_n"] = len(valid)
        summary[f"{name}_eligible_n"] = eligible
        summary[f"{name}_failed"] = sum(r.get(f"{name}_status") == "failed" for r in scored)
        summary[f"{name}_skipped_abstention"] = skipped
        summary[f"{name}_coverage"] = len(valid)/eligible if eligible else None
    summary["status"] = "completed_with_failures" if any(summary[f"{n}_failed"] for n in names) else "completed"
    return summary


def save_results(stem, scored, summary):
    stem.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in scored for k in row}, key=lambda k: (k != "question", k))
    # Atomic files plus CSV digest make a crash between writes detectable on merge.
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8-sig", newline="", dir=stem.parent, delete=False) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(scored)
        csv_tmp = Path(file.name)
    summary["csv_sha256"] = sha256(csv_tmp.read_bytes())
    os.replace(csv_tmp, f"{stem}.csv")
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=stem.parent, delete=False) as file:
        json.dump(summary, file, ensure_ascii=False, indent=2, allow_nan=False)
        json_tmp = file.name
    os.replace(json_tmp, f"{stem}.json")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outputs", help="JSONL with question, contexts, answer; optional query_id and abstained")
    parser.add_argument("--limit", type=int, help="score only first N rows; partial merge requires same selection")
    parser.add_argument("--metrics", nargs="+", help="score selected metrics and safely merge unchanged other metrics")
    parser.add_argument("--env-file", type=Path, help="LLM_* or DEEPSEEK_* dotenv; default evaluation/llm_config.env")
    parser.add_argument("--output-dir", type=Path, default=RESULTS)
    args = parser.parse_args(argv)
    try:
        rows, source = read_rows(args.outputs, args.limit)
        cfg = load_config(args.env_file)
        names = args.metrics or available_metrics(cfg)
        unknown = set(names)-set(available_metrics(cfg))
        if unknown:
            raise ValueError(f"Unknown or unavailable metrics: {sorted(unknown)}; answer_relevancy requires EMBEDDING_MODEL")
        identity = provenance(rows, source, cfg)
        run = Path(args.outputs).stem
        stem = args.output_dir / f"rag_{run}"
        earlier = load_prior(stem, identity) if args.metrics else None  # Before model/embedding initialization or paid calls.
        metrics = build_metrics(cfg, names)
        print(f"Scoring {len(rows)} queries x {len(metrics)} metrics with {cfg['LLM_MODEL']}")
        scored = merge_scores(asyncio.run(score_all(rows, metrics)), earlier, names)
        summary = summarize(scored, run, cfg, identity)
        save_results(stem, scored, summary)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, "RAG evaluation failed: " + str(exc) + "\n")
    for name in summary["metrics"]:
        print(f"  {name}: {summary[name]} ({summary[name+'_valid_n']}/{summary[name+'_eligible_n']} eligible scored; "
              f"{summary[name+'_failed']} failed; {summary[name+'_skipped_abstention']} abstentions skipped)")
    print(f"Answer coverage: {summary['n_answered']}/{summary['n']}; wrote {stem}.csv and {stem}.json")
    return 1 if summary["status"] == "completed_with_failures" else 0


if __name__ == "__main__":
    raise SystemExit(main())

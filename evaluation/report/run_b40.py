"""Run generation on the three frozen B40 retrieval variants (paid API calls).

    python evaluation/report/run_b40.py --env-file /private/key.env \
        --output-dir results/b40_new --variants B40_learned

Every question uses a fresh uncached classification/summary run with the checker
off. Timings exclude retrieval, query rewriting, and reranking. Results, RAGAS
exports, timings, and a provenance manifest stay inside the required new/empty
output directory. Historical runs are never reused; failures stop the run.

For a checker comparison on identical saved drafts, use
``evaluation/report/run_paired_checker.py``. The old checker variant names have
been removed: regenerating drafts with the same current checker is not a valid
comparison of checker versions.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INPUTS = Path(__file__).parent / "b40_inputs"
QUERIES = ROOT / "evaluation/retrieval_ablation_40_v2/queries.jsonl"
VARIANTS = {
    "B40_bm25": "01_bm25",
    "B40_dense": "02_dense",
    "B40_learned": "08_learned_v2",
}
ASPECTS = {"camera", "battery", "display", "performance", "service", "carrier"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def code_provenance() -> dict:
    """Hash the actual working sources, including uncommitted local fixes."""
    paths = {Path(__file__).resolve()}
    for folder in ("generation", "retrieval", "scripts"):
        paths.update((ROOT / folder).rglob("*.py"))
    hashes = {str(path.relative_to(ROOT)): sha256(path) for path in sorted(paths)}
    result = {
        "source_sha256": hashes,
        "source_fingerprint": hashlib.sha256(
            json.dumps(hashes, sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }
    try:
        result["git_head"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        result["git_dirty"] = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip())
    except (OSError, subprocess.CalledProcessError):
        result["git_head"] = None
        result["git_dirty"] = None
    return result


def preflight(args):
    """Validate the entire batch before creating output or spending API calls."""
    output = args.output_dir.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"Output directory must be new or empty: {output}")
    env_file = args.env_file.resolve() if args.env_file else None
    if env_file is not None and not env_file.is_file():
        raise ValueError(f"Environment file does not exist: {env_file}")
    queries = [json.loads(line) for line in QUERIES.read_text(encoding="utf-8-sig").splitlines()
               if line.strip()]
    if not queries:
        raise ValueError("The B40 query file is empty")
    seen = set()
    for query in queries:
        if not isinstance(query, dict):
            raise ValueError("Each query must be an object")
        qid = query.get("query_id")
        if not isinstance(qid, str) or not re.fullmatch(r"Q\d{3}", qid) or qid in seen:
            raise ValueError(f"Invalid or duplicate query_id: {qid!r}")
        seen.add(qid)
        if not isinstance(query.get("query"), str) or not query["query"].strip():
            raise ValueError(f"Empty query text: {qid}")
        aspects = query.get("aspects", [])
        if not isinstance(aspects, list) or any(a not in ASPECTS for a in aspects):
            raise ValueError(f"Invalid aspects: {qid}")
    variants = list(dict.fromkeys(args.variants))
    jobs = []
    for variant in variants:
        if variant not in VARIANTS:
            raise ValueError(f"Unknown retrieval variant: {variant}")
        for query in queries:
            source = INPUTS / VARIANTS[variant] / f"{query['query_id']}.json"
            chunks = json.loads(source.read_text(encoding="utf-8-sig"))
            if not isinstance(chunks, list) or not chunks:
                raise ValueError(f"Expected a nonempty chunk list: {source}")
            jobs.append({"variant": variant, "query": query, "source": source,
                         "input_sha256": sha256(source)})
    return output, env_file, queries, variants, jobs


def read_export(path: Path, key: str, question: str) -> dict:
    lines = [line for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError(f"Expected exactly one export record: {path}")
    record = json.loads(lines[0])
    if not isinstance(record, dict) or record.get(key) != question:
        raise ValueError(f"Export question does not match this run: {path}")
    return record


def write_manifest(path: Path, manifest: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def execute(args, *, run_command=None) -> int:
    output, env_file, queries, variants, jobs = preflight(args)
    manifest = {
        "schema_version": 1, "status": "running", "started_at": utc_now(),
        "scope": "classification_and_summary_on_frozen_retrieval; checker_off; retrieval_excluded",
        "checker_enabled": False, "cache_enabled": False,
        "variants": variants, "n_questions": len(queries),
        "queries_source": str(QUERIES.resolve()), "queries_sha256": sha256(QUERIES),
        "code": code_provenance(), "env_file_supplied": env_file is not None,
        "inputs": [{"variant": job["variant"], "query_id": job["query"]["query_id"],
                    "path": str(job["source"].resolve()), "sha256": job["input_sha256"]}
                   for job in jobs],
        "completed": [], "failures": [],
    }
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "run_manifest.json"
    # Exclusive creation also prevents two instances claiming the same empty path.
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    run_command = run_command or subprocess.run
    exports = {variant: {"rag_outputs": [], "timings": []} for variant in variants}
    current = {}
    try:
        for job in jobs:
            variant, query, source = job["variant"], job["query"], job["source"]
            current = {"variant": variant, "query_id": query["query_id"]}
            if sha256(source) != job["input_sha256"]:
                raise ValueError(f"Input changed after preflight: {source}")
            result_path = output / "runs" / variant / f"{query['query_id']}.json"
            result_path.parent.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, "-m", "scripts.group_c", "run", "--input", str(source),
                   "--question", query["query"], "--language", "en", "--no-cache", "--no-check",
                   "--output", str(result_path)]
            if env_file:
                cmd += ["--env-file", str(env_file)]
            if query.get("aspects"):
                cmd += ["--aspects", *query["aspects"]]
            done = run_command(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
            if done.returncode:
                # Never persist raw child output, which may contain provider secrets.
                raise RuntimeError(f"Group C exited with status {done.returncode}")
            result = json.loads(result_path.read_text(encoding="utf-8-sig"))
            if (not isinstance(result, dict) or result.get("question") != query["query"]
                    or result.get("verification_enabled") is not False):
                raise ValueError(f"Result does not match the requested unchecked run: {result_path}")
            for suffix, folder, key in (("rag", "rag_outputs", "question"), ("timings", "timings", "query")):
                exports[variant][folder].append(read_export(
                    result_path.with_suffix(f".{suffix}.jsonl"), key, query["query"]))
            if sha256(source) != job["input_sha256"]:
                raise ValueError(f"Input changed during run: {source}")
            manifest["completed"].append({**current, "result": str(result_path.relative_to(output)),
                                          "result_sha256": sha256(result_path),
                                          "model": result.get("api", {}).get("model")})
            write_manifest(manifest_path, manifest)
            print(variant, query["query_id"], "ok", flush=True)
        current = {}
        if sha256(QUERIES) != manifest["queries_sha256"]:
            raise ValueError("Query file changed during run")
        if code_provenance()["source_fingerprint"] != manifest["code"]["source_fingerprint"]:
            raise ValueError("Pipeline source code changed during run")
        for variant, folders in exports.items():
            for folder, rows in folders.items():
                target = output / folder / f"{variant}.jsonl"
                target.parent.mkdir(exist_ok=True)
                target.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                                  encoding="utf-8")
        manifest["status"] = "complete"
    except (OSError, ValueError, RuntimeError) as exc:
        manifest["status"] = "failed"
        manifest["failures"].append({**current, "error": str(exc)})
        print(f"B40 run failed: {exc}", file=sys.stderr)
    finally:
        manifest["finished_at"] = utc_now()
        write_manifest(manifest_path, manifest)
    return 0 if manifest["status"] == "complete" else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, help="Private API environment file (never copied to results)")
    parser.add_argument("--output-dir", type=Path, required=True, help="New or empty directory for this run")
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    args = parser.parse_args(argv)
    try:
        return execute(args)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"B40 preflight failed: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())

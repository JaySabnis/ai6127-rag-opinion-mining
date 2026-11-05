"""Check frozen B40_learned drafts, comparing checker off with the current checker.

No retrieval, classification or summary generation runs here. Input files retain
their original bytes. The two arms share the same evidence, sentences and labels.
This measures only the checker intervention, not end-to-end system latency or
the historical pre-fix checker. A failed request is never a completed result.

python -m evaluation.report.run_paired_checker --env-file /private/key.env \
    --output-dir results/paired_checker --limit 2 --max-api-calls 4
python -m evaluation.report.run_paired_checker --env-file /private/key.env \
    --output-dir results/paired_checker --resume --max-api-calls 80
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generation.client import DeepSeekClient, write_json
from generation.schema import normalize_chunks
from generation.verifier import FaithfulnessChecker

SCHEMA_VERSION = 1
SCOPE = ("checker_only_paired_frozen_B40_learned; off_vs_current_checker; "
         "no_retrieval_classification_or_generation; no_historical_checker_reproduction")
CODE_FILES = ("generation/verifier.py", "generation/schema.py", "generation/client.py",
              "retrieval/product_identity.py", "evaluation/report/run_paired_checker.py")
CORE_FIELDS = ("question", "chunks", "sentences", "predictions", "statistics", "draft_answer")


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def digest_json(value):
    return digest_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                   separators=(",", ":"), allow_nan=False).encode("utf-8"))


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def render(sentences):
    return "\n".join(s["text"] + " " + " ".join(f"[{c}]" for c in s["citations"])
                     for s in sentences)


def code_provenance():
    hashes = {name: digest_bytes((ROOT / name).read_bytes()) for name in CODE_FILES}
    def git(*args):
        try:
            result = subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True)
            return result.stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    dirty = git("status", "--porcelain", "--untracked-files=no")
    return {"git_head": git("rev-parse", "HEAD"), "git_tracked_dirty": bool(dirty) if dirty is not None else None,
            "source_sha256": hashes, "source_fingerprint": digest_json(hashes)}


def load_sources(input_dir):
    sources = []
    for path in sorted(Path(input_dir).glob("Q*.json")):
        raw = path.read_bytes()
        run = json.loads(raw)
        if not isinstance(run, dict) or any(key not in run for key in CORE_FIELDS):
            raise ValueError(f"{path.name}: expected a complete frozen Group C run")
        if run.get("verification_enabled") is not False:
            raise ValueError(f"{path.name}: source must be the checker-off B40_learned run")
        if not isinstance(run["question"], str) or not run["question"].strip():
            raise ValueError(f"{path.name}: question must be nonempty")
        normalize_chunks(run["chunks"])
        if not isinstance(run["predictions"], list) or not isinstance(run["statistics"], list):
            raise ValueError(f"{path.name}: frozen classification/statistics must be lists")
        sentences = run["sentences"]
        if not isinstance(sentences, list):
            raise ValueError(f"{path.name}: frozen sentences must be a list")
        for sentence in sentences:
            if (not isinstance(sentence, dict) or not isinstance(sentence.get("text"), str)
                    or not sentence["text"].strip() or not isinstance(sentence.get("citations"), list)
                    or any(not isinstance(c, str) for c in sentence["citations"])):
                raise ValueError(f"{path.name}: invalid frozen sentence")
        if run["draft_answer"] != render(sentences):
            raise ValueError(f"{path.name}: frozen draft does not match frozen sentences")
        core = {key: run[key] for key in CORE_FIELDS}
        sources.append({"query_id": path.stem, "path": path, "raw": raw, "run": run,
                        "input_sha256": digest_bytes(raw), "core_sha256": digest_json(core)})
    if not sources:
        raise ValueError("No Q*.json frozen runs found; refusing to regenerate missing drafts")
    return sources


def validate_pricing(pricing):
    if pricing is None:
        return None
    required = ("input_per_million", "cached_input_per_million", "output_per_million")
    if not isinstance(pricing, dict) or any(k not in pricing for k in required):
        raise ValueError("Pricing JSON requires input_per_million, cached_input_per_million, output_per_million")
    for key in required:
        value = pricing[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("Pricing values must be finite nonnegative numbers")
    if not pricing.get("currency") or not pricing.get("source_url") or not pricing.get("as_of"):
        raise ValueError("Pricing JSON must specify currency, source_url and as_of")
    return pricing


def telemetry_delta(before, after, pricing=None):
    usage = copy.deepcopy(after.get("usage", [])[len(before.get("usage", [])):])
    calls = after.get("api_calls", 0) - before.get("api_calls", 0)
    hits = after.get("cache_hits", 0) - before.get("cache_hits", 0)
    prompt = sum(u.get("prompt_tokens", 0) for u in usage)
    output = sum(u.get("completion_tokens", 0) for u in usage)
    cached = sum(u.get("prompt_cache_hit_tokens", (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)) for u in usage)
    known = all(isinstance(u.get("prompt_tokens"), int) and isinstance(u.get("completion_tokens"), int) for u in usage)
    estimated = None
    if pricing is not None and known and 0 <= cached <= prompt:
        estimated = ((prompt-cached)*pricing["input_per_million"] + cached*pricing["cached_input_per_million"]
                     + output*pricing["output_per_million"]) / 1_000_000
    return {"api_calls": calls, "cache_hits": hits, "usage": usage,
            "prompt_tokens": prompt, "cached_prompt_tokens": cached, "completion_tokens": output,
            "total_tokens": sum(u.get("total_tokens", u.get("prompt_tokens", 0)+u.get("completion_tokens", 0)) for u in usage),
            "requests_without_usage": max(0, calls-len(usage)),
            "estimated_cost": estimated, "currency": pricing.get("currency") if pricing else None,
            "cost_scope": "estimate_for_recorded_usage_only; requests_without_usage_may_have_unknown_cost",
            "pricing": pricing}


def result_exports(output_dir, sources, manifest):
    pairs = []
    for source in sources:
        path = output_dir / "pairs" / (source["query_id"] + ".json")
        if path.exists():
            pair = json.loads(path.read_text())
            if pair.get("input_sha256") != source["input_sha256"] or pair.get("identity_sha256") != manifest["identity_sha256"]:
                raise ValueError("Pair identity mismatch: " + source["query_id"])
            pairs.append(pair)
    complete = [p for p in pairs if p.get("status") == "completed"]
    failed = [p["query_id"] for p in pairs if p.get("status") != "completed"]
    for arm in ("off", "new"):
        records = [{"query_id": p["query_id"], "question": p["frozen"]["question"],
                    "contexts": [c["text"] for c in p["frozen"]["chunks"]],
                    "answer": p[arm]["answer"], "abstained": p[arm]["abstained"],
                    "scope": SCOPE} for p in complete]
        path = output_dir / "rag_outputs" / f"checker_{arm}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False)+"\n" for r in records))
    attempts = [a for p in pairs for a in p["attempts"]]
    telemetry = [a["api"] for a in attempts if "api" in a]
    summary = {"scope": SCOPE, "source_queries": len(sources), "completed_pairs": len(complete),
               "failed_or_interrupted_query_ids": failed,
               "pending_query_ids": sorted({s["query_id"] for s in sources}-{p["query_id"] for p in pairs}),
               "status": "completed" if len(complete) == len(sources) else "incomplete",
               "new_abstentions": sum(p["new"]["abstained"] for p in complete),
               "draft_sentences": sum(len(p["frozen"]["sentences"]) for p in complete),
               "retained_sentences": sum(len(p["new"]["retained_sentences"]) for p in complete),
               "api_calls_all_recorded_attempts": sum(t["api_calls"] for t in telemetry),
               "prompt_tokens_all_recorded_attempts": sum(t["prompt_tokens"] for t in telemetry),
               "completion_tokens_all_recorded_attempts": sum(t["completion_tokens"] for t in telemetry),
               "requests_without_usage": sum(t["requests_without_usage"] for t in telemetry),
               "interrupted_attempts_without_telemetry": sum("api" not in a for a in attempts),
               "estimated_cost_recorded_attempts": (sum(t["estimated_cost"] for t in telemetry)
                       if telemetry and all(t["estimated_cost"] is not None for t in telemetry) else None),
               "checker_elapsed_s_all_recorded_attempts": sum(a.get("checker_elapsed_s", 0) for a in attempts),
               "latency_scope": "checker calls only; frozen off arm has no new inference latency; not end-to-end throughput",
               "rag_export_scope": "same successfully completed pairs in both arms; abstentions retained as empty answers",
               "completed_query_ids": [p["query_id"] for p in complete], "updated_at": utc_now()}
    summary["completed_pair_coverage"] = len(complete) / len(sources)
    summary["new_answer_coverage_on_completed_pairs"] = (
        (len(complete)-summary["new_abstentions"]) / len(complete) if complete else None)
    summary["new_sentence_retention_ratio"] = (
        summary["retained_sentences"] / summary["draft_sentences"] if summary["draft_sentences"] else None)
    write_json(output_dir / "summary.json", summary)
    manifest.update(status=summary["status"], completed_query_ids=summary["completed_query_ids"], updated_at=utc_now())
    write_json(output_dir / "manifest.json", manifest)
    return summary


def run_paired(input_dir, output_dir, client, *, limit=None, resume=False, pricing=None):
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
        raise ValueError("limit must be a positive integer")
    pricing = validate_pricing(pricing)
    sources = load_sources(input_dir)  # Preflight every source before a paid call.
    output_dir = Path(output_dir)
    provenance = code_provenance()
    identity = {"schema_version": SCHEMA_VERSION, "scope": SCOPE,
                "input_sha256": {s["query_id"]: s["input_sha256"] for s in sources},
                "code_source_sha256": provenance["source_sha256"],
                "model": client.model, "base_url": client.base_url, "pricing": pricing}
    fingerprint = digest_json(identity)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        if not resume:
            raise ValueError("Output already contains a run; use --resume or a new output directory")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("identity_sha256") != fingerprint:
            raise ValueError("Resume refused: input, code, model, endpoint or pricing changed; use a new output directory")
    else:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise ValueError("Output directory is nonempty without a valid manifest")
        manifest = {"identity": identity, "identity_sha256": fingerprint, "created_at": utc_now(),
                    "code": provenance, "status": "incomplete", "completed_query_ids": []}
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(manifest_path, manifest)
    (output_dir / "frozen_inputs").mkdir(exist_ok=True)
    (output_dir / "pairs").mkdir(exist_ok=True)
    for source in sources:
        frozen = output_dir / "frozen_inputs" / (source["query_id"]+".json")
        if frozen.exists():
            if digest_bytes(frozen.read_bytes()) != source["input_sha256"]:
                raise ValueError("Frozen input changed: " + source["query_id"])
        else:
            frozen.write_bytes(source["raw"])
    for source in sources[:limit]:
        qid, original = source["query_id"], source["run"]
        path = output_dir / "pairs" / (qid+".json")
        if path.exists():
            pair = json.loads(path.read_text())
            if pair.get("identity_sha256") != fingerprint or pair.get("core_sha256") != source["core_sha256"]:
                raise ValueError("Pair fingerprint changed: " + qid)
            if pair.get("status") == "completed":
                continue
        else:
            pair = {"query_id": qid, "identity_sha256": fingerprint,
                    "input_sha256": source["input_sha256"], "core_sha256": source["core_sha256"],
                    "frozen": {k: copy.deepcopy(original[k]) for k in CORE_FIELDS},
                    "off": {"status": "not_checked", "verification_enabled": False,
                            "answer": original["draft_answer"], "abstained": not bool(original["draft_answer"])},
                    "attempts": []}
        attempt = {"attempt": len(pair["attempts"])+1, "status": "started", "started_at": utc_now()}
        pair["attempts"].append(attempt)
        pair["status"] = "in_progress"
        write_json(path, pair)  # A crash is visible; it cannot look like success.
        before = copy.deepcopy(client.telemetry())
        started = time.perf_counter()
        try:
            checks = FaithfulnessChecker(client).check(copy.deepcopy(original["sentences"]),
                       copy.deepcopy(original["chunks"]), language=original.get("language", "en"))
            retained = [s for s in checks if s["status"] == "supported"]
            answer = render(retained)
            pair["new"] = {"status": "completed", "verification_enabled": True, "checks": checks,
                           "retained_sentences": retained, "answer": answer, "abstained": not bool(answer),
                           "display_answer": answer or "No summary claims passed the evidence check."}
            pair["status"] = attempt["status"] = "completed"
        except (RuntimeError, ValueError, OSError) as exc:
            error = {"type": type(exc).__name__, "message": str(exc)}
            pair["new"] = {"status": "failed", "verification_enabled": True, "answer": None,
                           "checks": None, "error": error}
            pair["status"] = attempt["status"] = "failed"
            attempt["error"] = error
        attempt["checker_elapsed_s"] = time.perf_counter()-started
        attempt["finished_at"] = utc_now()
        attempt["api"] = telemetry_delta(before, client.telemetry(), pricing)
        write_json(path, pair)
        result_exports(output_dir, sources, manifest)
    summary = result_exports(output_dir, sources, manifest)
    selected = {s["query_id"] for s in sources[:limit]}
    summary["selected_queries_completed"] = selected <= set(summary["completed_query_ids"])
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "results/group_c/b40/B40_learned")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-api-calls", type=int, default=80, help="Request budget for this invocation, including retries")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pricing-file", type=Path, help="Optional documented per-million token rates; otherwise cost remains unknown")
    args = parser.parse_args(argv)
    if args.max_api_calls < 1:
        parser.error("--max-api-calls must be positive")
    try:
        pricing = json.loads(args.pricing_file.read_text()) if args.pricing_file else None
        client = DeepSeekClient(env_file=args.env_file, model=args.model, cache_dir=None, max_calls=args.max_api_calls)
        summary = run_paired(args.input_dir, args.output_dir, client, limit=args.limit, resume=args.resume, pricing=pricing)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, "Paired checker failed: " + str(exc) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["selected_queries_completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

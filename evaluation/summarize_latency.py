"""Summarize query timings without confusing downstream work with end-to-end latency.

Accepts existing Group C stage-only JSONL exports and rows containing an
authoritative ``total_s``. Explicit totals are never added to stage times.
Legacy stage-only totals are estimates, excluding startup and missing stages.

    python -m evaluation.summarize_latency timings.jsonl --output summary.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, median, quantiles

RESULTS = Path(__file__).parent / "results"
DOWNSTREAM_STAGES = {"classification_s", "generation_s", "verification_s"}
RETRIEVAL_STAGES = {"rewrite_s", "retrieve_s", "retrieval_s", "rerank_s"}
QUERY_STAGES = DOWNSTREAM_STAGES | RETRIEVAL_STAGES
SCOPES = {"downstream", "retrieval", "end_to_end", "unspecified"}


def stats(values):
    """Descriptive statistics; n makes missing stage observations visible."""
    p95 = quantiles(values, n=20, method="inclusive")[-1] if len(values) > 1 else values[0]
    return {"n": len(values), "mean": round(mean(values), 3),
            "median": round(median(values), 3), "p95": round(p95, 3),
            "max": round(max(values), 3)}


def read_timings(path):
    rows = []
    with Path(path).open(encoding="utf-8-sig") as file:
        for number, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Invalid JSON on line {number}: {exc}") from None
            if not isinstance(row, dict):
                raise ValueError(f"Line {number} must be a JSON object")
            rows.append(row)
    return rows


def summarize(rows, *, run="timings"):
    """Return a new summary, leaving caller-owned rows unchanged.

    Old Group C exports infer downstream scope from known stage names. Unknown
    timings remain descriptive statistics but are not silently added to totals.
    Mixed scopes are rejected rather than averaged together.
    """
    if not rows:
        raise ValueError("No timing rows")
    checked, scopes, bases, ignored = [], set(), set(), set()
    for number, original in enumerate(rows, 1):
        if not isinstance(original, dict):
            raise ValueError(f"Row {number} must be an object")
        row = dict(original)
        for key, value in row.items():
            if key.endswith("_s") and (isinstance(value, bool) or
                    not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0):
                raise ValueError(f"Row {number}: {key} must be a finite non-negative number")
        for key in ("n_chunks", "n_classifications"):
            if key in row and (isinstance(row[key], bool) or not isinstance(row[key], int) or row[key] < 0):
                raise ValueError(f"Row {number}: {key} must be a non-negative integer")
        present = QUERY_STAGES & row.keys()
        if "total_s" not in row and {"retrieve_s", "retrieval_s"} <= present:
            raise ValueError(f"Row {number}: use only one retrieval stage name, or provide total_s")
        if "scope" in row and "timing_scope" in row and row["scope"] != row["timing_scope"]:
            raise ValueError(f"Row {number}: scope and timing_scope disagree")
        scope = row.get("timing_scope", row.get("scope"))
        if scope is None:
            scope = ("downstream" if present and present <= DOWNSTREAM_STAGES else
                     "retrieval" if present and present <= RETRIEVAL_STAGES else "unspecified")
        if not isinstance(scope, str) or scope not in SCOPES:
            raise ValueError(f"Row {number}: timing_scope must be one of {sorted(SCOPES)}")
        if (scope == "downstream" and present & RETRIEVAL_STAGES or
                scope == "retrieval" and present & DOWNSTREAM_STAGES):
            raise ValueError(f"Row {number}: timing_scope contradicts its stage fields")
        scopes.add(scope)
        if "total_s" in row:
            bases.add("measured_total")
        elif present:
            row["total_s"] = sum(row[key] for key in present)
            if not math.isfinite(row["total_s"]):
                raise ValueError(f"Row {number}: summed stage durations are not finite")
            bases.add("sum_of_available_stages")
        else:
            raise ValueError(f"Row {number}: provide total_s or a recognized query stage")
        ignored.update(k for k in row if k.endswith("_s") and k not in QUERY_STAGES | {"total_s"})
        checked.append(row)
    if len(scopes) != 1:
        raise ValueError("Mixed timing scopes; summarize downstream, retrieval and end-to-end runs separately")
    scope = next(iter(scopes))
    summary = {"run": run, "n_queries": len(checked), "timing_scope": scope, "scope": scope,
               "total_basis": next(iter(bases)) if len(bases) == 1 else "mixed",
               "throughput_basis": "sequential_estimate_from_summed_query_durations",
               "concurrent_throughput_measured": False}
    for stage in sorted({k for row in checked for k in row if k.endswith("_s")}):
        summary[stage] = stats([row[stage] for row in checked if stage in row])
    summary["first_query_total_s"] = round(checked[0]["total_s"], 3)
    if len(checked) > 1:
        summary["subsequent_query_total_mean_s"] = round(mean(row["total_s"] for row in checked[1:]), 3)
    # Observation order provides no evidence of a cold process or warm API cache.
    summary["query_states"] = sorted({str(row.get("query_state", "unspecified")) for row in checked})
    elapsed = sum(row["total_s"] for row in checked)
    qpm = round(60 * len(checked) / elapsed, 2) if elapsed else None
    summary["serial_queries_per_minute_estimate"] = qpm
    summary["queries_per_minute"] = qpm  # compatibility alias; same stated basis
    warnings = ["Query latency excludes startup or file-writing outside the declared scope; "
                "the first row is not a cold-start measurement.",
                "Serial throughput is an estimate, not a concurrent load test or a scalability result."]
    if "sum_of_available_stages" in bases:
        warnings.append("Some totals sum only recorded stages; omitted stages and orchestration overhead are not measured.")
    if scope == "downstream":
        warnings.append("Downstream scope excludes query rewriting, retrieval, reranking and index/model loading.")
    if scope == "unspecified":
        warnings.append("The timing scope is not established; do not report this as end-to-end latency.")
    if ignored:
        summary["fields_excluded_from_derived_total"] = sorted(ignored)
    classified = [row for row in checked if row.get("classification_s", 0) > 0 and "n_chunks" in row]
    if classified:
        summary["chunks_classified_per_second"] = round(
            sum(row["n_chunks"] for row in classified) / sum(row["classification_s"] for row in classified), 1)
        summary["classification_rate_basis"] = "input_chunks / classification stage time (not aspect-label decisions)"
        summary["classification_rate_n_queries"] = len(classified)
    summary["warnings"] = warnings
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("timings", type=Path)
    parser.add_argument("--output", type=Path, help="Default: evaluation/results/latency_<input stem>.json")
    args = parser.parse_args(argv)
    try:
        summary = summarize(read_timings(args.timings), run=args.timings.stem)
        output = args.output or RESULTS / f"latency_{summary['run']}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Cannot summarize timings: {exc}\n")
    print(json.dumps(summary, indent=2, allow_nan=False))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()

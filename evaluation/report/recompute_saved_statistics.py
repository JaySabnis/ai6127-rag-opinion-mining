"""Recompute ratios from frozen predictions without inference or label changes.

Writes a separate audit directory; never overwrites historical model outputs.
Run with python -m evaluation.report.recompute_saved_statistics --output-dir PATH.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from generation.sentiment import aggregate_sentiment

ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recompute(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents:
        raise ValueError("Output must be outside the historical source directory")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be empty; historical files are never replaced")
    paths = sorted(source.glob("*/Q*.json"))
    if not paths:
        raise ValueError("No saved variant/Q*.json runs found")
    output.mkdir(parents=True, exist_ok=True)
    manifest, changes = [], []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        stats = aggregate_sentiment(data["predictions"])
        old = data["statistics"]
        # Added identity metadata alone is not a change to a numerical result.
        signature = lambda rows: sorted((r["aspect"], tuple(sorted(r["counts"].items())),
                                         r["n_relevant_reviews"], r["n_excluded_aspect_absent"])
                                        for r in rows)
        changed = signature(stats) != signature(old)
        dest = output / path.relative_to(source)
        dest.parent.mkdir(parents=True, exist_ok=True)
        payload = {"source": path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.name,
                   "source_sha256": sha256(path), "question": data["question"],
                   "statistics_before": old, "statistics": stats,
                   "numerical_change": changed,
                   "scope": "reaggregate frozen predictions only; no model inference or human-label edits"}
        dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        manifest.append({"file": path.relative_to(source).as_posix(), "sha256": sha256(path)})
        changes.append({"variant": path.parent.name, "query_id": path.stem,
                        "groups_before": len(old), "groups_after": len(stats),
                        "numerical_change": changed})
    with (output / "changes.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(changes[0]))
        writer.writeheader()
        writer.writerows(changes)
    summary = {"n_runs": len(paths), "n_numerical_changes": sum(r["numerical_change"] for r in changes),
               "input_files": manifest,
               "source_hashes": {str(p.relative_to(ROOT)): sha256(p) for p in
                   (ROOT / "generation/schema.py", ROOT / "generation/sentiment.py",
                    ROOT / "retrieval/product_identity.py", Path(__file__).resolve())}}
    (output / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "results/group_c/b40")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = recompute(args.source, args.output_dir)
    print(f"Recomputed {result['n_runs']} saved runs; {result['n_numerical_changes']} numerical changes.")


if __name__ == "__main__":
    main()

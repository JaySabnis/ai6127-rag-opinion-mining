"""
Run the Group C pipeline on Group B's 40 evaluation questions and collect the
RAGAS / latency inputs.

    python evaluation/report/run_b40.py

Each (variant, question) is one `python -m scripts.group_c run --no-cache`
call, so the timings are real. Finished runs are skipped on a rerun.
Outputs:
    results/group_c/b40/<variant>/<qid>.json (+ .rag.jsonl, .timings.jsonl)
    evaluation/rag_outputs/<variant>.jsonl
    evaluation/timings/<variant>.jsonl
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INPUTS = Path(__file__).parent / "b40_inputs"
QUERIES = ROOT / "evaluation/retrieval_ablation_40_v2/queries.jsonl"
OUT = ROOT / "results/group_c/b40"
# variant -> (retrieval config, evidence checker on?)
VARIANTS = {
    "B40_bm25": ("01_bm25", False),
    "B40_dense": ("02_dense", False),
    "B40_learned": ("08_learned_v2", False),
    "B40_learned_checker": ("08_learned_v2", True),
    # Same as above after the verifier fix (bad evidence no longer wipes the answer).
    "B40_learned_checker_fixed": ("08_learned_v2", True),
}


def main() -> None:
    queries = [json.loads(l) for l in QUERIES.read_text(encoding="utf-8").splitlines() if l.strip()]
    failures = []
    for variant, (config, check) in VARIANTS.items():
        (OUT / variant).mkdir(parents=True, exist_ok=True)
        for q in queries:
            out = OUT / variant / f"{q['query_id']}.json"
            if out.with_suffix(".rag.jsonl").exists():
                continue
            cmd = [sys.executable, "-m", "scripts.group_c", "run",
                   "--input", str(INPUTS / config / f"{q['query_id']}.json"),
                   "--question", q["query"], "--language", "en", "--no-cache",
                   "--output", str(out)]
            if q["aspects"]:
                cmd += ["--aspects", *q["aspects"]]
            if not check:
                cmd.append("--no-check")
            done = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
            status = "ok" if done.returncode == 0 else "FAILED"
            print(variant, q["query_id"], status, flush=True)
            if done.returncode:
                failures.append((variant, q["query_id"], done.stderr.strip()[-300:]))

        # Collect in question order; write without a BOM (the eval scripts reject it).
        for kind, folder in (("rag", "rag_outputs"), ("timings", "timings")):
            lines = []
            for q in queries:
                path = (OUT / variant / f"{q['query_id']}.json").with_suffix(f".{kind}.jsonl")
                if path.exists():
                    lines.append(path.read_text(encoding="utf-8").strip())
            (ROOT / "evaluation" / folder / f"{variant}.jsonl").write_text(
                "\n".join(lines) + "\n", encoding="utf-8")

    for f in failures:
        print("FAILED:", *f)
    print(f"{len(failures)} failures")


if __name__ == "__main__":
    main()

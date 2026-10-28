"""
Collect the per-run result files into Markdown tables for the Q5
ablation study.

Each system variant is evaluated as its own run, named after what is
switched on, e.g.

    baseline, rerank, rewrite, rerank+rewrite, faithfulness_filter

so evaluation/results/ holds rag_<run>.json, classifier_<run>.json and
latency_<run>.json. Run (from the repository root):

    uv run python evaluation/ablation_table.py

It prints the tables and writes them to evaluation/results/ablation.md.
"""

from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path(__file__).parent / "results"


def table(title: str, prefix: str, columns: dict) -> str:
    runs = [json.loads(p.read_text()) for p in sorted(RESULTS.glob(f"{prefix}_*.json"))]
    if not runs:
        return ""
    lines = [f"### {title}", "", "| Run | " + " | ".join(columns) + " |",
             "|---" * (len(columns) + 1) + "|"]
    for r in runs:
        cells = []
        for get in columns.values():
            try:
                value = get(r)
            except (KeyError, TypeError):
                value = None
            cells.append("-" if value is None else f"{value:.3f}" if isinstance(value, float) else str(value))
        lines.append(f"| {r['run']} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    sections = [
        table("Classification", "classifier", {
            "Subjectivity macro-F1": lambda r: r["subjectivity"]["macro avg"]["f1-score"],
            "Polarity macro-F1": lambda r: r["polarity"]["macro avg"]["f1-score"],
            "Overall macro-F1": lambda r: r["overall"]["macro avg"]["f1-score"],
            "Overall accuracy": lambda r: r["overall"]["accuracy"],
        }),
        table("RAG (RAGAS)", "rag", {
            "Faithfulness": lambda r: r["faithfulness"],
            "Answer relevancy": lambda r: r.get("answer_relevancy"),
            "Context precision": lambda r: r["context_precision"],
            "Context relevance": lambda r: r["context_relevance"],
            "Queries": lambda r: r["n"],
        }),
        table("Latency", "latency", {
            "Mean total (s)": lambda r: r["total_s"]["mean"],
            "p95 total (s)": lambda r: r["total_s"]["p95"],
            "Cold total (s)": lambda r: r["cold_total_s"],
            "Queries/min": lambda r: r["queries_per_minute"],
        }),
    ]
    text = "\n".join(s for s in sections if s)
    if not text:
        raise SystemExit(f"No result files in {RESULTS}")
    print(text)
    (RESULTS / "ablation.md").write_text(text, encoding="utf-8")
    print(f"Wrote {RESULTS / 'ablation.md'}")


if __name__ == "__main__":
    main()

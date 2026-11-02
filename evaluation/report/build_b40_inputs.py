"""
Build Group C pipeline inputs from Group B's 40-question retrieval ablation.

For each chosen retrieval configuration and question, take the top-k chunks
from results/ablation_latest/runs/<config>.trec, look up their text in the
corpus, and write evaluation/report/b40_inputs/<config>/<query_id>.json.

    python evaluation/report/build_b40_inputs.py --corpus ../data/corpus.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "results/ablation_latest/runs"
OUT = Path(__file__).parent / "b40_inputs"
CONFIGS = ["01_bm25", "02_dense", "08_learned_v2"]
KEEP = ("chunk_id", "doc_id", "text", "brand", "model", "source_url")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    ranked = defaultdict(dict)  # config -> qid -> [chunk_id by rank]
    for config in CONFIGS:
        rows = defaultdict(list)
        for line in (RUNS / f"{config}.trec").read_text(encoding="utf-8").splitlines():
            qid, _, chunk_id, rank, *_ = line.split()
            rows[qid].append((int(rank), chunk_id))
        for qid, items in rows.items():
            ranked[config][qid] = [c for _, c in sorted(items)[: args.top_k]]

    wanted = {c for per_q in ranked.values() for ids in per_q.values() for c in ids}
    chunks = {}
    with open(args.corpus, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["chunk_id"] in wanted:
                chunks[row["chunk_id"]] = {k: row.get(k) for k in KEEP}
    missing = wanted - chunks.keys()
    assert not missing, f"{len(missing)} chunk ids not in corpus, e.g. {sorted(missing)[:3]}"

    for config, per_q in ranked.items():
        (OUT / config).mkdir(parents=True, exist_ok=True)
        for qid, ids in per_q.items():
            (OUT / config / f"{qid}.json").write_text(
                json.dumps([chunks[c] for c in ids], ensure_ascii=False, indent=1), encoding="utf-8")
        print(config, len(per_q), "questions")
    print(len(chunks), "distinct chunks")


if __name__ == "__main__":
    main()

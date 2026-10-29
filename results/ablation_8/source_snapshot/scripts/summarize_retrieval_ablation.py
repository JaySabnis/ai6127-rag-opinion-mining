from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess

import ir_measures as ir
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "evaluation/retrieval_ablation_40"
OUT = ROOT / "results/ablation_8"
NAMES = {
    "01_bm25": "1. BM25 only", "02_dense": "2. Dense only", "03_rrf": "3. Hybrid RRF",
    "04_strip_model": "4. RRF + model removal", "05_aspect": "5. + retrieval aspect boost",
    "06_rerank": "6. + legacy reranker", "07_rewrite": "7. + query rewriting",
    "08_learned": "8. Replace legacy with learned reranker",
}
METRICS = ["P@10", "pooled_Recall@100", "MRR@10", "nDCG@10"]
PAIRS = [("03_rrf", "01_bm25"), ("03_rrf", "02_dense"),
         ("04_strip_model", "03_rrf"), ("05_aspect", "04_strip_model"),
         ("06_rerank", "05_aspect"), ("07_rewrite", "06_rerank"),
         ("07_rewrite", "05_aspect"), ("08_learned", "07_rewrite"), ("08_learned", "05_aspect")]


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    queries = read_jsonl(DATA / "queries.jsonl")
    qids = [q["query_id"] for q in queries]
    qrels = list(ir.read_trec_qrels(str(DATA / "qrels.trec")))
    grades = defaultdict(dict)
    for r in qrels:
        assert r.doc_id not in grades[r.query_id]
        grades[r.query_id][r.doc_id] = r.relevance
    recall_grades = {qid: dict(v) for qid,v in grades.items()}
    extra = json.loads((OUT / "supplemental_labels.json").read_text())
    supplemental = []
    for qid, items in extra.items():
        for cid, grade in items.items():
            assert cid not in grades[qid] and grade in (0,1,2,3)
            grades[qid][cid] = grade
            supplemental.append(ir.Qrel(qid,cid,grade))
    ranking_qrels = qrels + supplemental
    (OUT / "ranking_qrels.trec").write_text("".join(
        f"{r.query_id} 0 {r.doc_id} {r.relevance}\n" for r in ranking_qrels), encoding="utf8")
    measures = [ir.P(rel=2) @ 10, ir.Recall(rel=2) @ 100, ir.RR(rel=2) @ 10,
                ir.nDCG(dcg="log2", gains={0: 0, 1: 1, 2: 2, 3: 3}) @ 10]
    lookup = dict(zip(measures, METRICS, strict=True))
    old = defaultdict(list)
    for r in sorted(read_jsonl(DATA / "runs.jsonl"), key=lambda x: (x["system"], x["query_id"], x["rank"])):
        old[r["system"], r["query_id"]].append(r["doc_id"])
    rows, run_rows, replication = [], [], []
    values, errors = {}, []
    (OUT / "runs").mkdir(parents=True, exist_ok=True)
    snapshots = {qid: json.loads((OUT / "collection" / f"{qid}.json").read_text(encoding="utf-8")) for qid in qids}
    for system in NAMES:
        lines = []
        for qid in qids:
            hits = snapshots[qid]["runs"][system]
            assert len(hits) == 100, "Recall@100 requires a fresh Top100 run"
            assert len({h["chunk_id"] for h in hits}) == len(hits)
            assert len({h["doc_id"] for h in hits}) == len(hits)
            unknown = [h["chunk_id"] for h in hits[:10] if h["chunk_id"] not in grades[qid]]
            assert not unknown, (system, qid, "unjudged Top10", unknown)
            replication.append({"system": system, "query_id": qid,
                                "same_top20_as_dataset_run": old[system, qid][:20] == [h["chunk_id"] for h in hits[:20]],
                                "same_top10_as_dataset_run": old[system, qid][:10] == [h["chunk_id"] for h in hits[:10]]})
            for rank, h in enumerate(hits, 1):
                cid = h["chunk_id"]
                score = h["rerank_score"] if system in ("06_rerank", "07_rewrite", "08_learned") else h["score"]
                lines.append(f"{qid} Q0 {cid} {rank} {101-rank} {system}\n")
                run_rows.append({"system": system, "query_id": qid, "doc_id": cid,
                                 "rank": rank, "score": 101-rank, "actual_score": score})
                if rank <= 10 and grades[qid][cid] < 2:
                    errors.append({"system": system, "query_id": qid, "rank": rank,
                                   "chunk_id": cid, "relevance": grades[qid][cid], "text": h["text"]})
        run_path = OUT / "runs" / f"{system}.trec"
        run_path.write_text("".join(lines), encoding="utf-8")
        run = list(ir.read_trec_run(str(run_path)))
        per_q = defaultdict(dict)
        aggregate = {}
        for measure in measures:
            judgments = qrels if measure == measures[1] else ranking_qrels
            for result in ir.iter_calc([measure], judgments, run):
                per_q[result.query_id][lookup[result.measure]] = result.value
            aggregate.update(ir.calc_aggregate([measure], judgments, run))
        for qid in qids:
            assert set(per_q[qid]) == set(METRICS)
            # Independent arithmetic verifies relevance threshold, cutoff,
            # linear nDCG gains and whole-pool ideal ranking.
            g = [grades[qid][h["chunk_id"]] for h in snapshots[qid]["runs"][system][:10]]
            nrel = sum(v >= 2 for v in recall_grades[qid].values())
            dcg = lambda a: sum(v / math.log2(i + 2) for i, v in enumerate(a))
            independent = [sum(v >= 2 for v in g) / 10,
                           sum(recall_grades[qid].get(h["chunk_id"], 0) >= 2 for h in snapshots[qid]["runs"][system][:100]) / nrel,
                           next((1 / (i + 1) for i, v in enumerate(g) if v >= 2), 0),
                           dcg(g) / dcg(sorted(grades[qid].values(), reverse=True)[:10])]
            for metric, expected in zip(METRICS, independent, strict=True):
                assert abs(per_q[qid][metric] - expected) < 1e-9
            rows.append({"system": system, "query_id": qid, **per_q[qid]})
        values[system] = np.array([[per_q[qid][k] for k in METRICS] for qid in qids])
        for i, measure in enumerate(measures):
            assert abs(aggregate[measure] - values[system][:, i].mean()) < 1e-9
    rng = np.random.default_rng(42)
    draws = rng.integers(0, len(qids), size=(10000, len(qids)))
    summaries = [{"system": system, "configuration": NAMES[system], "queries": len(qids),
                  **{m: float(v) for m, v in zip(METRICS, values[system].mean(axis=0), strict=True)}} for system in NAMES]
    deltas = []
    for new, base in PAIRS:
        d = values[new] - values[base]
        boot = d[draws].mean(axis=1)
        lo, hi = np.quantile(boot, [0.025, 0.975], axis=0)
        for i, metric in enumerate(METRICS):
            deltas.append({"system": new, "baseline": base, "metric": metric,
                           "delta": float(d[:, i].mean()), "ci95_low": float(lo[i]), "ci95_high": float(hi[i]),
                           "wins": int((d[:, i] > 1e-12).sum()),
                           "ties": int((np.abs(d[:, i]) <= 1e-12).sum()),
                           "losses": int((d[:, i] < -1e-12).sum())})
    groups = defaultdict(list)
    for i, q in enumerate(queries):
        groups[q["aspects"][0] if q["query_type"] == "single_aspect" else q["query_type"]].append(i)
    breakdown = [{"system": system, "group": group, "queries": len(idx),
                  **{m: float(v) for m, v in zip(METRICS, values[system][idx].mean(axis=0), strict=True)}}
                 for group, idx in groups.items() for system in NAMES]
    negatives = [{"system": system, "top10_relevant_pairs": int(round(values[system][:, 0].sum() * 10)),
                  "queries_with_no_relevant_top10": int((values[system][:, 2] == 0).sum())} for system in NAMES]
    write_csv(OUT / "summary.csv", summaries)
    write_csv(OUT / "per_query.csv", rows)
    write_csv(OUT / "paired_deltas.csv", deltas)
    write_csv(OUT / "by_query_group.csv", breakdown)
    write_csv(OUT / "replication.csv", replication)
    write_csv(OUT / "top10_counts.csv", negatives)
    for name, records in [("runs.jsonl", run_rows), ("top10_nonrelevant.jsonl", errors)]:
        (OUT / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    report = {
        "query_count": len(qids), "judged_pairs": len(qrels), "supplemental_top10_judgments": len(supplemental), "systems": len(NAMES),
        "top100_judged_fraction": sum(h["chunk_id"] in grades[qid] for qid in qids for system in NAMES for h in snapshots[qid]["runs"][system][:100]) / (100 * len(qids) * len(NAMES)), "top20_judged_fraction": sum(h["chunk_id"] in grades[qid] for qid in qids for system in NAMES for h in snapshots[qid]["runs"][system][:20]) / (20 * len(qids) * len(NAMES)), "top10_judged_fraction": 1.0,
        "top20_replicated_queries": sum(r["same_top20_as_dataset_run"] for r in replication),
        "total_system_queries": len(replication), "summary": summaries, "paired_deltas": deltas,
        "by_query_group": breakdown, "top10_counts": negatives,
        "pooled_recall_at100_theoretical_macro_ceiling": float(np.mean([
            min(100, n) / n for n in [sum(v >= 2 for v in grades[qid].values()) for qid in qids]])),
        "protocol": {"binary_relevant": ">=2", "cutoffs": {"P": 10, "Recall": 100, "MRR": 10, "nDCG": 10}, "ndcg_gain": "linear 0-3",
                     "aggregation": "macro average over 40 original queries", "bootstrap": "10000 paired query resamples; seed 42; percentile 95% intervals; exploratory, no multiple-comparison correction",
                     "rewrite": "cached outputs of original live DeepSeek calls; no new API calls",
                     "search_scope": "full original 459418-chunk indexes, product-filtered by system",
                     "comparisons": "parser single product in 01-06; extracted products in 07-08; global Top100",
                     "recall": "Original fixed qrels only; supplemental Top10 labels excluded from both recall numerator and denominator",
                     "ranking": "Original qrels plus directly judged new Top10 chunks, same merged qrels for all systems",
                     "learned_score_status": "Development/resubstitution: deployed model trained on original 40 queries; not held-out evaluation"},
        "provenance": {"code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                       "ir_measures_version": ir.__version__, "qrels_sha256": digest(DATA / "qrels.trec"),
                       "queries_sha256": digest(DATA / "queries.jsonl"), "rewrites_sha256": digest(DATA / "rewrites.jsonl"),
                       "runner_sha256": digest(ROOT / "scripts/run_retrieval_ablation.py"),
                       "model_sha256": digest(ROOT / "retrieval/models/review_ranker_v1.json"),
                       "supplemental_sha256": digest(OUT / "supplemental_labels.json"),
                       "statistics_script_sha256": digest(Path(__file__))},
    }
    (OUT / "statistics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"summary": summaries, "top20_replicated_queries": report["top20_replicated_queries"],
                      "top10_counts": negatives, "recall_ceiling": report["pooled_recall_at100_theoretical_macro_ceiling"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

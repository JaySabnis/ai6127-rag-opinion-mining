import json
from pathlib import Path

import ir_measures as ir


def main():
    root = Path(__file__).resolve().parent
    qrels = list(ir.read_trec_qrels(str(root / "qrels.trec")))
    known = {(q.query_id, q.doc_id) for q in qrels}
    qids = {q.query_id for q in qrels}
    measures = [ir.P(rel=2) @ 10, ir.Recall(rel=2) @ 100,
                ir.RR(rel=2) @ 10,
                ir.nDCG(dcg="log2", gains={0: 0, 1: 1, 2: 2, 3: 3}) @ 10]
    results = {}
    for path in sorted((root / "runs").glob("*.trec")):
        run = list(ir.read_trec_run(str(path)))
        if {r.query_id for r in run} != qids:
            raise ValueError(f"{path.name}: query coverage differs from qrels")
        for qid in qids:
            ranked = sorted((r for r in run if r.query_id == qid), key=lambda r: r.score, reverse=True)
            if len({r.doc_id for r in ranked}) != len(ranked):
                raise ValueError(f"{path.name}/{qid}: duplicate chunk IDs")
            if len(ranked) < 100:
                raise ValueError(f"{path.name}/{qid}: Recall@100 requires 100 results; existing Top20 runs cannot be used")
            missing = [r.doc_id for r in ranked[:10] if (qid, r.doc_id) not in known]
            if missing:
                raise ValueError(f"{path.name}/{qid}: label unjudged Top10 chunks first: {missing}")
        scores = ir.calc_aggregate(measures, qrels, run)
        results[path.stem] = {name: scores[m] for name, m in zip(
            ["P@10", "pooled_Recall@100", "MRR@10", "nDCG@10"], measures, strict=True)}
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
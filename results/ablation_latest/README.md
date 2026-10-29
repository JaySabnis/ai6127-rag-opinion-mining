# Full-Corpus Retrieval Ablation

- [Chinese HTML report](report_zh.html) / [English HTML report](report_en.html)
- [Summary](summary.csv), [per-query scores](per_query.csv), [paired comparisons](paired_deltas.csv), [subgroups](by_query_group.csv), [statistics](statistics.json)
- [Run settings](run_manifest.json) and [execution manifest](execution_manifest.json)

## Eight Configurations

1. BM25 only.
2. Dense only.
3. BM25/dense RRF.
4. Remove model names after product filtering.
5. Add retrieval aspect boosting.
6. Add rank fusion reranking.
7. Add query rewriting and product/aspect parsing.
8. Use the trained ranking model on configuration 7's retrieval candidates.

Configurations 1/2 are parallel baselines; 3–7 incrementally add components. Configurations 6–8 all reorder the first 20 candidates, preserve ranks 21–100 and use model-stripped semantic queries. Configurations 6/7 use `0.2/(10+retrieval_rank) + 0.8/(10+semantic_rank)`; configuration 8 uses the model trained on 100 separate questions.

## Canonical Data and Protocol

The shared index contains 459,418 chunks. Each saved run has 100 results per question and at most one chunk per source review. All eight systems and four metrics use the same [2,640 judgments](../../evaluation/retrieval_ablation_40_v2/qrels.trec); every Top10 is fully judged.

The [candidate corpus](../../evaluation/retrieval_ablation_40_v2/corpus.jsonl) stores 7,312 source chunks once. [inputs.jsonl](inputs.jsonl) stores query-dependent retrieval features, candidate IDs, scoring queries and frozen MiniLM logits. [runs/](runs/) is the only retained copy of the eight Top100 TREC outputs.

Metrics are P@10, pooled Recall@100, MRR@10 and nDCG@10. Grades >=2 are binary relevant; nDCG uses linear gains 0–3 and logarithmic discount. Recall counts known positives, not exhaustive corpus relevance. Configurations 5/6 and 7/8 share candidate membership, so their respective Recall@100 values match.

Configuration 8: P@10=0.8925, pooled Recall@100=0.7585, MRR@10=0.9675, nDCG@10=0.8568. Evaluation40 is excluded from this fit and selection but influenced earlier development, so it is not a historically untouched blind test.

## Reproduce from Repository Artifacts

```powershell
uv run python -X utf8 -m scripts.export_evaluation_dataset
uv run python -X utf8 -m scripts.train_ranker_100
uv run python -X utf8 -m scripts.run_retrieval_ablation
uv run --group evaluation python -X utf8 -m scripts.report_latest_ablation
```

The `evaluation` dependency group pins ir_measures 0.4.3 in `uv.lock`. These commands need neither the full source corpus nor full indexes. Training uses saved fixed candidates; ranking uses frozen semantic scores verified against the candidate order and scoring query. All eight rankings pass through the production ranking implementation. Reporting independently verifies the rank fusion formula, trained-model order and all 1,280 per-query metric values. Paired bootstrap uses 10,000 query resamples and seed 42.

To rerun MiniLM inference, add `--rescore` to the ablation command. To collect retrieval pools again from local shared indexes, add `--refresh-retrieval`; cached rewrites are still used. New unjudged Top10 results must be labeled before reporting.

The execution manifest hashes current data, model, outputs and source files. Git retains the corresponding source revision, so no duplicate source snapshots or model copies are stored here.

# Latest Full-Corpus Retrieval Ablation

This directory is the current entry point for the eight-system comparison. Configuration 8 uses model v2 trained on 100 dedicated questions.

## Reports

- [Chinese HTML report](report_zh.html)
- [English HTML report](report_en.html)

Reports include full-width tables, paired differences and intervals, subgroup results, training settings and interactive metric/group selectors. Browser printing is supported. An optional `results/ablation_latest.zip` can be produced locally; archives are excluded from version control.

## Configurations

1. BM25 only.
2. Dense only.
3. BM25/dense RRF fusion.
4. Remove the model name after product filtering.
5. Add retrieval aspect boosting.
6. Add legacy MiniLM/RRF/aspect reranking.
7. Add query rewriting, product and aspect extraction.
8. Reuse configuration 7's Top100 and replace the reranker with the training100 fusion model.

Configurations 1 and 2 are parallel baselines. Configurations 3-7 add components; configuration 8 replaces the reranking implementation. Legacy reranking scores all 100 candidates; the latest model reorders 20 and preserves the tail. The 8-versus-7 comparison therefore combines changes in window, semantic query handling and fusion.

## Common evaluation protocol

There are 40 fixed queries and 100 saved results per system-query run, drawn from the shared 459,418-chunk full index. Existing candidates and frozen MiniLM logits are reused. All metrics were recomputed without rerunning retrieval or query rewriting.

Every system and every metric uses this directory's `qrels.trec`: version 2's 2,626 judgments plus six later Top10 supplements, for 2,632 pairs. Metrics are Precision@10, pooled Recall@100, MRR@10 and nDCG@10, macro-averaged over queries. Grades >=2 are binary relevant; nDCG uses linear 0-3 gains.

Top10 judgments are complete. Ranks 11-100 need not all be judged; Recall credits known positive hits only. Reordering the same Top100 membership cannot change Recall@100. Supplements were not used in training the latest model.

## Latest model results

- Precision@10: **0.8925**.
- Pooled Recall@100: **0.7576**.
- MRR@10: **0.9675**.
- nDCG@10: **0.8573**.

The fusion layer uses 100 separate training queries and 2,000 labels; MiniLM remains frozen. Evaluation40 was excluded from this fit and selection but influenced earlier development, so a new blind test remains necessary for generalization claims.

## Files

- `summary.csv`: eight-system aggregate metrics.
- `per_query.csv`: per-query metrics.
- `paired_deltas.csv`: paired differences, bootstrap intervals and wins/ties/losses.
- `by_query_group.csv`: subgroup results.
- `statistics.json`: full results, training summaries, comparisons and protocol.
- `runs/`: eight Top100 TREC runs.
- `qrels.trec`, `queries.jsonl`: common judgments and fixed questions.
- `model.json`: v2 parameter snapshot.
- `source_snapshot/`: report source snapshot.
- `execution_manifest.json`: hashes and verification records.
- `cleanup_manifest.json`: removed reports, duplicate archives and diagnostics.

## Reproduce

From the repository root:

```powershell
uv run --with ir_measures python -X utf8 -m scripts.report_latest_ablation
```

This requires the saved `results/ablation_8/` outputs, `results/reranker_training_100/` results and corresponding model files. See the [training and evaluation guide](../../evaluation/README.md).

# Dedicated Reranker Training Set: 100 Queries

The set contains 100 English queries covering 16 products: 96 single-aspect queries (16 each for camera, battery, display, performance, service and carrier) and four camera-and-battery queries. Query text, target products, chunks, source reviews and exact text are disjoint from evaluation40.

## Fixed candidate collection

Search the full shared indexes with product filtering, model-name removal, BM25/dense RRF and retrieval aspect boosting. Retrieve 50 results at depth 150 per channel, keeping at most one chunk per source review. Freeze the first 20 candidates before judging. No candidates are replaced to balance the labels.

There are 2,000 query-chunk judgments, 1,797 unique chunks and 1,703 source reviews. Grade counts are 0: 225; 1: 209; 2: 592; 3: 974.

## Relevance grades

- 0: wrong product or no answer-bearing content.
- 1: vague, indirect or weak relevance.
- 2: clear but brief or partial support.
- 3: detailed direct experience, context or comparison.

Positive and negative opinions can both be relevant. Multi-aspect evidence covering only one aspect is capped at 2. Explicit product references take precedence over metadata. Battery health is not endurance, shipping speed is not performance, and bootloader unlocking is not carrier unlocking. Candidate files contain original text and relevance grades without rationale or evidence fields.

## Files

- `queries.jsonl`: the 100 fixed questions, product groups and aspect descriptions.
- `dataset_labeled.jsonl`: queries with the 20 candidates, original text and labels.
- `qrels.jsonl`, `qrels.trec`, `qrels.tsv`: equivalent judgment formats; qrels `doc_id` is the chunk ID.
- `corpus.jsonl`: source metadata for all judged chunks.
- `scored_candidates.jsonl`: original retrieval order, retrieval features, model-stripped queries and frozen MiniLM logits for training reproduction.
- `collection/`: per-query collection checkpoints.
- `direct_labels.json`: 20 grades per query in its original retrieval order.
- `manifest.json`: collection settings, counts, hashes and separation checks.

## Training and model

```powershell
uv run python -X utf8 -m scripts.finalize_training_100
uv run python -X utf8 -m scripts.train_ranker_100
```

Run from the repository root. MiniLM is frozen; only the Ridge fusion layer is fitted. Product-group cross-validation selects parameters using training100 only, then the frozen model is checked on evaluation40.

The default artifact is `retrieval/models/review_ranker_v2.json`. It uses seven features, relevance-grade targets and ridge penalty 100. Nested product-group validation gives nDCG@10=0.8635. Training fit is not a test score. Evaluation40 influenced earlier project development and is not an untouched final test set.

See the [reranker guide](../../retrieval/RERANKER_README.md), [training results](../../results/reranker_training_100/README.md) and [latest ablation report](../../results/ablation_latest/README.md). Delivery archives can be generated locally and are excluded from version control.

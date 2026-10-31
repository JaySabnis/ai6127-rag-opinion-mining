# Reranker Training100

The set contains **100 questions with 20 fixed retrieval candidates each**, totaling 2,000 judgments across 16 products. There are 96 single-aspect questions (16 each for camera, battery, display, performance, service and carrier) and four camera-and-battery questions. It includes 1,797 unique chunks and 1,703 source reviews; grade counts are 0=225, 1=209, 2=592, 3=974.

The judged training and evaluation sets have disjoint queries, target products, chunks, source reviews and exact text.

## Candidate Collection

Search the full shared indexes with product filtering, model-name removal, BM25/dense RRF and retrieval aspect boosting. Retrieve 50 candidates at depth 150 per channel, keeping at most one chunk per review, and freeze the first 20 before grading. Candidates are not replaced to balance the labels.

## Files

- [queries.jsonl](queries.jsonl): the 100 questions, product groups and aspects.
- [dataset_labeled.jsonl](dataset_labeled.jsonl): original candidate text, retrieval rank and relevance.
- [qrels.jsonl](qrels.jsonl) and [qrels.trec](qrels.trec): judgments keyed by question and chunk ID.
- [corpus.jsonl](corpus.jsonl): source metadata for judged chunks.
- [scored_candidates.jsonl](scored_candidates.jsonl): the single collection checkpoint, with model-stripped queries, original retrieval order, features and frozen MiniLM logits.
- [manifest.json](manifest.json): counts, source checks, dataset separation and hashes.

The [evaluation guide](../README.md) defines grades 0–3. Positive and negative opinions can both be relevant; one-aspect support for multi-aspect questions is capped at 2. Labels contain no rationale or evidence fields.

## Training and Validation

```powershell
uv run python -X utf8 -m scripts.collect_reranker_training_100
uv run python -X utf8 -m scripts.finalize_training_100
uv run python -X utf8 -m scripts.train_ranker_100
```

Collection resumes from `scored_candidates.jsonl`; completed candidate sets are reused. The finalizer verifies source text and dataset separation using local `data/corpus.jsonl` and rebuilds exports from ID-keyed qrels. It does not depend on positional label arrays or individual collection files.

MiniLM is frozen. Only the ridge fusion coefficients are trained. Product-group cross-validation selects parameters on training100; evaluation40 does not participate. The model uses seven features, grade targets and ridge penalty 100. Nested validation gives nDCG@10=0.8635; training fit is not a test score.

See the [reranker guide](../../retrieval/RERANKER_README.md), [training results](../../results/reranker_training_100/README.md) and [ablation report](../../results/ablation_latest/README.md).

# Retrieval Evaluation40: Version 2

Version 2 retains all 40 questions and 2,564 original judgments, adding 62 Top-10 supplements from the eight-system experiment. It contains **2,626 query-chunk judgments**, 2,084 unique chunks and 2,000 source reviews. Each query has 49-85 judged candidates and 21-59 binary positives. Grade counts are 0: 813; 1: 352; 2: 665; 3: 796.

## Evaluation protocol

All four metrics use the same `qrels.trec`: Precision@10, pooled Recall@100, MRR@10 and nDCG@10. Grades >=2 are binary relevant. nDCG uses linear gains 0, 1, 2, 3 and query-specific IDCG from all judgments.

Recall@100 divides known positives found in the first 100 results by all known positives for the query. Ranks 11-100 do not require exhaustive judging and unjudged items earn no known-positive credit. All Top-10 results must be judged.

The latest HTML report adds six later supplements and evaluates against **2,632 judgments**, using `results/ablation_latest/qrels.trec`. This directory remains the fixed 2,626-pair version. Use the appropriate qrels; do not copy Recall values across versions.

## Relevance rubric

- 0: wrong product or no content answering the question.
- 1: weak, vague or indirect relevance.
- 2: a clear partial answer or brief direct assessment.
- 3: specific direct experience, comparison, context or outcome.

Positive and negative opinions can both be relevant. Explicit product references take precedence over metadata; aspect tags are not relevance labels. Battery health is not endurance, shipping speed is not performance, and bootloader unlocking is not carrier unlocking. For multi-aspect questions, one-aspect support is capped at 2. Comparison evidence about either target product can be relevant.

## Files

- `queries.jsonl`: fixed queries and descriptive product/aspect targets; targets are not injected as extra retrieval inputs.
- `dataset_labeled.jsonl`: all judged candidates and original text, grouped by question.
- `qrels.trec`, `qrels.jsonl`, `qrels.tsv`: equivalent labels; qrels `doc_id` is chunk ID.
- `corpus.jsonl`: full metadata for the judged subset. Actual retrieval searches the full indexes.
- `runs/`, `runs.jsonl`: eight previously saved Top-100 runs, not newly collected runs.
- `pool_membership.jsonl`: candidate ranks in the saved systems and label version origins. Original candidates without current ranks remain included.
- `manifest.json`: counts, file hashes and validation records.

## Run evaluation

From the repository root:

```powershell
uv run --with ir_measures python -X utf8 evaluation/retrieval_ablation_40_v2/evaluate.py
```

The script requires 100 saved results per question and complete Top-10 judgments. New unjudged Top-10 results must be supplemented consistently before comparing systems.

The packaged eighth run is the earlier learned model from the original eight-system collection. The latest training100 model is evaluated in `results/ablation_latest/`; the packaged runs here remain frozen.

## Model and dataset separation

The current v2 model is trained only on 100 dedicated queries with 20 candidates each, totaling 2,000 training pairs. These 40 queries were excluded from the current fit and selection. Earlier model v1 did use them, and they influenced previous development, so they are not an untouched final test set. See [training100](../reranker_training_100/README.md).

Version 1 remains in `evaluation/retrieval_ablation_40/`. Local archives are optional delivery files and are not committed.

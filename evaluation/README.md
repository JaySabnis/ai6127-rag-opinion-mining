# Training and Evaluation

There are two maintained datasets:

- [Training100](reranker_training_100/README.md): 100 questions, 20 fixed retrieval candidates per question, 2,000 judgments.
- [Evaluation40](retrieval_ablation_40_v2/README.md): 40 questions, 2,640 judgments shared by all eight configurations and metrics.

The judged training and evaluation sets have no overlap in queries, target products, chunk IDs, source reviews or exact text. Parameter selection uses product-group validation on training100 only. Evaluation40 influenced earlier development and is not a historically untouched blind test.

## Relevance Grades

- 0: wrong product or no content answering the question.
- 1: vague, indirect or weak relevance.
- 2: a clear partial answer or brief direct assessment.
- 3: specific direct experience, context, comparison or outcome.

Positive and negative opinions can both be relevant. Explicit product references take precedence over metadata; aspect tags are not relevance labels. Support for only one part of a multi-aspect query is capped at 2. Comparison evidence about either target product can be relevant.

## Metrics

Precision@10, pooled Recall@100 and MRR@10 use grades >=2 as relevant. nDCG@10 uses linear gains 0–3 and `log2(rank+1)` discount. IDCG uses the best ten grades in the question's complete judged pool. Compute each metric per question and macro-average over the 40 questions.

Every system's Top10 is fully judged. Ranks 11–100 are not exhaustively labeled; Recall counts known positives found in the first 100 divided by all known positives for the question. It is pooled rather than exhaustive corpus recall. Qrels `doc_id` means chunk ID; `source_doc_id` in labeled data identifies the original review.

## Reproduce

From the repository root:

```powershell
uv run python -X utf8 -m scripts.export_evaluation_dataset
uv run python -X utf8 -m scripts.train_ranker_100
uv run python -X utf8 -m scripts.run_retrieval_ablation
uv run --group evaluation python -X utf8 -m scripts.report_latest_ablation
```

The report runner uses shared candidate text, saved retrieval inputs and frozen semantic logits. Add `--rescore` to repeat MiniLM inference or `--refresh-retrieval` to repeat retrieval with the shared local indexes. Neither mode calls the rewrite API; cached rewrites are retained. New unjudged Top10 results must be graded and exported before reporting.

Training collection is resumable through `scored_candidates.jsonl`; labels are keyed by query and chunk ID. `scripts.finalize_training_100` verifies the training set against the full local source corpus and refreshes exports. See the [ablation guide](../results/ablation_latest/README.md) and [reranker guide](../retrieval/RERANKER_README.md).

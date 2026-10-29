# Dedicated Training100: Model and Evaluation Results

Completed 100 new queries with 20 fixed retrieval candidates each, for 2,000 graded pairs. Training and evaluation40 have no overlap in query text, target products, chunks, source reviews or exact text.

## Training

MiniLM remains frozen. A Ridge fusion layer uses product-group five-fold selection over 32 feature/regularization/target combinations. All fitting and selection uses training100 only. Nested validation keeps each product entirely within a fold.

The selected seven features are raw MiniLM logit, semantic reciprocal rank, retrieval reciprocal rank, aspect coverage, log keyword count, keyword density and log text length. Features are standardized within the candidate window. Final scores are the weighted sum; ridge penalty is 100 and targets are query-centered relevance grades 0-3.

Nested product-group validation: P@10=0.8450, MRR@10=0.9783, nDCG@10=0.8635. Training fit is not a test score.

## Evaluation40 check

Reuse the same cached rewritten Top100 candidates and frozen MiniLM logits from the eight-system experiment; replace ranking weights only. Both learned models reorder the first 20 candidates. No retrieval or rewriting API rerun is needed.

The new model and retrieval-only control exposed six unjudged Top10 pairs. These were graded after freezing the model and saved in `evaluation_supplement.json`; they did not affect model fitting. All four metrics use `evaluation_qrels.trec`, containing 2,632 judgments: packaged v2 plus these six supplements.

- New model: P@10=0.8925, pooled Recall@100=0.7576, MRR@10=0.9675, nDCG@10=0.8573.
- Previous model: P@10=0.8875, pooled Recall@100=0.7576, MRR@10=1.0000, nDCG@10=0.8628.
- Retrieval-only control: P@10=0.8500, pooled Recall@100=0.7576, MRR@10=0.9438, nDCG@10=0.7801.

The previous model used evaluation40 in fitting; the new model did not. Its higher historical nDCG does not establish superior generalization. The paired nDCG difference is about -0.0055 and its bootstrap interval crosses zero. Evaluation40 influenced earlier project development and is not a historically untouched final test set.

## Artifacts and reproduction

The application defaults to `retrieval/models/review_ranker_v2.json`; v1 remains available through `--learned-model`. Current bilingual reports are in [ablation_latest](../ablation_latest/README.md). Historical raw metrics and candidates remain available; superseded display reports and duplicate archives were removed.

From the repository root:

```powershell
uv run python -X utf8 -m scripts.finalize_training_100
uv run python -X utf8 -m scripts.train_ranker_100
```

This directory contains the model, training statistics, nested out-of-fold results, evaluation40 metrics/runs, supplemental labels and execution manifest. The unpacked training dataset is at `evaluation/reranker_training_100/`; local ZIP delivery files are not committed.

Checks completed: 2,000 valid labels; source-data consistency; no overlap with evaluation40; production rankings matching saved weights on 100 queries; four systems and four metrics verified with ir_measures; 29 tests passed.

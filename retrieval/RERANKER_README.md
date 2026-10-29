# Reranker

The default strategy is `learned`, using [review_ranker_v2.json](models/review_ranker_v2.json). The model combines a frozen MiniLM cross-encoder with a Ridge linear fusion layer. Training updates only fusion coefficients; MiniLM is not fine-tuned.

## Ranking flow

Input candidates must be ordered by retrieval relevance. Product filtering precedes model-name removal from the query. MiniLM reads the processed query and original chunk text, without product/aspect prefixes added to the text.

1. Take the first 20 retrieval candidates and compute raw MiniLM logits.
2. Extract features and standardize them within the query's candidate window.
3. Calculate a learned linear score and sort descending; preserve original order for ties.
4. Keep candidates at positions 21 onward in retrieval order, then apply `top_k`.

With fewer than 20 candidates, all available candidates are scored. The trained window is 20; changing it requires validation. Reordering within the same Top100 set cannot change Recall@100.

## Seven selected features

- `semantic_raw`: frozen MiniLM relevance logit.
- `semantic_rank`: `1 / (10 + rank)` after sorting raw logits.
- `retrieval_rank`: `1 / (10 + rank)` in the original retrieval order.
- `aspect_coverage`: matched distinct query aspects divided by distinct query aspects.
- `keyword_count`: `log(1 + count)` for aspect-keyword matches in the text.
- `keyword_density`: keyword-match count divided by text word count.
- `length`: `log(1 + words)`.

Query aspects are detected from the model-stripped semantic query. If none are detected, coverage and keyword features are zero. Retrieval already incorporates RRF, BM25 and dense signals; the selected seven features do not directly consume their raw scores.

Within each candidate window:

```text
z_i = (x_i - mean(x_i)) / max(std(x_i), 1e-6)
score = sum(w_i * z_i)
```

The score is a query-relative ranking value, not a probability. `cross_encoder_score` remains bounded to 0-0.5 for display; the learned features use the raw logit.

## Training

[Training100](../evaluation/reranker_training_100/README.md) contains 100 queries with 20 fixed retrieval candidates each: 2,000 relevance judgments across 16 products. Query text, products, chunks, source reviews and exact text are disjoint from evaluation40.

Ridge regression minimizes:

```text
min_w ||Xw - y_centered||^2 + alpha * ||w||^2
```

Labels are centered within each query. Training optimizes squared error with regularization; nDCG@10 selects settings. Search includes four feature subsets, four penalties and two targets, for 32 settings. Five-fold product-group validation keeps each product entirely within one fold. Nested validation uses inner folds for selection and outer folds for assessment.

The final model uses seven features, grade targets and `alpha=100`, then fits all 100 training queries. Evaluation40 and later evaluation supplements do not participate in this fit or selection. Inference reads the model JSON, not qrels or labels associated with query/chunk IDs.

## Usage

Run commands from the repository root. Install project dependencies with `uv sync`. Initial MiniLM loading requires an existing model cache or a download connection.

```powershell
uv run python -m scripts.run_reranking --input tests/fixtures/rerank_candidates.json --query "Galaxy S10 battery life" --aspects battery --top-k 3 --output results/reranking/demo.json
```

Add `--local-files-only` when MiniLM is cached. The supported strategies are `learned` and `rank_fusion`. Rank fusion uses `0.2/(10+r) + 0.8/(10+m)` within the first 20 candidates, with no fitted coefficients. It is used by ablation configurations 6/7; configuration 8 uses the trained model. Both preserve the tail.

Application integration:

```python
from retrieval.rerank import CrossEncoderReranker, RerankerConfig

reranker = CrossEncoderReranker(RerankerConfig())
ranked = reranker.rerank(
    question,
    candidates,
    scoring_query=retrieval_result.search_text,
    query_aspects=retrieval_result.parsed.aspects,
    top_k=10,
)
```

Each candidate must contain `text`; supply `chunk_id`, `doc_id`, `model`, `brand` and `aspect_tags` when available. Preserve retrieval order. Pass the model-stripped query through `scoring_query`; if omitted, the parser uses candidate product metadata to process the query.

## Retrain and verify

```powershell
uv run python -X utf8 -m scripts.finalize_training_100
uv run python -X utf8 -m scripts.train_ranker_100
```

The first command validates and exports existing labels; the second fits the fusion layer on training100 only. Collection reuses cached candidates. Remap and verify labels if candidate order or identities change.

Nested product-group validation: P@10=0.8450, MRR@10=0.9783, nDCG@10=0.8635. Evaluation40 check: P@10=0.8925, pooled Recall@100=0.7585, MRR@10=0.9675, nDCG@10=0.8568.

Training fit, internal validation and evaluation40 are distinct protocols. Evaluation40 influenced earlier design and is not a historically untouched final test. See the [Chinese report](../results/ablation_latest/report_zh.html) and [English report](../results/ablation_latest/report_en.html).

## Implementation

- [rerank.py](rerank.py): loading, scoring and ranking API.
- [learned_rank.py](learned_rank.py): feature extraction and normalization.
- [train_ranker_100.py](../scripts/train_ranker_100.py): dedicated fitting and grouped validation.
- [Model parameters](models/review_ranker_v2.json): feature columns, weights, configuration and data hashes.
- [Training records](../results/reranker_training_100/README.md): fitting and evaluation results.

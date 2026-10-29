# Training and Retrieval Evaluation

This directory contains the dedicated reranker training data and fixed retrieval evaluation sets. Reproduction requires matching query, candidate, judgment and model versions.

## Current entry points

- [Dedicated training100](reranker_training_100/README.md): 100 queries, 20 fixed retrieval candidates per query, 2,000 judgments.
- [Evaluation40 version 2](retrieval_ablation_40_v2/README.md): 40 queries, 2,626 judgments and eight saved Top-100 runs.
- [Latest ablation](../results/ablation_latest/README.md): eight systems evaluated against 2,632 common judgments, with the training100 model in configuration 8.
- [Chinese HTML report](../results/ablation_latest/report_zh.html) / [English HTML report](../results/ablation_latest/report_en.html).

`retrieval_ablation_40/` retains the original dataset for historical reproduction. Use version 2 for its fixed packaged evaluation. Use the latest report directory's qrels to reproduce the latest report.

## Training data

The 100 English training queries cover 16 products: 96 single-aspect queries, with 16 each for camera, battery, display, performance, service and carrier, plus four camera-and-battery queries.

Candidates come from the full shared indexes using product filtering, model-name removal, hybrid RRF and aspect boosting. Retrieve 50 results with depth 150 per channel, keep one chunk per review, and freeze the first 20 before judging. Candidates are not replaced to obtain a chosen positive/negative ratio.

The training and evaluation40 sets have no overlap in query text, target products, chunk IDs, source review IDs or exact chunk text. Parameter selection uses product-group cross-validation on training100 only. A product cannot appear in both training and validation within the same fold.

## Relevance rubric

- **0:** wrong product or no content answering the question.
- **1:** weak, indirect or vague relevance with limited answer value.
- **2:** a clear partial answer or direct but brief assessment.
- **3:** specific and direct experience, context, comparison or outcome.

Both positive and negative opinions can be relevant. Explicit product references in the text take precedence over metadata. Do not substitute aspect tags for relevance judgments. Battery health is not battery endurance; shipping speed is not phone performance; bootloader unlocking is not carrier unlocking. For multi-aspect questions, support for only one aspect is capped at grade 2. For comparison questions, evidence about either target product can be relevant.

## Metrics and judgment versions

Metrics are Precision@10, pooled Recall@100, MRR@10 and nDCG@10, macro-averaged over queries.

- P, Recall and MRR treat grades >=2 as relevant.
- Recall@100 is the number of known positive chunks found in the first 100 results divided by all known positives for that query. It is pooled recall, not exhaustive corpus recall.
- nDCG uses linear gains 0, 1, 2, 3 and logarithmic discount `log2(rank + 1)`. IDCG uses the best judgments from the complete judged pool for that query.
- All Top-10 results must be judged. Ranks 11-100 do not require exhaustive labeling; unjudged items earn no known-positive retrieval credit.

The latest report uses version 2's 2,626 judgments plus six later Top-10 supplements, for 2,632 shared judgments across all four metrics and all eight systems. Supplements were not used in fitting or selecting the latest model. Recall values from different judgment versions must not be mixed.

In qrels, `doc_id` means **chunk_id**. In the source corpus, `doc_id` identifies the original review; labeled candidates expose it as `source_doc_id`.

## Commands

Run from the repository root.

Validate and export the completed training labels:

```powershell
uv run python -X utf8 -m scripts.finalize_training_100
```

Train the fusion layer and then evaluate the frozen model on evaluation40:

```powershell
uv run python -X utf8 -m scripts.train_ranker_100
```

Evaluate the eight packaged version-2 runs with version-2 qrels:

```powershell
uv run --with ir_measures python -X utf8 evaluation/retrieval_ablation_40_v2/evaluate.py
```

Recompute all eight systems with the latest common judgments and write both HTML reports:

```powershell
uv run --with ir_measures python -X utf8 -m scripts.report_latest_ablation
```

The report command reuses saved Top-100 runs; it does not repeat retrieval. Training candidate collection uses `scripts.collect_reranker_training_100`, which reuses checkpoints. If candidate identities or order change, remap and recheck labels before training.

## Interpreting results

MiniLM remains frozen; training fits only the feature fusion layer. Evaluation40 was excluded from this fit and parameter selection, but influenced earlier project development. It is not a historically untouched final test set. Report training fit, internal cross-validation and evaluation40 results separately.

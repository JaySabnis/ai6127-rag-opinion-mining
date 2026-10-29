# Full-Corpus Retrieval Evaluation: Original 40 Queries

This original pooled dataset contains 40 English queries, 2,564 query-chunk judgments, 2,035 unique chunks and 1,957 source reviews. Each question has 49-82 judged candidates. Grade counts are 0: 806; 1: 348; 2: 650; 3: 760.

There are 36 single-aspect questions, six each for camera, battery, display, performance, service and carrier, plus two multi-aspect and two product-comparison questions.

## Candidate pool

Candidates were obtained by searching the complete shared index of 459,418 chunks. The pool unions the original Top20 from BM25, dense, RRF, model-name removal, retrieval aspect boosting, legacy reranking and query rewriting. Chunk IDs are deduplicated, with at most one chunk per source review in each system's output. Candidate counts follow the actual union and no fixed positive/negative ratio is imposed.

## Relevance grades

- 0: wrong product or no answer-bearing content.
- 1: weak, vague or indirect relevance.
- 2: clear partial support or a direct brief assessment.
- 3: specific experience, context, comparison or outcome.

Positive and negative opinions can be relevant. Text explicitly about another product takes precedence over metadata. Battery health is not endurance, shipping speed is not performance, and bootloader unlocking is not carrier unlocking. Multi-aspect evidence covering only one aspect is capped at 2. Comparison evidence can describe either target product.

## Files and current usage

`queries.jsonl` stores the fixed questions; `dataset_labeled.jsonl` groups candidate text and grades by query. `qrels.trec`, `qrels.tsv` and `qrels.jsonl` contain equivalent judgments, where qrels `doc_id` is a chunk ID. `corpus.jsonl` is the judged source subset, not the retrieval search space. `pool_membership.jsonl` records pooling ranks. Rewrite outputs and collection configuration are retained for reproduction.

The original `runs/` contain Top20 results. They cannot directly measure Recall@100. Current metric definitions are P@10, pooled Recall@100, MRR@10 and nDCG@10; grades >=2 are binary relevant and nDCG uses linear 0-3 gains. `evaluate.py` rejects insufficient run depth or unjudged Top10 results.

For packaged Top100 evaluation, use [version 2](../retrieval_ablation_40_v2/README.md). For the latest common 2,632-judgment protocol and training100 model, use the [latest report](../../results/ablation_latest/README.md). This directory is retained for historical reproduction; the 100-query model training data are stored separately.

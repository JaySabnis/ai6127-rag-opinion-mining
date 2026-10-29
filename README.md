# AI6127 RAG-Based Aspect-Level Opinion Mining

A retrieval-augmented system for answering aspect-level questions about smartphone reviews, such as "How is the camera on the Galaxy S10?". The application retrieves review evidence; answer generation is not connected yet.

## Quick Start

```powershell
uv sync
uv run streamlit run app/streamlit_app.py
```

For query rewriting, configure `DEEPSEEK_API_KEY` in a local `.env` using [.env.example](.env.example). Sample and uploaded candidates can be used without the full corpus indexes. See the [application guide](app/README.md).

## Hybrid Retrieval

Full-corpus retrieval searches **459,418 chunks** with BM25 and BGE/FAISS, fuses ranks with RRF, and applies product filtering and retrieval aspect boosting. Product names are removed from search text after filtering.

- Download the [chunked corpus](https://www.kaggle.com/datasets/jaysabnis/ai6127-rag-opinion-mining-chunked-dataset) and place it at `data/corpus.jsonl`.
- Download [retrieval_indexes.zip from the shared Drive](https://drive.google.com/file/d/1Ep4F4cRi38eR8q2Ga4SLSGcBfng9OGq8/view?usp=drive_link) and extract it at the repository root to obtain `indexes/`. Reuse this index for consistent comparisons.
- Corpus, indexes, secrets and delivery archives remain local and are excluded from Git.

See the [retrieval guide](retrieval/README.md) for interfaces and parser behavior.

## Ranking and Evaluation

Two ranking methods are retained:

- `rank_fusion`: combine retrieval and MiniLM ranks with `0.2/(10+r) + 0.8/(10+m)`.
- `learned` (application default): frozen MiniLM plus a ridge-regularized linear fusion layer trained on 100 separate questions, each with 20 fixed candidates.

Both reorder the first 20 candidates and preserve the remaining retrieval order. MiniLM reads model-stripped queries and original chunk text. See the [reranker guide](retrieval/RERANKER_README.md) and [model parameters](retrieval/models/review_ranker_v2.json).

The evaluation set contains **40 questions and 2,640 judgments**. All eight configurations share one qrels: BM25, dense, RRF, model-name removal, aspect boosting, rank fusion reranking, query rewriting, and the trained ranking model. Metrics are **Precision@10, pooled Recall@100, MRR@10 and nDCG@10**.

- [Chinese HTML report](results/ablation_latest/report_zh.html) / [English HTML report](results/ablation_latest/report_en.html)
- [Aggregate metrics](results/ablation_latest/summary.csv), [per-query metrics](results/ablation_latest/per_query.csv), [paired comparisons](results/ablation_latest/paired_deltas.csv)
- [Reproduction guide](results/ablation_latest/README.md) and [results index](results/README.md)

The trained-model configuration scores P@10=0.8925, pooled Recall@100=0.7585, MRR@10=0.9675 and nDCG@10=0.8568. The judged training and evaluation sets have disjoint queries, target products, chunks, source reviews and exact text. Evaluation40 influenced earlier project development and is not a historically untouched blind test.

## Reproduce the Report

These commands use repository artifacts; no full corpus download, index rebuilding or rewrite API call is needed:

```powershell
uv run python -X utf8 -m scripts.export_evaluation_dataset
uv run python -X utf8 -m scripts.train_ranker_100
uv run python -X utf8 -m scripts.run_retrieval_ablation
uv run --group evaluation python -X utf8 -m scripts.report_latest_ablation
```

The runner joins shared candidate text with saved retrieval features and verified frozen MiniLM logits. Use `--rescore` to repeat MiniLM inference; use `--refresh-retrieval` to repeat retrieval against the local shared indexes. Cached query rewrites are reused in either case. Label any new Top10 results before reporting.

## Documentation and Data

- [Cleaning pipeline](cleaning_pipeline/README.md)
- [Chunking structure](chunking/README_CHUNKING_STRUCTURE.md) and [knowledge transfer](chunking/README_GROUP_B_KNOWLEDGE_TRANSFER.md)
- [Retrieval setup and interfaces](retrieval/README.md)
- [Reranker features, training and usage](retrieval/RERANKER_README.md)
- [Training and evaluation guide](evaluation/README.md)
- [Training dataset](evaluation/reranker_training_100/README.md): [queries](evaluation/reranker_training_100/queries.jsonl), [labeled candidates](evaluation/reranker_training_100/dataset_labeled.jsonl), [qrels](evaluation/reranker_training_100/qrels.trec)
- [Evaluation dataset](evaluation/retrieval_ablation_40_v2/README.md): [queries](evaluation/retrieval_ablation_40_v2/queries.jsonl), [labeled candidates](evaluation/retrieval_ablation_40_v2/dataset_labeled.jsonl), [qrels](evaluation/retrieval_ablation_40_v2/qrels.trec)
- [Training validation results](results/reranker_training_100/README.md)

## Development

```powershell
uv run python -m unittest discover -s tests -v
uv run python -m scripts.search_cli "Galaxy S10 camera" --k 10
```

Repository layout: `cleaning_pipeline/` and `chunking/` prepare the knowledge base; `retrieval/` and `app/` implement search and evidence inspection; `evaluation/` stores the two datasets; `results/` stores one report set and training validation results; `scripts/` contains the supported workflows. Review ratings are overall ratings, not aspect sentiment labels.

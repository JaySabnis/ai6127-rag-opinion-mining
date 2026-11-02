# Retrieval Evaluation40

The set contains **40 fixed questions and 2,640 judgments** over 2,097 unique chunks and 2,013 source reviews. Grades: 0=816, 1=353, 2=672, 3=799. This is the only maintained evaluation judgment set; all eight configurations use it.

## Files

- [queries.jsonl](queries.jsonl): fixed questions, descriptive target products and aspects. Targets are not injected as additional retrieval inputs.
- [dataset_labeled.jsonl](dataset_labeled.jsonl): each question's labeled candidates and original text.
- [qrels.jsonl](qrels.jsonl) and [qrels.trec](qrels.trec): equivalent ID-keyed labels; `doc_id` means chunk ID.
- [corpus.jsonl](corpus.jsonl): 7,312 unique source chunks needed by judged evidence and saved retrieval pools. Chunk text and metadata are stored once across configurations. This subset is for reproduction; original retrieval searches all 459,418 chunks.
- [rewrites.jsonl](rewrites.jsonl) and [rewrite_prompt.txt](rewrite_prompt.txt): cached rewrite inputs and prompt.
- [manifest.json](manifest.json): counts, label distribution, coverage checks and hashes.

The [ablation inputs](../../results/ablation_latest/inputs.jsonl) store candidate IDs, query-dependent retrieval features and frozen MiniLM logits. The [eight runs](../../results/ablation_latest/runs/) store Top100 results once; they are not duplicated inside this dataset.

## Labels and Metrics

The [evaluation guide](../README.md) defines grades 0–3 and metric formulas. All Top10 results are judged. P@10, pooled Recall@100 and MRR@10 use grades >=2; nDCG@10 uses linear gains 0–3. Recall counts known positives rather than exhaustive corpus relevance.

```powershell
uv run python -X utf8 -m scripts.export_evaluation_dataset
uv run python -X utf8 -m scripts.run_retrieval_ablation
uv run --group evaluation python -X utf8 -m scripts.report_latest_ablation
```

The exporter checks labels against source text and complete Top10 coverage. If retrieval or ranking changes, label missing Top10 pairs before comparing systems. Supplemental labels are not used in fitting or model selection.

Training100 has disjoint query text, target products, chunks, source reviews and exact text. Evaluation40 is excluded from the current fit and selection but influenced earlier project development, so it is not an untouched final test.

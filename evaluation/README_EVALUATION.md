# Evaluation --- Overview

This README lists every evaluation script, **what data each one waits
for**, and what it produces.

It covers the evaluation parts of Question 4 and Question 5:

-   manually labeled evaluation set (≥ 1,000 records, agreement ≥ 80%)
-   precision, recall and F1 of the classifier
-   answer-level RAG metrics (RAGAS)
-   latency and throughput
-   ablation tables

The sampling and the annotation tool have their own READMEs:
`README_EVAL_SAMPLING.md` and `README_ANNOTATION_TOOL.md`.

------------------------------------------------------------------------

## 1. Overview

| Step | Script                 | Waits for (input folder)          | Produces                                   |
|------|------------------------|-----------------------------------|--------------------------------------------|
| 1    | `sample_eval_set.py`   | --- (done)                        | `eval_sample.csv`                          |
| 2    | `annotation_tool.html` | --- (done)                        | annotators' CSVs                           |
| 3    | `merge_labels.py`      | `labels/` ← annotators' CSVs      | `results/agreement.json`, `eval.xls`       |
| 4    | `eval_classifier.py`   | `predictions/` ← generation side  | `results/classifier_<run>.json`            |
| 5    | `eval_rag.py`          | `rag_outputs/` ← full pipeline    | `results/rag_<run>.csv` / `.json`          |
| 6    | `summarize_latency.py` | `timings/` ← full pipeline        | `results/latency_<run>.json`               |
| 7    | `ablation_table.py`    | `results/` (from steps 4--6)      | `results/ablation.md`                      |

The input folders are **empty placeholders** until the data arrives.

All commands run from the repository root. After pulling, run `uv sync`
once to install the new dependencies (scikit-learn, ragas, xlwt, xlrd).

------------------------------------------------------------------------

## 2. `merge_labels.py` --- Agreement and `eval.xls`

### Input

``` text
evaluation/labels/labels_batch<b>_<name>.csv
```

The files downloaded from the annotation tool, unchanged.

### Run

``` bash
uv run python evaluation/merge_labels.py
```

### What it does

1.  Computes agreement for every pair of annotators on the same batch:
    -   `label`: all 5 labels
    -   `subjectivity`: opinionated vs neutral (unusable excluded)
    -   `polarity`: positive / negative / mixed (both said opinionated)

    Each as **percent agreement** and **Cohen's kappa**, plus an overall
    pooled percent and mean kappa. Saved to `results/agreement.json`.

2.  Writes every chunk the annotators disagree on to
    `labels/disagreements.csv`.

3.  Fill the **`final_label`** column there and run the script again.
    Decisions already filled in are kept on every rerun.

4.  When no disagreement is left open, it writes **`eval.xls`**:

| Column         | Values                                           |
|----------------|--------------------------------------------------|
| `chunk_id`     | Join key to `corpus.jsonl`                       |
| `text`         | Chunk text                                       |
| `label`        | `positive`, `negative`, `mixed`, `neutral`       |
| `subjectivity` | `opinionated` or `neutral`                       |
| `polarity`     | `positive`, `negative`, `mixed` (empty if neutral)|

`unusable` chunks are dropped. A chunk labeled by only one annotator
(partner not finished) keeps that label.

------------------------------------------------------------------------

## 3. `eval_classifier.py` --- Precision, Recall, F1

### Input (from the generation side)

``` text
evaluation/predictions/<run_name>.csv
```

``` text
chunk_id,label
ab810480589b_1,positive
d34b852b21ed_0,neutral
```

`label` must be one of `positive`, `negative`, `mixed`, `neutral`.
Every chunk in `eval.xls` needs a prediction.

### Run

``` bash
uv run python evaluation/eval_classifier.py evaluation/predictions/<run_name>.csv
```

### Output

Per-class precision, recall and F1 for the two stages required by the
assignment, plus overall:

``` text
Stage 1  subjectivity   opinionated vs neutral         (all chunks)
Stage 2  polarity       positive / negative / mixed    (gold-opinionated chunks)
Overall  4 labels
```

plus a confusion matrix. Saved to `results/classifier_<run_name>.json`.

------------------------------------------------------------------------

## 4. `eval_rag.py` --- RAGAS Metrics

### Input (from the full pipeline)

``` text
evaluation/rag_outputs/<run_name>.jsonl
```

One line per query:

``` json
{"question": "How is the camera on the Galaxy S10?",
 "contexts": ["retrieved chunk text 1", "retrieved chunk text 2"],
 "answer": "Most reviewers praise the camera ..."}
```

### LLM judge (to fill in)

``` bash
cp evaluation/llm_config.env.example evaluation/llm_config.env
```

Then fill in `llm_config.env`:

``` text
LLM_BASE_URL=        # OpenAI-compatible endpoint
LLM_API_KEY=
LLM_MODEL=
EMBEDDING_MODEL=     # optional; needed only for answer_relevancy
```

`llm_config.env` is git-ignored. **Never commit an API key.**

### Run

Try 2 queries first to check the setup and the cost:

``` bash
uv run python evaluation/eval_rag.py evaluation/rag_outputs/<run_name>.jsonl --limit 2
```

Then the full run:

``` bash
uv run python evaluation/eval_rag.py evaluation/rag_outputs/<run_name>.jsonl
```

### Metrics (RAGAS 0.4.3, `ragas.metrics.collections`)

| Metric              | Question it answers                                 | Needs           |
|---------------------|-----------------------------------------------------|-----------------|
| `faithfulness`      | Is every claim in the answer supported by contexts? | LLM             |
| `answer_relevancy`  | Does the answer address the question?               | LLM + embeddings|
| `context_precision` | Are the useful contexts ranked first?               | LLM             |
| `context_relevance` | Are the retrieved contexts relevant to the question?| LLM             |

`context_precision` uses the without-reference variant, so no
hand-written reference answers are needed.

### Output

``` text
results/rag_<run_name>.csv     # score per query and metric
results/rag_<run_name>.json    # mean per metric
```

Failed judge calls are printed, left empty, and counted in
`<metric>_failed`, instead of being averaged in.

------------------------------------------------------------------------

## 5. `summarize_latency.py` --- Latency and Throughput

### Input (from the full pipeline)

``` text
evaluation/timings/<run_name>.jsonl
```

One line per query, **in the order the queries were run**:

``` json
{"query": "...", "retrieval_s": 0.21, "classification_s": 0.35, "generation_s": 2.8, "n_chunks": 20}
```

Every field ending in `_s` is treated as a stage time in seconds. They can
be recorded with `time.perf_counter()` around each stage.

### Run

``` bash
uv run python evaluation/summarize_latency.py evaluation/timings/<run_name>.jsonl
```

### Output

Mean, median, p95 and max per stage and in total; cold (first query) vs
warm; queries per minute; chunks classified per second.
Saved to `results/latency_<run_name>.json`.

------------------------------------------------------------------------

## 6. `ablation_table.py` --- Ablation Tables (Q5)

Evaluate each system variant as its own run, named after what is
switched on:

``` text
baseline
rerank
rewrite
rerank+rewrite
faithfulness_filter
```

Then:

``` bash
uv run python evaluation/ablation_table.py
```

It collects every `results/classifier_*.json`, `rag_*.json` and
`latency_*.json` into three Markdown tables, saved to
`results/ablation.md`.

------------------------------------------------------------------------

## 7. Tested

All scripts were run end to end on fake data (fake annotator CSVs,
predictions, timings and RAG outputs):

-   agreement, disagreement file, adjudication reruns, `eval.xls`
-   classifier report and confusion matrix read back from `eval.xls`
-   latency summary
-   ablation tables
-   `eval_rag.py` against an unreachable endpoint (error handling)

**Not yet tested:** `eval_rag.py` against a real LLM endpoint. Run it with
`--limit 2` first once `llm_config.env` is filled in.

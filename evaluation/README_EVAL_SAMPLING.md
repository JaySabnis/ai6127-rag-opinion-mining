# Evaluation Set --- Sampling

This README documents how the **held-out evaluation set** was sampled
from the chunked corpus before manual annotation.

The assignment requires it twice:

-   Section 3.1: gather and label a held-out test set ourselves, saved
    as `eval.xls`
-   Question 4: manually label at least 1,000 records with an
    inter-annotator agreement of at least 80%

This step only **selects** the chunks. Labeling, agreement and the
final `eval.xls` come after.

------------------------------------------------------------------------

## 1. Files

``` text
evaluation/
├── sample_eval_set.py       # sampling script
├── eval_sample.csv          # 1,200 sampled chunks, label columns empty
└── README_EVAL_SAMPLING.md  # this file
```

Labeling is done with `annotation_tool.html`; see
`README_ANNOTATION_TOOL.md`.

------------------------------------------------------------------------

## 2. How to Run

Input is the Subgroup A corpus (`corpus.jsonl`, from the Kaggle
dataset).

From the repository root:

``` bash
uv run python evaluation/sample_eval_set.py ../data/corpus.jsonl
```

Optional output path:

``` bash
uv run python evaluation/sample_eval_set.py ../data/corpus.jsonl \
  --out evaluation/eval_sample.csv
```

Output:

``` text
1-star: 420
2-star: 120
3-star: 120
4-star: 120
5-star: 420
Wrote 1200 chunks to evaluation/eval_sample.csv
```

Runs in about 5 seconds.

------------------------------------------------------------------------

## 3. Sampling Unit: One Chunk

Each record is **one chunk** from `corpus.jsonl`, not a whole review
and not a single sentence.

### Why

At query time the retriever returns chunks, so the classifier also
reads chunks.

Labeling the same unit means the evaluation scores reflect what the
system actually does.

``` text
Retriever → chunks → classifier
                         ↑
             evaluated on chunks
```

Most reviews are one chunk anyway (about 1.16 chunks per review).

------------------------------------------------------------------------

## 4. Stratified by Star Rating

The corpus is heavily positive.

Chunk-level rating distribution (459,418 chunks):

| Rating | Chunks  | Share |
|--------|---------|-------|
| 1★     | 88,590  | 19.3% |
| 2★     | 32,796  | 7.1%  |
| 3★     | 38,660  | 8.4%  |
| 4★     | 67,736  | 14.7% |
| 5★     | 231,636 | 50.4% |

A plain random sample would be about half 5-star and mostly positive.

The assignment asks for a balanced evaluation set, so the sample uses
fixed quotas instead:

| Rating | Share of sample | Chunks |
|--------|-----------------|--------|
| 1★     | 35%             | 420    |
| 2★     | 10%             | 120    |
| 3★     | 10%             | 120    |
| 4★     | 10%             | 120    |
| 5★     | 35%             | 420    |
| Total  | 100%            | 1,200  |

This gives:

``` text
Negative-leaning (1–2★):  45%
Positive-leaning (4–5★):  45%
Mixed / neutral   (3★):   10%
```

### Rating is only used for sampling

As the Knowledge Base README warns:

``` text
rating != sentiment of the chunk
```

The rating only steers the sample toward balance. The actual labels come
from annotators reading the text.

------------------------------------------------------------------------

## 5. No Duplicates

The assignment requires the evaluation set to contain no duplicates.

Two rules are applied:

### At most one chunk per review

Chunks from the same review share a `doc_id` and overlap by one
sentence.

Taking two of them would put the same reviewer and the same sentence
into the set twice.

``` text
doc_id = 889fe7ab1fdc

889fe7ab1fdc_0   ← at most one of these
889fe7ab1fdc_1   ←
```

### No identical texts

Different reviews can have identical text (e.g. "Great phone").

Texts are compared after lowercasing and collapsing whitespace; only the
first is kept.

------------------------------------------------------------------------

## 6. Why 1,200 Instead of 1,000

Some chunks will be dropped during annotation:

-   unreadable or unrelated to the phone
-   annotators cannot agree

The extra 200 keep the final set at **1,000 or more** after these are
removed.

------------------------------------------------------------------------

## 7. Reproducibility

The random seed is fixed:

``` text
SEED = 6127
```

Running the script again on the same `corpus.jsonl` produces exactly the
same 1,200 chunks.

If the corpus is regenerated, the sample must be regenerated and
versioned too.

------------------------------------------------------------------------

## 8. Annotation Sheet Design

### Rating is hidden

`eval_sample.csv` does **not** contain the star rating.

An annotator who sees "1★" tends to label the chunk negative without
reading it carefully.

The rating can be joined back from `corpus.jsonl` on `chunk_id` when
needed.

### Order is shuffled

The rows are shuffled across ratings, so annotators do not see 420
one-star chunks in a row.

------------------------------------------------------------------------

## 9. `eval_sample.csv` Columns

| Column         | Description                                         |
|----------------|-----------------------------------------------------|
| `chunk_id`     | Chunk ID from `corpus.jsonl` (join key)             |
| `doc_id`       | Parent review ID                                    |
| `brand`        | Phone brand                                         |
| `model`        | Phone model (may be empty)                          |
| `aspect_tags`  | Detected aspects, `;`-separated (may be empty)      |
| `text`         | Chunk text to label                                 |
| `subjectivity` | **Empty** --- filled by annotators                  |
| `polarity`     | **Empty** --- filled by annotators                  |

The file is UTF-8 with BOM, so Excel opens it with emoji and
non-ASCII characters intact.

------------------------------------------------------------------------

## 10. Held-Out Rule

These 1,200 `chunk_id`s are the **test set**.

**Do not use them to train or tune the classifier.**

The generation side must exclude them from any training data:

``` python
import csv

held_out = {r["chunk_id"] for r in csv.DictReader(
    open("evaluation/eval_sample.csv", encoding="utf-8-sig"))}
```

------------------------------------------------------------------------

## 11. Known Limitations

### Neutral chunks may be scarce

Subjectivity detection needs neutral (no opinion) examples.

Reviews are mostly opinionated, and 3★ is only 10% of the sample.

After labeling, count the neutral chunks. If there are too few for a
meaningful F1, sample more.

### Very short chunks are kept

72 of the 1,200 chunks have fewer than 5 words (median: 40 words).

They are real reviews, so they were not filtered. If they turn out to
make the task too easy, add a minimum length and resample.

------------------------------------------------------------------------

## 12. Next Steps

``` text
eval_sample.csv
      |
      v
Labeling guideline
      |
      v
2–3 annotators label independently
      |
      v
Inter-annotator agreement (target ≥ 80%)
      |
      v
Resolve disagreements, drop unusable chunks
      |
      v
eval.xls (≥ 1,000 records)
```

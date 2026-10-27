# Evaluation Set --- Annotation Tool

This README explains how to label the **1,200 sampled chunks** in
`eval_sample.csv` (see `README_EVAL_SAMPLING.md`) using a single HTML
page.

The labels become the manually labeled evaluation set required by
Question 4 (at least 1,000 records, inter-annotator agreement of at least
80%) and the final `eval.xls`.

------------------------------------------------------------------------

## 1. Files

``` text
evaluation/
├── annotation_tool.html        # the tool: open it in a browser
└── build_annotation_tool.py    # regenerates the tool from eval_sample.csv
```

Annotators only need **`annotation_tool.html`**.

------------------------------------------------------------------------

## 2. For Annotators

### What you need

Only a web browser (Chrome, Edge, Firefox, Safari).

-   No installation
-   No Python
-   No internet connection needed

### Steps

``` text
1. Double-click annotation_tool.html
2. Enter your name and choose your batch
3. Read each chunk and press 1–5
4. Click "Download my labels (CSV)"
5. Send the CSV file to the coordinator
```

### Keys

| Key | Label                  | Meaning                                      |
|-----|------------------------|----------------------------------------------|
| `1` | Positive               | Only positive opinions                       |
| `2` | Negative               | Only negative opinions                       |
| `3` | Mixed                  | Both a clear positive and a clear negative   |
| `4` | Neutral (no opinion)   | Only facts, questions, wishes, flat words    |
| `5` | Unusable               | Not English, gibberish, not about a phone    |
| `←` | ---                    | Previous chunk                               |
| `→` | ---                    | Next chunk                                   |

After a number key, the next chunk opens automatically.

Go back with `←` to change a label at any time.

### Progress is saved automatically

Labels are stored in the browser after every key press.

You can close the page and continue later, as long as you use:

``` text
same browser + same computer + same name + same batch
```

Clearing the browser data deletes the saved progress, so download the CSV
once you finish.

------------------------------------------------------------------------

## 3. Labeling Rules

The same rules are shown on the right side of the tool.

### One label per chunk

Read the whole chunk and judge it as a whole.

### Facts + one opinion → the opinion wins

``` text
"Got it Monday. Works great."   → Positive
```

### Clear complaints or praise count, even without opinion words

``` text
"Died after two months."        → Negative
```

### A wish alone is not an opinion

``` text
"I wish it had more storage."   → Neutral
```

### Comparisons are about the reviewed phone

``` text
"Much better than my old LG."   → Positive
```

### Sarcasm: label what the reviewer means

``` text
"Great, it died in 2 days."     → Negative
```

### Judge the text only

The star rating is **hidden on purpose**, so it cannot bias the label.

### Work independently

Do not discuss labels with other annotators until everyone is done.
Otherwise the agreement score no longer measures independent judgments.

### Do not overthink

Pick your best guess and move on (about 15--30 seconds per chunk).

------------------------------------------------------------------------

## 4. Where the Rules Come From

The label definitions are adapted from the
[SemEval-2014 Task 4 ABSA annotation guidelines](https://alt.qcri.org/semeval2014/task4/data/uploads/semeval14_absa_annotationguidelines.pdf):

| Our label | SemEval-2014                                                 |
|-----------|--------------------------------------------------------------|
| Positive  | positive (Section 3.1)                                       |
| Negative  | negative (Section 3.1)                                       |
| Mixed     | conflict: "both positive and negative sentiment" (3.1)       |
| Neutral   | neutral: facts, wishes, "moderate" / "not an issue" (3.2)    |

"Facts + opinion → the opinion wins" follows Section 5: positive or
negative polarities dominate over neutral ones.

### One deliberate change

SemEval-2014 Section 3.2C labels sentiment that is only implied as
**neutral** (e.g. "It took them 25 minutes to bring our appetizer").

We label clear complaints or praise as **positive/negative** instead
(e.g. "Died after two months" → Negative), because phone reviews state
problems this way very often.

------------------------------------------------------------------------

## 5. Batches and Annotators

The 1,200 chunks are split into **3 batches of 400** in file order.

`eval_sample.csv` is already shuffled across star ratings, so every
batch has a similar rating mix.

Each batch is labeled by **2 annotators independently**:

``` text
Batch 1 (400 chunks)  →  annotator A + annotator B
Batch 2 (400 chunks)  →  annotator C + annotator D
Batch 3 (400 chunks)  →  annotator E + annotator F
```

This gives every chunk two independent labels, which is what
inter-annotator agreement (e.g. Cohen's kappa) is computed on.

------------------------------------------------------------------------

## 6. Output CSV

The downloaded file is named:

``` text
labels_batch<batch>_<name>.csv
```

Example:

``` text
labels_batch2_Jay.csv
```

Columns:

| Column         | Values                                                   |
|----------------|----------------------------------------------------------|
| `chunk_id`     | Join key to `eval_sample.csv` / `corpus.jsonl`           |
| `batch`        | `1`, `2` or `3`                                          |
| `annotator`    | Name entered in the tool                                 |
| `label`        | `positive`, `negative`, `mixed`, `neutral`, `unusable`   |
| `subjectivity` | `opinionated` or `neutral` (empty for `unusable`)        |
| `polarity`     | `positive`, `negative` or `mixed` (empty otherwise)      |

`subjectivity` and `polarity` are derived from `label`, matching the
two-stage task in the assignment:

``` text
label        subjectivity    polarity
-----------  -------------   --------
positive     opinionated     positive
negative     opinionated     negative
mixed        opinionated     mixed
neutral      neutral         (empty)
unusable     (empty)         (empty)
```

Unlabeled chunks are exported with empty label columns.

------------------------------------------------------------------------

## 7. Rebuilding the Tool

Only needed if `eval_sample.csv` changes.

From the repository root:

``` bash
uv run python evaluation/build_annotation_tool.py
```

Output:

``` text
Wrote 1200 chunks in 3 batches to evaluation/annotation_tool.html
```

The script embeds the chunks into the HTML, so the page works offline
without loading any other file.

**Do not rebuild after annotation has started.** A rebuilt tool may
assign different chunks to each batch.

------------------------------------------------------------------------

## 8. Next Steps

``` text
6 annotator CSVs
      |
      v
Merge on chunk_id
      |
      v
Agreement per batch (target ≥ 80%)
      |
      v
Resolve disagreements, drop "unusable"
      |
      v
eval.xls (≥ 1,000 records)
```

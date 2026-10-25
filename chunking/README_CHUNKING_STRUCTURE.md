# Chunking Module --- Project Structure

This README documents **only the chunking-related files and utilities**
in the current project.

The chunking work takes the cleaned review dataset and converts each
review into retrieval-ready chunks with the required metadata.

------------------------------------------------------------------------

## 1. Chunking-Related Structure

The current repository contains the following chunking-related files:

``` text
ai6127-rag-opinion-mining/
│
├── chunking/
│   ├── __init__.py
│   ├── chunk.py
│   └── metadata.py
│
└── scripts/
    ├── check_missing_models.py
    ├── find_missing_corpus_docs.py
    ├── ingest_weaviate.py
    ├── run_chunking.py
    └── validate_corpus.py
```

There are also Python cache files and `.gitkeep` files visible in the
directories, but they are not part of the chunking logic.

------------------------------------------------------------------------

# 2. `chunking/`

This directory contains the **core chunk-generation logic**.

``` text
chunking/
├── __init__.py
├── chunk.py
└── metadata.py
```

------------------------------------------------------------------------

## `chunking/__init__.py`

Package initializer for the `chunking` Python module.

It allows the chunking code to be imported as a Python package.

There is no major chunking logic in this file.

------------------------------------------------------------------------

## `chunking/chunk.py`

### Purpose

This is the main **review-to-chunk conversion module**.

It takes a review's text and converts it into one or more retrieval
chunks according to the project's chunking rules.

### Main responsibilities

-   Normalize whitespace
-   Handle short reviews
-   Handle long reviews
-   Split long reviews at sentence boundaries
-   Maintain the target chunk size
-   Maintain one-sentence overlap
-   Preserve the review title
-   Handle title-only reviews

### Chunking rules

#### Reviews ≤150 words

A review is kept as one chunk.

``` text
Review
   ↓
1 chunk
```

If the review has a title, the title is placed before the review text in
that chunk.

------------------------------------------------------------------------

#### Reviews \>150 words

Long reviews are split at sentence boundaries.

The target is approximately:

``` text
100–150 words
```

with:

``` text
one-sentence overlap
```

between consecutive chunks.

Conceptually:

``` text
Review
   |
   +---- Chunk 0
   |      S1 S2 S3 S4
   |
   +---- Chunk 1
   |      S4 S5 S6 S7
   |
   +---- Chunk 2
          S7 S8 S9
```

The repeated sentence provides contextual continuity between neighboring
chunks.

### Why sentence boundaries are used

The chunker does not arbitrarily cut a sentence in the middle.

This keeps opinions semantically coherent.

As a result, an unusually long individual sentence can produce a chunk
above 150 words. This is a known property of the implementation because
sentence integrity takes precedence over cutting sentences.

### Title handling

For a multi-chunk review:

``` text
Title → first chunk only
```

The title is not repeated in every chunk.

### Title-only reviews

Some source reviews contain a useful title but an empty body.

These are retained:

``` text
title
```

becomes the chunk text.

This prevents useful source information from being lost.

------------------------------------------------------------------------

# 3. `chunking/metadata.py`

### Purpose

This module generates and attaches metadata to each chunk.

The metadata makes the chunks useful not only for semantic retrieval but
also for filtering, ranking, grouping, evaluation, and aspect-level
analysis.

### Metadata produced

Each chunk contains:

``` text
chunk_id
doc_id
chunk_index
text
chunk_word_count
brand
model
rating
timestamp
verified
helpful_votes
is_renewed
aspect_tags
mentions_other_model
source_url
```

### Important concepts

#### `chunk_id`

Unique identifier for the individual chunk.

Example:

``` text
ce07282e5976_0
```

------------------------------------------------------------------------

#### `doc_id`

Identifier of the original review/document.

Multiple chunks from the same review share the same `doc_id`.

Example:

``` text
doc_id = ce07282e5976

chunk 0 → ce07282e5976_0
chunk 1 → ce07282e5976_1
chunk 2 → ce07282e5976_2
```

This allows downstream retrieval code to connect chunks back to their
parent review.

------------------------------------------------------------------------

#### `chunk_index`

Position of the chunk within its original review.

It starts from:

``` text
0
```

and increments for subsequent chunks.

------------------------------------------------------------------------

#### `text`

The actual retrieval text produced by `chunk.py`.

This is the main field that downstream embedding/retrieval code uses.

------------------------------------------------------------------------

#### `chunk_word_count`

Number of words in the final chunk.

This is stored so downstream code does not need to recalculate it.

------------------------------------------------------------------------

#### `brand`

Mapped smartphone brand.

Examples:

``` text
Samsung
Apple
Motorola
LG
Google
```

------------------------------------------------------------------------

#### `model`

Mapped smartphone model when available.

Missing model values are preserved when the source data does not provide
enough information to reliably identify the model.

The pipeline does not invent missing model information.

------------------------------------------------------------------------

#### `rating`

Original review rating:

``` text
1–5
```

------------------------------------------------------------------------

#### `timestamp`

Original review date.

------------------------------------------------------------------------

#### `verified`

Whether the review was a verified purchase.

------------------------------------------------------------------------

#### `helpful_votes`

Number of helpful votes associated with the review.

------------------------------------------------------------------------

#### `is_renewed`

Indicates whether the product is a renewed/refurbished device.

------------------------------------------------------------------------

#### `aspect_tags`

Detected project aspects:

``` text
camera
battery
display
performance
service
carrier
```

A chunk can contain multiple aspects.

Example:

``` json
"aspect_tags": ["camera", "battery"]
```

------------------------------------------------------------------------

#### `mentions_other_model`

Indicates whether another phone model is detected in the chunk.

This can be useful for identifying comparison statements.

------------------------------------------------------------------------

#### `source_url`

Source URL associated with the review/product.

------------------------------------------------------------------------

# 4. `scripts/`

The `scripts/` directory contains executable utilities around the
chunking pipeline.

``` text
scripts/
├── check_missing_models.py
├── find_missing_corpus_docs.py
├── ingest_weaviate.py
├── run_chunking.py
└── validate_corpus.py
```

------------------------------------------------------------------------

## `scripts/run_chunking.py`

### Purpose

This is the **main entry point for generating the corpus**.

It connects:

``` text
clean_reviews.jsonl
        ↓
chunking/chunk.py
        ↓
chunking/metadata.py
        ↓
corpus.jsonl
        +
corpus_statistics.json
```

### Typical command

``` bash
uv run python -m scripts.run_chunking \
  --input ../clean_reviews.jsonl \
  --output ../corpus.jsonl \
  --stats ../corpus_statistics.json
```

For a small test run:

``` bash
uv run python -m scripts.run_chunking \
  --input ../clean_reviews.jsonl \
  --output ../corpus.jsonl \
  --stats ../corpus_statistics.json \
  --max-records 1000
```

### Output

The important outputs are:

``` text
../corpus.jsonl
../corpus_statistics.json
```

------------------------------------------------------------------------

# 5. `scripts/validate_corpus.py`

### Purpose

This is the main **post-generation validation script**.

It checks whether the generated `corpus.jsonl` is structurally and
logically valid.

It checks:

-   JSONL validity
-   required fields
-   unique chunk IDs
-   chunk indexes
-   word counts
-   empty chunks
-   ratings
-   boolean fields
-   aspect tags
-   source URLs
-   duplicate text
-   corpus statistics

### Why it matters

The chunking pipeline should not be considered complete simply because
`corpus.jsonl` was generated.

The validation script verifies that the generated dataset can safely be
handed to the downstream retrieval team.

------------------------------------------------------------------------

# 6. `scripts/check_missing_models.py`

### Purpose

Checks the source review dataset for missing model information.

This was used to determine whether missing models were caused by the
chunking pipeline or were already missing in the source data.

The result showed that some source reviews genuinely do not have a
reliable model mapping.

Therefore, missing models are treated as a **source-data limitation**,
not something the chunker should guess.

------------------------------------------------------------------------

# 7. `scripts/find_missing_corpus_docs.py`

### Purpose

Compares the source review documents against the generated corpus.

It verifies that every source review is represented in the resulting
corpus.

This became especially important for title-only reviews.

Initially, reviews with empty bodies were being dropped.

The chunking logic was updated so that:

``` text
empty body + useful title
```

becomes:

``` text
one title-only chunk
```

After the change:

``` text
Source reviews:        395,038
Corpus documents:      395,038
Missing corpus docs:   0
```

------------------------------------------------------------------------

# 8. `scripts/ingest_weaviate.py`

### Original purpose

This script was created to push the generated chunks into Weaviate.

It imported the chunk text and metadata into a `ReviewChunk` collection.

### Current status

**Weaviate is no longer part of the planned project pipeline.**

The available Weaviate instance was not suitable for the full corpus
size, and the team decided not to use it because the required setup
would be paid/limited.

Therefore:

``` text
ingest_weaviate.py
```

is now a **legacy/unused ingestion script**.

The finalized corpus should instead be handed to the downstream group
as:

``` text
corpus.jsonl
```

through the shared Google Drive.

Do not make Weaviate a dependency of the current chunking pipeline.

------------------------------------------------------------------------

# 9. Chunking Pipeline Flow

The complete chunking-related flow is:

``` text
../clean_reviews.jsonl
          |
          v
   scripts/run_chunking.py
          |
          +-------------------+
          |                   |
          v                   v
 chunking/chunk.py     chunking/metadata.py
          |                   |
          +---------+---------+
                    |
                    v
             ../corpus.jsonl
                    |
                    v
          scripts/validate_corpus.py
                    |
                    v
              Final handoff
```

The additional verification scripts are used to confirm that:

``` text
source reviews
      ↕
generated documents
      ↕
chunk metadata
```

remain consistent.

------------------------------------------------------------------------

# 10. Most Important Files

If someone is new to the project, these are the files to understand
first:

### 1. `chunking/chunk.py`

**Core chunking algorithm.**

Understand this to know how reviews become chunks.

### 2. `chunking/metadata.py`

**Core metadata generation.**

Understand this to know what information accompanies each chunk.

### 3. `scripts/run_chunking.py`

**Pipeline entry point.**

Understand this to know how the complete corpus is generated.

### 4. `scripts/validate_corpus.py`

**Final quality check.**

Understand this to know how the generated corpus was verified.

### 5. `scripts/find_missing_corpus_docs.py`

**Source-to-corpus completeness check.**

Useful when modifying title/body handling or other chunking behavior.

### 6. `scripts/check_missing_models.py`

**Source metadata diagnostic.**

Useful for understanding missing model values.

------------------------------------------------------------------------

# 11. Final Handoff Files

The chunking work ultimately produces:

``` text
corpus.jsonl
corpus_statistics.json
```

These are the important data artifacts.

The downstream retrieval group does not need to reproduce the chunking
pipeline unless they want to modify the preprocessing.

Their primary input is:

``` text
corpus.jsonl
```

with each line representing one retrieval chunk and containing both the
text and the metadata described above.

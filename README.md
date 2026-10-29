# AI6127 RAG-Based Aspect-Level Opinion Mining

Group C generation: see [README_GROUP_C.md](README_GROUP_C.md) for
DeepSeek sentiment classification, sample ratios, grounded summaries, and
sentence-level evidence checking. This branch is based on `retrieve`; the
evaluation tools live on the `evaluation` branch.

An AI6127 project for building a Retrieval-Augmented Generation (RAG)
system for **aspect-level opinion mining on smartphone reviews**.

The target is to answer queries such as:

> **"How is the camera on the iPhone 15?"**

using retrieved review evidence and generating a grounded aspect-level
opinion.

------------------------------------------------------------------------

## Project Pipeline

``` text
Raw Smartphone Reviews
        |
        v
Cleaning & Filtering
        |
        v
Clean Review Dataset
        |
        v
Chunking + Metadata Enrichment
        |
        v
Validated corpus.jsonl
        |
        v
Embeddings + Retrieval
        |
        v
Reranking
        |
        v
Aspect-Level Opinion Analysis
        |
        v
Grounded LLM Response
```

The Knowledge Base / Chunking stage is responsible for producing the
validated `corpus.jsonl` that is handed to the downstream retrieval
team.

------------------------------------------------------------------------

# Repository Structure

``` text
ai6127-rag-opinion-mining/
│
├── app/
│   ├── __init__.py
│   ├── pipeline.py          full-corpus search used by the UI
│   ├── streamlit_app.py
│   └── README.md
│
├── chunking/
│   ├── __init__.py
│   ├── chunk.py
│   └── metadata.py
│
├── cleaning_pipeline/
│
├── evaluation/
│
├── generation/
│
├── retrieval/
│   ├── __init__.py
│   ├── config.py            index paths and defaults
│   ├── corpus.py            row-aligned metadata + filters
│   ├── text.py              tokenizer
│   ├── bm25.py              BM25 index
│   ├── dense.py             BGE encoder + FAISS index
│   ├── query_parser.py      model / aspect detection, LLM name resolution
│   ├── hybrid.py            RRF fusion + aspect boost
│   ├── retriever.py         Retriever.retrieve()
│   ├── query_rewrite.py     DeepSeek query rewrite
│   ├── rerank.py            MiniLM reranker
│   └── README.md
│
├── scripts/
│   ├── __init__.py
│   ├── build_retrieval_indexes.py
│   ├── check_missing_models.py
│   ├── eval_model_resolution.py
│   ├── find_missing_corpus_docs.py
│   ├── ingest_weaviate.py
│   ├── run_chunking.py
│   ├── run_reranking.py
│   ├── search_cli.py
│   ├── smoke_test_retrieval.py
│   ├── test_pipeline.py
│   └── validate_corpus.py
│
├── tests/
│   ├── fixtures/
│   │   ├── make_fixture.py
│   │   ├── mini_corpus.jsonl
│   │   └── rerank_candidates.json
│   ├── test_rerank_and_ui.py
│   └── test_retrieval.py
│
├── data/                    corpus.jsonl (git-ignored)
├── indexes/                 built retrieval indexes (git-ignored)
│
├── .streamlit/
│   └── config.toml
├── .env.example
├── .gitignore
├── .python-version
├── main.py
├── pyproject.toml
├── uv.lock
└── README.md
```

------------------------------------------------------------------------

# Environment Setup

This repository already contains:

-   `pyproject.toml` --- project dependencies/configuration
-   `uv.lock` --- locked dependency versions
-   `.python-version` --- project Python version

## If you have cloned the repository

**Do not run `uv init`.**

The project is already initialized.

From the repository root, simply run:

``` bash
uv sync
```

This creates/updates the project's `.venv` and installs the dependencies
defined by `pyproject.toml` and `uv.lock`.

After that, use `uv run` to execute project commands.

For example:

``` bash
uv run python -m scripts.run_chunking --help
```

You do not need to manually activate the virtual environment.

### Typical workflow

``` bash
git clone <repository-url>
cd ai6127-rag-opinion-mining

uv sync

uv run python -m scripts.run_chunking --help
```

If the environment is already set up, **just use `uv run ...`**. There
is no need to initialize the project again.

------------------------------------------------------------------------

# Knowledge Base / Chunking

The Knowledge Base stage converts cleaned smartphone reviews into
retrieval-ready chunks with metadata.

The main files are:

``` text
chunking/
├── chunk.py
└── metadata.py
```

and:

``` text
scripts/
├── run_chunking.py
├── validate_corpus.py
├── check_missing_models.py
└── find_missing_corpus_docs.py
```

------------------------------------------------------------------------

## `chunking/chunk.py`

Contains the core review-to-chunk logic.

### Reviews ≤150 words

A short review remains one chunk:

``` text
1 review → 1 chunk
```

The complete review is retained.

### Reviews \>150 words

Long reviews are split at sentence boundaries.

The target is approximately:

``` text
100–150 words per chunk
```

with:

``` text
one-sentence overlap
```

Example:

``` text
Chunk 0:
S1 S2 S3 S4

Chunk 1:
S4 S5 S6 S7

Chunk 2:
S7 S8 S9
```

The overlap helps preserve context across chunk boundaries.

### Sentence-boundary behavior

The implementation prioritizes complete sentence boundaries instead of
cutting sentences in the middle.

Therefore, a very long individual sentence can produce a chunk above 150
words.

This is intentional and should be treated as a documented property of
the chunking strategy.

### Title handling

For a normal review, the title is placed before the text in the first
chunk.

For a multi-chunk review:

``` text
Title → first chunk only
```

### Title-only reviews

Some source records have a useful review title but an empty body.

These are retained as a one-chunk review:

``` text
empty body + useful title
        ↓
title becomes chunk text
```

This prevents useful source information from being lost.

------------------------------------------------------------------------

# `chunking/metadata.py`

Generates the metadata attached to every chunk.

The chunk schema is:

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

### Important fields

  Field                    Purpose
  ------------------------ ------------------------------------------------
  `chunk_id`               Unique ID for the retrieval chunk
  `doc_id`                 ID of the original review/document
  `chunk_index`            Position of the chunk within its parent review
  `text`                   Actual retrieval text
  `chunk_word_count`       Word count of the chunk
  `brand`                  Smartphone brand
  `model`                  Smartphone model when reliably known
  `rating`                 Original 1--5 star review rating
  `timestamp`              Review date
  `verified`               Verified-purchase indicator
  `helpful_votes`          Helpful-vote count
  `is_renewed`             Whether the product is renewed/refurbished
  `aspect_tags`            Detected project aspects
  `mentions_other_model`   Whether another phone model is detected
  `source_url`             Source URL

Supported aspects:

``` text
camera
battery
display
performance
service
carrier
```

A chunk can have multiple aspects:

``` json
"aspect_tags": ["camera", "battery"]
```

------------------------------------------------------------------------

# `chunk_id` and `doc_id`

These two fields are important for downstream retrieval.

Example:

``` text
doc_id = ce07282e5976

chunk_id = ce07282e5976_0
chunk_id = ce07282e5976_1
chunk_id = ce07282e5976_2
```

`doc_id` identifies the original review.

`chunk_id` identifies the individual retrieval chunk.

This allows Group B to retrieve chunks independently while still being
able to group evidence from the same parent review.

------------------------------------------------------------------------

# `scripts/run_chunking.py`

This is the main entry point for generating the corpus.

Input:

``` text
../clean_reviews.jsonl
```

Output:

``` text
../corpus.jsonl
../corpus_statistics.json
```

## Full run

From the repository root:

``` bash
uv run python -m scripts.run_chunking \
  --input ../clean_reviews.jsonl \
  --output ../corpus.jsonl \
  --stats ../corpus_statistics.json
```

## Small test run

Before processing the full dataset, a smaller test can be run:

``` bash
uv run python -m scripts.run_chunking \
  --input ../clean_reviews.jsonl \
  --output ../corpus_test.jsonl \
  --stats ../corpus_statistics_test.json \
  --max-records 1000
```

This is useful for checking code changes before regenerating the full
corpus.

------------------------------------------------------------------------

# `scripts/validate_corpus.py`

Validates the generated `corpus.jsonl`.

It checks:

-   JSONL validity
-   Required fields
-   Unique chunk IDs
-   Chunk indexes
-   Word counts
-   Empty chunks
-   Ratings
-   Boolean fields
-   Aspect tags
-   Source URLs
-   Duplicate text
-   Corpus statistics

The final corpus passed the structural validation.

------------------------------------------------------------------------

# `scripts/check_missing_models.py`

Checks whether missing model values are already present in the source
dataset.

The pipeline does not invent model information when the source data does
not provide a reliable mapping.

Missing model values are therefore preserved.

------------------------------------------------------------------------

# `scripts/find_missing_corpus_docs.py`

Checks source-to-corpus completeness.

It verifies that every source review is represented in the generated
corpus.

After title-only review handling was implemented:

``` text
Source reviews:       395,038
Corpus documents:     395,038
Missing documents:    0
```

------------------------------------------------------------------------

# Final Corpus

The finalized Knowledge Base contains:

  Metric                           Value
  ------------------------- ------------
  Documents                      395,038
  Chunks                         459,418
  Words                       22,754,056
  Unique word types              103,954
  Average chunks/document         \~1.16

The main artifact is:

``` text
corpus.jsonl
```

Supporting statistics:

``` text
corpus_statistics.json
```

------------------------------------------------------------------------

# JSONL Format

`corpus.jsonl` uses JSON Lines.

**One line = one retrieval chunk.**

Example:

``` json
{
  "chunk_id": "ce07282e5976_0",
  "doc_id": "ce07282e5976",
  "chunk_index": 0,
  "text": "Excellent phone. The camera quality is very good...",
  "chunk_word_count": 42,
  "brand": "Samsung",
  "model": "Galaxy S10",
  "rating": 5,
  "timestamp": "2019-01-01",
  "verified": true,
  "helpful_votes": 12,
  "is_renewed": false,
  "aspect_tags": ["camera"],
  "mentions_other_model": false,
  "source_url": "..."
}
```

The downstream embedding pipeline should primarily embed:

``` text
text
```

while keeping the remaining metadata attached to the indexed chunk.
------------------------------------------------------------------------
## Dataset Access

The finalized chunked dataset is also available on Kaggle:

**[AI6127 RAG Opinion Mining — Chunked Dataset](https://www.kaggle.com/datasets/jaysabnis/ai6127-rag-opinion-mining-chunked-dataset)**

The Kaggle dataset contains the generated chunked review data and can be used as an alternative to the Google Drive handoff.

------------------------------------------------------------------------

# Hybrid Retrieval

BM25 (implemented on a sparse weight matrix) and dense retrieval
(`BAAI/bge-small-en-v1.5` + exact FAISS search) are fused with Reciprocal Rank
Fusion. Model filters are applied inside both rankers and cover every corpus
spelling of a phone; the DeepSeek rewrite's product names are checked against
the corpus catalogue (`resolve_model()`), with the rule-based parser as the
fallback. Put the corpus at `data/corpus.jsonl`, then:

```bash
uv sync
uv run python -m scripts.build_retrieval_indexes --stage all
uv run python -m scripts.search_cli "How is the camera on the Galaxy S10?" --mode all
uv run python -m scripts.test_pipeline        # rewrite -> retrieve -> rerank on the real indexes
```

On the real index a hybrid query takes ~30–70 ms, reranking 50 candidates
~100 ms (MPS) and the DeepSeek rewrite ~1.4 s. See
[retrieval/README.md](retrieval/README.md) for the interface, design and measurements.

# Query Rewrite, Reranker and UI

DeepSeek V4 Pro rewrites questions into English retrieval queries and extracts
products and aspect labels. MiniLM scores the question paired with each candidate's
original text, then combines its bounded relevance score `0.25*(1+tanh(s))` in [0, 0.5] (weight 0.6; contribution [0, 0.3]) with a normalized pure-RRF
prior (weight 1.0) and gated aspect coverage (weight 1.2). A sigmoid gate centered at -6
controls half the aspect reward; the other half is fixed for matched aspect coverage.

The Streamlit UI separates conversation on the left from evidence on the
right. Evidence appears after a successful query, and saved results can be
selected by question without rerunning the models.
Set `DEEPSEEK_API_KEY` in `.env`, then run:

```powershell
uv sync
uv run streamlit run app/streamlit_app.py
```

In the sidebar, **Full corpus** runs the whole pipeline over all 459,418 chunks
(rewrite → product check → hybrid retrieval → rerank; indexes must be built
first); **Sample data** and **Upload candidates** rerank a fixed pool. Answer
generation is not connected yet. See [retrieval/README.md](retrieval/README.md)
for interfaces and [app/README.md](app/README.md) for UI details.

# Retrieval Architecture

The finalized corpus is independent of the retrieval backend.

A possible downstream architecture is:

``` text
                 User Query
                     |
          +----------+----------+
          |                     |
          v                     v
     Dense Search           BM25 Search
          |                     |
          +----------+----------+
                     |
                     v
              Candidate Pool
                     |
                     v
              Hybrid Ranking
                     |
                     v
                 Reranker
                     |
                     v
              Top-k Evidence
                     |
                     v
          Aspect-Level Analysis
                     |
                     v
              LLM Generation
```

The retrieval team is responsible for selecting:

-   Embedding model
-   Vector/index backend
-   BM25 implementation
-   Hybrid retrieval strategy
-   Reranking approach

------------------------------------------------------------------------

# Why Dense + BM25?

Smartphone reviews contain many exact terms:

``` text
iPhone 15
Galaxy S10
5G
OLED
USB-C
Snapdragon
```

BM25/lexical retrieval is useful for exact product and technical terms.

Dense retrieval is useful for semantic matches.

For example:

``` text
Query:
"How long does the phone last on a charge?"

Review:
"The battery easily gets me through the entire day."
```

The wording differs, but the meaning is related.

A hybrid approach can combine both signals.

------------------------------------------------------------------------

# Aspect-Level Opinion Mining

The system is designed to retrieve opinions about specific aspects.

Example:

``` text
Query:
"How is the camera on the Galaxy S10?"
```

The target information is:

``` text
Product → Galaxy S10
Aspect  → camera
Opinion → positive / negative / mixed
Evidence → supporting review snippets
```

The `aspect_tags` metadata can be used as an additional
retrieval/filtering signal.

------------------------------------------------------------------------

# Important: Rating Is Not Aspect Sentiment

The review rating represents the overall review.

It should not automatically be treated as the sentiment of every aspect.

Example:

``` text
Rating: 5

"The camera is excellent, but the battery is terrible."
```

This means:

``` text
Overall → positive
Camera  → positive
Battery → negative
```

Aspect-level sentiment should therefore be inferred from the review
text.

------------------------------------------------------------------------

# Weaviate Status

We originally created:

``` text
scripts/ingest_weaviate.py
```

for Weaviate ingestion.

A small import test worked, but the available Weaviate setup had an
object limit that was insufficient for the full corpus.

The project therefore **does not use Weaviate in the final
architecture**.

The corpus is now distributed through the shared Google Drive.

The final flow is:

``` text
Google Drive
     |
     v
corpus.jsonl
     |
     v
Group B
     |
     +--> Embeddings
     +--> BM25
     +--> Vector/index backend
     |
     v
Retrieval
```

The Knowledge Base does not depend on a particular vector database.

------------------------------------------------------------------------

# Data Handoff

Recommended shared folder:

``` text
AI6127 RAG Opinion Mining/
└── Knowledge Base/
    ├── corpus.jsonl
    ├── corpus_statistics.json
    ├── README_PROJECT_STRUCTURE.md
    └── README_GROUP_B_KNOWLEDGE_TRANSFER.md
```

`corpus.jsonl` is the primary handoff file.

`corpus_statistics.json` contains supporting statistics.

The two README files document the implementation and the handoff.

------------------------------------------------------------------------

# Validation Summary

The finalized corpus was validated with:

``` text
Documents:                 395,038
Chunks:                    459,418
Unique chunk IDs:          459,418
Duplicate chunk IDs:       0
Malformed JSON lines:      0
Word-count mismatches:     0
Empty chunks:              0
Missing required fields:   0
Invalid ratings:           0
Invalid boolean fields:    0
```

Expected warnings include:

-   Chunks below 100 words because short reviews remain intact.
-   Chunks above 150 words when a very long sentence cannot be split
    without breaking sentence boundaries.
-   Repeated text caused by legitimate overlap or repeated review
    content.

------------------------------------------------------------------------

# Project Status

## Knowledge Base / Chunking

-   [x] Smartphone review filtering
-   [x] Review cleaning
-   [x] Duplicate handling
-   [x] Brand/model metadata
-   [x] Sentence-aware chunking
-   [x] One-sentence overlap
-   [x] Title preservation
-   [x] Title-only review handling
-   [x] Aspect tagging
-   [x] Chunk metadata
-   [x] Corpus generation
-   [x] Corpus validation
-   [x] Source-to-corpus completeness verification
-   [x] Model metadata preservation verification
-   [x] Corpus statistics
-   [x] Weaviate removed from final architecture
-   [x] Google Drive handoff

## Retrieval / Generation

-   [x] Embedding pipeline (bge-small, resumable sharded encoding)
-   [x] Vector/index backend (local FAISS, exact inner-product search)
-   [x] BM25 retrieval
-   [x] Hybrid retrieval (RRF, model filters, aspect boost enabled by default)
-   [x] DeepSeek query rewrite with product/aspect extraction
-   [x] LLM product names checked against the corpus catalogue
-   [x] Cross-encoder reranking module and standalone CLI
-   [x] End-to-end check: rewrite -> retrieve -> rerank (`scripts/test_pipeline.py`)
-   [x] UI wired to full-corpus retrieval (`Full corpus` source)
-   [x] Streamlit chat UI with separate evidence panel
-   [ ] Aspect-level opinion classification
-   [ ] Final RAG generation
-   [ ] End-to-end evaluation

------------------------------------------------------------------------

# Quick Commands

## Install/sync the existing project

**Do not run `uv init`.**

The repository is already initialized.

``` bash
uv sync
```

## Run the chunking pipeline

``` bash
uv run python -m scripts.run_chunking \
  --input ../clean_reviews.jsonl \
  --output ../corpus.jsonl \
  --stats ../corpus_statistics.json
```

## Run a 1,000-review test

``` bash
uv run python -m scripts.run_chunking \
  --input ../clean_reviews.jsonl \
  --output ../corpus_test.jsonl \
  --stats ../corpus_statistics_test.json \
  --max-records 1000
```

## Validate the corpus

``` bash
uv run python scripts/validate_corpus.py \
  --input ../corpus.jsonl
```

------------------------------------------------------------------------

# Responsibility Boundary

``` text
SUBGROUP A
    |
    +-- Cleaning
    +-- Chunking
    +-- Metadata
    +-- Validation
    |
    v
corpus.jsonl
    |
    | HANDOFF
    v
GROUP B
    |
    +-- Embeddings
    +-- Vector / lexical indexing
    +-- BM25
    +-- Hybrid retrieval
    +-- Reranking
    |
    v
Aspect-Level Opinion Analysis
    |
    v
Generation
```

The core handoff principle is:

> **Subgroup A produces validated, sentence-aware, metadata-rich
> retrieval chunks. Group B turns those chunks into the retrieval system
> and downstream RAG pipeline.**

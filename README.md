# RAG-Based Aspect-Level Opinion Mining for Smartphone Reviews

An AI6127 group project that develops a Retrieval-Augmented Generation
(RAG) system for aspect-level opinion mining over smartphone reviews.

The system allows users to ask questions about a specific smartphone and
aspect, such as:

-   How is the battery life on the Samsung Galaxy S24?
-   What do people think of the iPhone 15 camera?
-   Is the Pixel 8 good for gaming?
-   What are common complaints about the iPhone 15 Pro Max?

The system retrieves relevant review passages, analyzes their sentiment,
aggregates opinions, and generates a concise evidence-grounded summary.

------------------------------------------------------------------------

## Project Overview

The project combines information retrieval, aspect-level sentiment
analysis, and RAG to answer questions about consumer opinions on
smartphones.

For a user query, the system aims to:

1.  Identify the relevant smartphone and aspect.
2.  Retrieve relevant review passages.
3.  Combine sparse and dense retrieval.
4.  Fuse and re-rank retrieved results.
5.  Detect whether retrieved text is opinionated.
6.  Classify opinion polarity as positive or negative.
7.  Aggregate sentiment for the queried aspect.
8.  Generate a short summary grounded in retrieved evidence.
9.  Present supporting review snippets, metadata, and source
    information.

------------------------------------------------------------------------

## System Architecture

``` text
                         User Query
                             |
                             v
                    Query Processing
                             |
                 +-----------+-----------+
                 |                       |
                 v                       v
             BM25 Search          Dense Retrieval
                                     |
                              Sentence-Transformers
                                     |
                 +-----------+-----------+
                             |
                             v
                    Hybrid Fusion (RRF)
                             |
                             v
                    Cross-Encoder
                      Re-ranking
                             |
                             v
                    Metadata Filtering
                             |
                             v
                 Retrieved Review Chunks
                             |
                             v
                 Subjectivity Detection
                             |
                             v
                    Polarity Detection
                             |
                             v
                 Sentiment Aggregation
                             |
                             v
                      LLM Generation
                             |
                             v
                    Grounded Summary
                             |
                             v
                      Streamlit UI
```

------------------------------------------------------------------------

## Data Pipeline

The project uses a large smartphone review corpus that is cleaned and
transformed into retrieval-ready passages.

``` text
Source Dataset
      |
      v
Cleaning Pipeline
      |
      v
Cleaned Review Dataset
      |
      v
Chunking + Metadata
      |
      v
Embeddings
      |
      v
Weaviate
      |
      v
Retrieval Pipeline
```

The large dataset is **not stored in this Git repository**.

The cleaned/chunked data, embeddings, and associated metadata are stored
externally, with **Weaviate** used for vector storage and retrieval.

------------------------------------------------------------------------

## Data Sources

The primary corpus is based on the Amazon Customer Reviews / Amazon
Reviews 2023 dataset from the Hugging Face Hub, filtered to smartphones.

A supplementary set of public phone-review pages may be collected to
increase source diversity.

A separate held-out evaluation set of approximately 1,000+ reviews is
maintained for evaluation and is kept separate from the
training/indexing corpus.

------------------------------------------------------------------------

## Data Processing

The cleaning pipeline performs:

-   Product/phone extraction
-   De-duplication
-   English-language filtering
-   Removal of boilerplate and spam reviews
-   Basic text normalization
-   Review-specific microtext normalization

Reviews are naturally short passages. Reviews above approximately 150
words are split into smaller sub-passages using sentence boundaries.

Each chunk can contain metadata such as:

``` text
chunk_id
text
product
aspect
rating
timestamp
source_url
```

------------------------------------------------------------------------

## Retrieval Pipeline

### Sparse Retrieval

BM25 is used for lexical matching between the query and review passages.

### Dense Retrieval

Sentence-Transformer models are used to generate embeddings for semantic
similarity search.

The embeddings and associated chunk information are stored in Weaviate.

### Hybrid Retrieval

Sparse and dense retrieval results are combined using Reciprocal Rank
Fusion (RRF).

### Re-ranking

A Sentence-Transformers Cross-Encoder is used to re-score the top
candidates after hybrid retrieval.

### Metadata Filtering

Metadata can be used to restrict retrieval by:

-   Smartphone model
-   Aspect
-   Date
-   Other available review metadata

------------------------------------------------------------------------

## Opinion Mining and Generation

The downstream pipeline follows a two-stage classification approach.

### 1. Subjectivity Detection

Determine whether a retrieved passage expresses an opinion or is
primarily factual/neutral.

### 2. Polarity Detection

For opinionated passages, classify the sentiment as:

-   Positive
-   Negative

The polarity results are aggregated to produce an aspect-level sentiment
split.

Example:

``` text
iPhone 15 — Camera

Positive: 78%
Negative: 22%

Reviews analyzed: 143
```

The retrieved and polarity-labelled passages are then provided to an LLM
to generate a concise natural-language summary.

The generator is instructed to remain grounded in retrieved evidence and
avoid unsupported claims.

------------------------------------------------------------------------

## Repository Structure

``` text
rag-aspect-opinion-mining/
│
├── README.md
├── pyproject.toml
├── uv.lock
├── .gitignore
├── .env.example
│
├── cleaning_pipeline/
│   ├── __init__.py
│   ├── 01_extract_phones.py
│   ├── 02_clean.py
│   ├── common.py
│   ├── microtext.py
│   └── README.md
│
├── chunking/
│   ├── __init__.py
│   ├── chunk.py
│   └── metadata.py
│
├── retrieval/
│   ├── __init__.py
│   ├── embeddings.py
│   ├── weaviate_client.py
│   ├── bm25.py
│   ├── hybrid_search.py
│   └── reranker.py
│
├── generation/
│   ├── __init__.py
│   ├── classifier.py
│   ├── llm.py
│   └── prompts.py
│
├── evaluation/
│   ├── __init__.py
│   ├── retrieval.py
│   ├── classification.py
│   ├── ragas.py
│   └── ablation.py
│
├── app/
│   ├── __init__.py
│   └── streamlit_app.py
│
├── tests/
│   ├── test_cleaning.py
│   ├── test_chunking.py
│   ├── test_retrieval.py
│   └── test_generation.py
│
└── scripts/
    ├── run_cleaning.py
    ├── run_chunking.py
    ├── ingest_weaviate.py
    └── run_evaluation.py
```

### Directory Responsibilities

#### `cleaning_pipeline/`

Contains the existing data extraction and cleaning pipeline.

Responsibilities:

-   Extract smartphone-related reviews
-   Clean review text
-   Remove duplicates and unwanted content
-   Normalize review-specific microtext
-   Prepare the cleaned corpus for chunking

#### `chunking/`

Converts cleaned reviews into retrieval-ready passages and attaches
metadata.

#### `retrieval/`

Contains:

-   Embedding generation
-   Weaviate connection
-   BM25 retrieval
-   Dense retrieval
-   Hybrid retrieval
-   Reciprocal Rank Fusion
-   Cross-encoder re-ranking
-   Metadata filtering

#### `generation/`

Contains:

-   Subjectivity classification
-   Polarity/sentiment classification
-   LLM integration
-   Prompt templates
-   Grounding and citation logic

#### `evaluation/`

Contains evaluation code for:

-   Retrieval metrics
-   Classification metrics
-   RAGAS evaluation
-   Ablation experiments

#### `app/`

Contains the Streamlit application used for the project demonstration.

The UI is intended to provide:

-   Query input
-   Product/aspect filters
-   Retrieved review snippets
-   Rating and source information
-   Sentiment distribution
-   Generated summary
-   Query latency

#### `tests/`

Contains unit and integration tests for project components.

#### `scripts/`

Contains executable entry points for running major pipeline stages.

------------------------------------------------------------------------

## Environment and Dependency Management

The project uses **uv** for Python environment and dependency
management.

The virtual environment is local to each developer and is **not
committed to Git**.

### Create the environment

``` bash
uv venv
```

### Activate the environment

macOS / Linux:

``` bash
source .venv/bin/activate
```

Windows:

``` bash
.venv\Scripts\activate
```

### Install dependencies

``` bash
uv sync
```

Dependencies are defined in `pyproject.toml`, while `uv.lock` records
locked versions for reproducible team environments.

------------------------------------------------------------------------

## Environment Variables

Create a local `.env` file based on `.env.example`.

Example:

``` text
WEAVIATE_URL=
WEAVIATE_API_KEY=

# Add only if required by the selected models/APIs
OPENAI_API_KEY=
HF_TOKEN=
```

Do not commit `.env` or API keys to the repository.

------------------------------------------------------------------------

## Running the Pipeline

The exact commands may evolve as implementation progresses.

### Cleaning

``` bash
python cleaning_pipeline/01_extract_phones.py
python cleaning_pipeline/02_clean.py
```

### Chunking

``` bash
python chunking/chunk.py
```

### Ingest into Weaviate

``` bash
python scripts/ingest_weaviate.py
```

### Retrieval

``` bash
python retrieval/hybrid_search.py
```

### Run the Streamlit application

``` bash
streamlit run app/streamlit_app.py
```

------------------------------------------------------------------------

## Evaluation

### Retrieval Metrics

The retrieval pipeline will be evaluated using:

-   Precision@K
-   Recall@K
-   MRR
-   NDCG

### Classification Metrics

The subjectivity and polarity classifiers will be evaluated using:

-   Precision
-   Recall
-   F1

### RAG Evaluation

RAG-specific evaluation will include:

-   Faithfulness
-   Answer relevance
-   Context precision/relevance

### System Evaluation

Additional measurements include:

-   End-to-end query latency
-   Reviews processed per second
-   Indexing cost
-   Scalability

------------------------------------------------------------------------

## Planned Experiments

### Retrieval Ablation

``` text
BM25
   vs
Dense Retrieval
   vs
BM25 + Dense Retrieval (RRF)
   vs
BM25 + Dense + Cross-Encoder Re-ranking
```

### Aspect-Aware Retrieval

Compare retrieval with and without aspect/product metadata filtering.

### Query Expansion

Evaluate query rewriting or expansion for short and ambiguous queries.

### Generation and RAG Ablation

Evaluate the contribution of:

-   Re-ranking
-   Query rewriting
-   Faithfulness checking
-   Aspect-level sentiment vs whole-review sentiment

------------------------------------------------------------------------

## Technology Stack

  Component              Technology
  ---------------------- -------------------------------------
  Language               Python
  Environment            uv
  Data Processing        Pandas, Hugging Face Datasets
  Web/Data Extraction    Trafilatura
  Chunking               LangChain Text Splitter
  Sparse Retrieval       BM25
  Embeddings             Sentence-Transformers
  Vector Database        Weaviate
  Re-ranking             Sentence-Transformers Cross-Encoder
  Classification         Hugging Face Transformers
  Generation             Ollama / Hosted API
  UI                     Streamlit
  Retrieval Evaluation   ir_measures
  RAG Evaluation         RAGAS
  Testing                Pytest

------------------------------------------------------------------------

## Data Storage Policy

Large datasets and generated vector data are intentionally excluded from
Git.

``` text
GitHub Repository
│
├── Source code
├── Configuration
├── Tests
└── Documentation

External Storage
│
├── Large cleaned dataset
├── Chunked data
├── Embeddings
└── Weaviate vector collection
```

Do not commit:

-   Large datasets
-   Embedding files
-   Vector indexes
-   Model weights
-   `.env` files
-   API keys
-   Local virtual environments

------------------------------------------------------------------------

## Project Status

### Completed

-   [x] Initial repository setup
-   [x] Initial review cleaning pipeline

### In Progress

-   [ ] Finalize cleaned corpus
-   [ ] Implement chunking pipeline
-   [ ] Define Weaviate schema
-   [ ] Implement Weaviate ingestion
-   [ ] Implement embedding pipeline
-   [ ] Implement BM25 retrieval
-   [ ] Implement dense retrieval
-   [ ] Implement hybrid retrieval
-   [ ] Implement cross-encoder re-ranking
-   [ ] Implement subjectivity classification
-   [ ] Implement polarity classification
-   [ ] Implement LLM generation
-   [ ] Build Streamlit application
-   [ ] Run retrieval and classification evaluation
-   [ ] Run RAGAS evaluation
-   [ ] Run ablation studies

------------------------------------------------------------------------

## Team Responsibilities

The project is divided into three main areas:

### Knowledge Base

-   Data collection
-   Data cleaning
-   Chunking
-   Metadata tagging
-   Corpus statistics
-   Held-out evaluation set preparation

### Retrieval

-   BM25 retrieval
-   Dense retrieval
-   Weaviate integration
-   Hybrid fusion
-   Cross-encoder re-ranking
-   Streamlit retrieval interface
-   Retrieval evaluation

### Generation & Evaluation

-   Subjectivity classification
-   Polarity classification
-   LLM generation
-   Grounding/citation
-   Manual annotation
-   Classification evaluation
-   RAGAS evaluation
-   Ablation studies

All team members contribute to the final report, slides, and project
presentation.

------------------------------------------------------------------------

## Academic Project

**Course:** AI6127\
**Project:** RAG-Based Aspect-Level Opinion Mining for Smartphone
Reviews\
**Institution:** Nanyang Technological University (NTU)

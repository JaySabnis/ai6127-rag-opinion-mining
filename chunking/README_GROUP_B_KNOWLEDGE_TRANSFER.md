# Knowledge Transfer: Subgroup A → Group B

## How the Smartphone Review Knowledge Base Works

This document explains exactly what Group B receives from Subgroup A,
why the data was structured this way, how the chunking works, what every
JSON object means, and how the corpus should be used for embeddings and
retrieval.

------------------------------------------------------------------------

# 1. What You Are Receiving

The primary handoff artifact is:

``` text
corpus.jsonl
```

It contains the finalized retrieval chunks generated from the cleaned
smartphone-review dataset.

Current finalized corpus:

``` text
Documents: 395,038
Chunks:    459,418
Words:     22,754,056
Types:     103,954
```

A document means an original review.

A chunk means a retrieval unit derived from that review.

Therefore:

``` text
1 review != necessarily 1 chunk
```

but most reviews are short enough to remain one chunk.

The corpus also comes with:

``` text
corpus_statistics.json
```

which contains the generated corpus statistics and chunking
configuration.

------------------------------------------------------------------------

# 2. Why We Chunked Reviews

A complete review is not always a good retrieval unit.

Consider a 500-word review:

``` text
The phone looks good...
The screen is bright...
The battery lasts all day...
The camera is excellent...
The carrier setup was difficult...
...
```

If the user asks:

> "How is the camera?"

Retrieving the entire 500-word review is less efficient than retrieving
the section containing the camera opinion.

Therefore, long reviews are divided into smaller retrieval units.

The goal is:

``` text
Review
  |
  +--> Chunk 0
  +--> Chunk 1
  +--> Chunk 2
  ...
```

This gives the retrieval system more precise evidence.

------------------------------------------------------------------------

# 3. Chunking Rules

The project requirement was:

-   Reviews \<=150 words remain as one chunk.
-   Reviews \>150 words are split at sentence boundaries.
-   Target chunk size is approximately 100--150 words.
-   Consecutive chunks overlap by one sentence.
-   The title is added to the first chunk.
-   `doc_id` is preserved across all chunks from the same review.

------------------------------------------------------------------------

# 4. Short Review Example

Suppose:

``` text
Title:
Great phone

Text:
The battery is excellent and the camera is very good.
```

Because the review is short:

``` text
one review
    |
    v
one chunk
```

The chunk text becomes approximately:

``` text
Great phone. The battery is excellent and the camera is very good.
```

------------------------------------------------------------------------

# 5. Long Review Example

Suppose a review has these sentences:

``` text
S1
S2
S3
S4
S5
S6
S7
S8
```

The chunker may produce:

``` text
Chunk 0:
S1
S2
S3
S4

Chunk 1:
S4
S5
S6
S7

Chunk 2:
S7
S8
```

The overlap is intentional.

The repeated sentence gives the next chunk enough context to remain
meaningful.

Without overlap:

``` text
Chunk 0 -> ends with context
Chunk 1 -> starts abruptly
```

With overlap:

``` text
Chunk 0 -> context
Chunk 1 -> previous sentence + new content
```

This helps retrieval when an important opinion crosses a chunk boundary.

------------------------------------------------------------------------

# 6. Why Sentence Boundaries Matter

We deliberately split at sentence boundaries rather than blindly taking
every N words.

Bad approach:

``` text
"The camera produces very good photos but the battery"
```

followed by:

``` text
"life is disappointing..."
```

The opinion is now artificially broken.

Our approach keeps complete sentences together.

Therefore, a very long individual sentence can produce a chunk greater
than 150 words.

This is a known and documented property of the corpus.

The 150-word value is a **target/max guideline for normal sentence
groups**, while sentence integrity is preserved.

------------------------------------------------------------------------

# 7. Title Handling

Titles are useful opinion signals.

For normal reviews:

``` text
title + body
```

are combined in the first chunk.

Example:

``` text
Title:
Battery is amazing

Body:
I used this phone all day...
```

First chunk:

``` text
Battery is amazing. I used this phone all day...
```

For a multi-chunk review, the title is added only to Chunk 0.

------------------------------------------------------------------------

# 8. Title-Only Reviews

Some source records have:

``` text
text = ""
```

but contain a useful title.

These were not discarded.

Example:

``` text
Title:
Great quality
Body:
(empty)
```

becomes:

``` text
Chunk:
Great quality
```

This matters because deleting these records would unnecessarily lose
source information.

------------------------------------------------------------------------

# 9. Parent-Document Relationship

The most important relationship for Group B is:

``` text
doc_id
```

Suppose:

``` text
doc_id = abc123
```

produces:

``` text
abc123_0
abc123_1
abc123_2
```

Then:

``` text
chunk_id       doc_id
---------      ------
abc123_0       abc123
abc123_1       abc123
abc123_2       abc123
```

The chunks can be independently retrieved, while `doc_id` allows Group B
to recover/group evidence belonging to the same original review.

This is useful when:

-   multiple chunks from the same review are retrieved
-   evidence needs to be grouped
-   parent-document context is needed
-   duplicate evidence from one review should be controlled

------------------------------------------------------------------------

# 10. JSONL Format

`corpus.jsonl` means:

> JSON Lines

Each line is **one JSON object representing one retrieval chunk**.

Conceptually:

``` json
{
  "chunk_id": "...",
  "doc_id": "...",
  "chunk_index": 0,
  "text": "...",
  "chunk_word_count": 42,
  "brand": "Samsung",
  "model": "Galaxy S10",
  "rating": 5,
  "timestamp": "2019-...",
  "verified": true,
  "helpful_votes": 12,
  "is_renewed": false,
  "aspect_tags": ["camera", "battery"],
  "mentions_other_model": false,
  "source_url": "..."
}
```

There is one such object per line.

This format is convenient because Group B can stream the file instead of
loading the entire 459k-chunk corpus into memory.

------------------------------------------------------------------------

# 11. Schema Explained

## `chunk_id`

Unique ID of the chunk.

Use this as the identity of the retrieval unit.

It should be preserved through:

``` text
embedding
indexing
retrieval
evaluation
```

------------------------------------------------------------------------

## `doc_id`

Original review/document ID.

Use it to connect chunks back to the parent review.

------------------------------------------------------------------------

## `chunk_index`

Chunk position within the parent review.

Example:

``` text
0
1
2
3
```

------------------------------------------------------------------------

## `text`

The actual chunk text.

This is the primary field to embed.

For dense retrieval:

``` text
text -> embedding model -> vector
```

For BM25:

``` text
text -> tokenizer/index -> lexical retrieval
```

------------------------------------------------------------------------

## `chunk_word_count`

Number of words in `text`.

Useful for:

-   debugging
-   analysis
-   retrieval filtering
-   statistics

------------------------------------------------------------------------

## `brand`

Phone brand.

Example:

``` text
Apple
Samsung
Motorola
Google
LG
```

This can be useful for metadata filtering.

------------------------------------------------------------------------

## `model`

Phone model when known.

Example:

``` text
iPhone 11
Galaxy S10
Moto G Fast (2020)
```

Some source records have no reliable model.

**Do not automatically replace missing models with guesses.**

The missing-model cases were verified as source-data limitations.

------------------------------------------------------------------------

## `rating`

Original review star rating:

``` text
1–5
```

This is extremely useful for the downstream opinion-mining task.

For example:

``` text
rating = 5
```

is strong contextual evidence of an overall positive review, although it
should not automatically be treated as the sentiment of every individual
aspect.

A five-star review can still say:

> "The camera is great but the battery is terrible."

Therefore, Group B should distinguish:

``` text
overall rating
```

from:

``` text
aspect-level sentiment
```

------------------------------------------------------------------------

## `timestamp`

Review date.

Useful for:

-   temporal analysis
-   filtering
-   trend analysis
-   time-aware evaluation

------------------------------------------------------------------------

## `verified`

Whether the source review was verified.

------------------------------------------------------------------------

## `helpful_votes`

Number of helpful votes.

Potentially useful as a ranking or evidence-quality feature.

For example, Group B could experiment with:

``` text
retrieval relevance
+
helpful-vote signal
```

but this should be an experiment rather than an assumption.

------------------------------------------------------------------------

## `is_renewed`

Whether the product is a renewed/refurbished device.

This can be useful when comparing review populations.

------------------------------------------------------------------------

## `aspect_tags`

Detected aspects from:

``` text
camera
battery
display
performance
service
carrier
```

Example:

``` json
"aspect_tags": ["camera", "display"]
```

This allows Group B to filter or boost candidate chunks.

For a query:

> "How is the camera?"

a possible retrieval strategy is:

``` text
semantic query
+
camera aspect filter/boost
```

However, aspect tags should not necessarily replace semantic retrieval.
They are metadata signals.

------------------------------------------------------------------------

## `mentions_other_model`

Indicates whether another phone model was detected in the chunk.

This is useful for comparison-heavy reviews.

Example:

``` text
"Compared with my old Galaxy S9, this phone has a much better camera."
```

The system may need to understand that:

``` text
current phone != mentioned comparison phone
```

------------------------------------------------------------------------

## `source_url`

Original source URL.

Preserve this when displaying evidence or tracing a result back to its
source.

------------------------------------------------------------------------

# 12. How Group B Should Build Embeddings

The basic flow should be:

``` text
corpus.jsonl
     |
     v
Read one JSON object
     |
     v
Extract `text`
     |
     v
Embedding model
     |
     v
Vector
```

The metadata should be stored alongside the vector.

Conceptually:

``` text
Vector
  |
  +-- chunk_id
  +-- doc_id
  +-- text
  +-- brand
  +-- model
  +-- rating
  +-- aspect_tags
  +-- timestamp
  +-- other metadata
```

This allows retrieval to return both:

``` text
semantic similarity
```

and:

``` text
the original evidence + metadata
```

------------------------------------------------------------------------

# 13. Recommended Retrieval Architecture

A sensible architecture is:

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
             Aspect-level Analysis
                        |
                        v
                   LLM Answer
```

The exact implementation is a Group B decision.

------------------------------------------------------------------------

# 14. Why Hybrid Retrieval Makes Sense

Smartphone reviews contain many exact terms:

``` text
iPhone 15
Galaxy S10
Snapdragon
5G
USB-C
OLED
battery
camera
```

BM25 can be very useful for exact terms.

Dense embeddings are useful for semantic variations.

For example:

``` text
Query:
"How long does the phone last on a charge?"
```

A review may say:

``` text
"The battery easily gets me through a full day."
```

There may be little exact word overlap, but semantic retrieval should
recognize the relationship.

Therefore:

``` text
BM25 + dense retrieval
```

can complement each other.

------------------------------------------------------------------------

# 15. Aspect-Level Retrieval

The project is not just general review retrieval.

The important target is:

``` text
product + aspect + opinion
```

For example:

> "How is the camera on the iPhone 15?"

The retrieval system should ideally prioritize:

``` text
brand/model = iPhone 15
aspect = camera
```

and then retrieve semantically relevant evidence.

Conceptually:

``` text
Query
  |
  +--> Product signal
  |
  +--> Aspect signal
  |
  +--> Semantic meaning
  |
  v
Relevant review chunks
```

------------------------------------------------------------------------

# 16. Important Warning About Ratings

Do not use the star rating as direct aspect sentiment.

Example:

``` text
Rating: 5

Text:
"The camera is excellent, but the battery is awful."
```

The overall rating is positive.

The camera is positive.

The battery is negative.

Therefore:

``` text
rating != aspect sentiment
```

The rating is contextual metadata that can help the model, not a
substitute for aspect-level sentiment extraction.

------------------------------------------------------------------------

# 17. Duplicate and Overlap Considerations

Because chunks intentionally overlap by one sentence, related chunks can
contain repeated text.

Also, different reviews can legitimately contain identical or nearly
identical short text.

Therefore, retrieval should be careful about returning:

``` text
Chunk A from review X
Chunk B from review X
```

as if they were two independent pieces of evidence.

Use `doc_id` to group evidence when necessary.

------------------------------------------------------------------------

# 18. Validation Status

The final corpus was validated.

Key structural results:

``` text
Documents:                 395,038
Chunks:                    459,418
Unique chunk IDs:          459,418
Duplicate chunk IDs:       0
Malformed JSON lines:      0
Word-count mismatches:     0
Empty chunks:              0
Missing required fields:   0
```

The corpus validation passed.

The validation also reported expected warnings:

``` text
Chunks <100 words
```

This is expected because short reviews remain intact.

``` text
Chunks >150 words
```

These occur where preserving sentence boundaries requires a very long
sentence to remain intact.

``` text
Duplicate text groups
```

These can occur naturally because of repeated reviews and overlap.

------------------------------------------------------------------------

# 19. Missing Models

There are some chunks where:

``` text
model = missing
```

This should **not** be treated as a chunking failure.

The source dataset itself contains reviews where a reliable model could
not be mapped.

The source-vs-corpus verification showed:

``` text
Source reviews: 395,038
Source missing model: 2,245

Missing model correctly preserved: 2,245
Incorrectly filled: 0

Model correctly preserved: 392,793
Model lost: 0
Model changed: 0
```

Therefore, downstream code should gracefully handle:

``` text
model = null / empty
```

rather than assuming every record has a model.

------------------------------------------------------------------------

# 20. Weaviate Was Removed

We originally prepared an ingestion path for Weaviate.

The test import worked, but the available Weaviate instance had an
object limit that was too small for the full corpus.

The corpus contains:

``` text
459,418 chunks
```

so the available 100k-object limit was not sufficient.

We therefore decided **not to use Weaviate for the project**.

This is not a Knowledge Base data problem.

The finalized corpus is independent of Weaviate.

The correct architecture is now:

``` text
Google Drive
     |
     v
corpus.jsonl
     |
     v
Group B downloads it
     |
     +--> Embeddings
     |
     +--> BM25
     |
     +--> Vector store / local index / chosen backend
     |
     v
Retrieval
```

The storage/index technology can be changed without changing the corpus.

------------------------------------------------------------------------

# 21. Google Drive Handoff

The recommended shared folder is:

``` text
AI6127 RAG Opinion Mining/
└── Knowledge Base/
    ├── corpus.jsonl
    ├── corpus_statistics.json
    └── KNOWLEDGE_BASE_HANDOFF.md
```

`corpus.jsonl` is the main file.

`corpus_statistics.json` provides supporting statistics.

The handoff documentation explains the schema and pipeline.

------------------------------------------------------------------------

# 22. How to Read the Corpus

Python example:

``` python
import json

with open("corpus.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        record = json.loads(line)

        text = record["text"]
        chunk_id = record["chunk_id"]
        doc_id = record["doc_id"]

        # Send `text` to the embedding model.
        # Store the rest as metadata.
```

Because the file is JSONL, it can be streamed.

You do not need to load all 459k objects into memory at once.

------------------------------------------------------------------------

# 23. Example Embedding Input

Given:

``` json
{
  "chunk_id": "abc123_0",
  "doc_id": "abc123",
  "chunk_index": 0,
  "text": "Excellent camera. Photos are sharp...",
  "brand": "Samsung",
  "model": "Galaxy S10",
  "rating": 5,
  "aspect_tags": ["camera"]
}
```

The embedding input should primarily be:

``` text
Excellent camera. Photos are sharp...
```

The metadata should remain attached to the resulting vector/index entry.

------------------------------------------------------------------------

# 24. Example Query Flow

User:

> "How is the camera on the Galaxy S10?"

Possible processing:

``` text
Query
  |
  v
Extract / infer:
  product = Galaxy S10
  aspect  = camera
  |
  v
Dense + BM25 retrieval
  |
  v
Filter/boost relevant metadata
  |
  v
Retrieve top chunks
  |
  v
Group by doc_id if necessary
  |
  v
Extract camera-specific opinions
  |
  v
Generate answer
```

The final answer should be grounded in retrieved evidence rather than
relying only on the LLM's general knowledge.

------------------------------------------------------------------------

# 25. What Group B Should NOT Do

Do not:

1.  Modify `chunk_id`
2.  Break the `doc_id` relationship
3.  Assume every review has a model
4.  Treat star rating as aspect sentiment
5.  Remove titles without a reason
6.  Remove title-only reviews automatically
7.  Assume every chunk must be exactly 100--150 words
8.  Depend on Weaviate
9.  Change the corpus without versioning the resulting dataset

If preprocessing must change, create a new corpus version so the
retrieval and evaluation results remain reproducible.

------------------------------------------------------------------------

# 26. Recommended Handoff Contract

Subgroup A provides:

``` text
corpus.jsonl
corpus_statistics.json
chunking code
metadata code
validation code
```

Group B owns:

``` text
embedding model
embedding generation
vector/lexical index
BM25
hybrid retrieval
reranking
retrieval evaluation
```

The downstream generation layer owns:

``` text
aspect-level opinion extraction
sentiment classification
evidence aggregation
final answer generation
```

------------------------------------------------------------------------

# 27. One-Sentence Summary

The simplest way to understand the handoff is:

> **Subgroup A has transformed raw smartphone reviews into validated,
> sentence-aware, metadata-rich retrieval chunks; Group B's job is now
> to turn those chunks into an efficient retrieval system using
> embeddings, lexical search, reranking, and eventually generation.**

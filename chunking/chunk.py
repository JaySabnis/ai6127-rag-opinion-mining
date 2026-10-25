"""
Sentence-aware review chunking.

Task 2 requirements:
- <= 150 words: keep the whole review as one chunk
- > 150 words: split at sentence boundaries
- Target chunk size: 100-150 words
- Overlap: one sentence between consecutive chunks
- Title is prepended only to the first chunk
"""

from __future__ import annotations

import re
from typing import Any


MIN_CHUNK_WORDS = 100
MAX_CHUNK_WORDS = 150


def word_count(text: str) -> int:
    """Count words using whitespace tokenization."""
    return len(text.split())


def split_sentences(text: str) -> list[str]:
    """
    Lightweight sentence splitter.

    We deliberately keep this dependency-free so the chunking pipeline
    does not require a separate NLP model just for sentence boundaries.

    Handles common ., !, ? sentence endings while avoiding splitting
    decimal numbers.
    """
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    # Protect common abbreviations and decimal numbers temporarily.
    protected = {
        "e.g.": "e§g§",
        "i.e.": "i§e§",
        "etc.": "etc§",
        "mr.": "mr§",
        "mrs.": "mrs§",
        "dr.": "dr§",
        "vs.": "vs§",
    }

    for original, replacement in protected.items():
        text = re.sub(
            re.escape(original),
            replacement,
            text,
            flags=re.IGNORECASE,
        )

    # Split only when punctuation is followed by whitespace + uppercase/
    # digit/quote. This avoids most accidental splits.
    parts = re.split(
        r"(?<=[.!?])\s+(?=[A-Z0-9\"'])",
        text,
    )

    restored = []

    reverse_protected = {
        "e§g§": "e.g.",
        "i§e§": "i.e.",
        "etc§": "etc.",
        "mr§": "mr.",
        "mrs§": "mrs.",
        "dr§": "dr.",
        "vs§": "vs.",
    }

    for part in parts:
        part = part.strip()

        for replacement, original in reverse_protected.items():
            part = part.replace(replacement, original)

        if part:
            restored.append(part)

    return restored


def _make_chunk(sentences: list[str]) -> str:
    return " ".join(sentence.strip() for sentence in sentences).strip()


def chunk_review(
    text: str,
    title: str | None = None,
    min_words: int = MIN_CHUNK_WORDS,
    max_words: int = MAX_CHUNK_WORDS,
) -> list[str]:
    """
    Convert one review into retrieval chunks.

    For reviews <= max_words:
        one chunk containing the whole review.

    For longer reviews:
        sentence-aware chunks targeting 100-150 words with
        one-sentence overlap.
    """
    text = re.sub(r"\s+", " ", str(text or "")).strip()

    # ---------------------------------------------------------
    # Case 0: title-only review
    # ---------------------------------------------------------
    # Keep reviews where the body is empty but the title
    # contains useful information.
    if not text:
        if title:
            title = str(title).strip()

            if title:
                return [title]

        return []

    total_words = word_count(text)

    # ---------------------------------------------------------
    # Case 1: short review -> one chunk
    # ---------------------------------------------------------
    if total_words <= max_words:
        if title:
            title = str(title).strip()
            if title:
                return [f"{title}. {text}"]

        return [text]

    # ---------------------------------------------------------
    # Case 2: long review -> sentence-aware chunks
    # ---------------------------------------------------------
    sentences = split_sentences(text)

    if not sentences:
        return [text]

    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    i = 0

    while i < len(sentences):
        sentence = sentences[i]
        sentence_words = word_count(sentence)

        # -----------------------------------------------------
        # Extremely long individual sentence
        # -----------------------------------------------------
        if sentence_words > max_words:
            if current:
                chunks.append(_make_chunk(current))
                current = []
                current_words = 0

            # Do not break a sentence because the requirement says
            # to split at sentence boundaries.
            chunks.append(sentence)

            # Next chunk should overlap by one sentence where possible.
            current = [sentence]
            current_words = sentence_words

            i += 1
            continue

        # -----------------------------------------------------
        # Sentence fits into current chunk
        # -----------------------------------------------------
        if current_words + sentence_words <= max_words:
            current.append(sentence)
            current_words += sentence_words
            i += 1
            continue

        # -----------------------------------------------------
        # Current chunk is ready
        # -----------------------------------------------------
        
        if current:
            chunks.append(_make_chunk(current))

            # One-sentence overlap where possible.
            overlap_sentence = current[-1]
            overlap_words = word_count(overlap_sentence)

            # If the overlap sentence itself leaves no room for the
            # current sentence, don't reuse it indefinitely.
            if overlap_words + sentence_words <= max_words:
                current = [overlap_sentence]
                current_words = overlap_words
                # Reconsider the current sentence so it is included
                # in the next chunk.
                continue
            else:
                # Cannot maintain the overlap without exceeding max_words.
                # Start the next chunk with the current sentence.
                current = []
                current_words = 0

                # The current sentence must now be processed.
                current.append(sentence)
                current_words = sentence_words
                i += 1
                continue

        # Do not increment i here.
        # The current sentence gets considered again for the
        # next chunk.

    # ---------------------------------------------------------
    # Final chunk
    # ---------------------------------------------------------
    if current:
        final_chunk = _make_chunk(current)

        if not chunks or final_chunk != chunks[-1]:
            chunks.append(final_chunk)

    # ---------------------------------------------------------
    # Merge very small final chunks where possible.
    # ---------------------------------------------------------
    if len(chunks) > 1:
        last_words = word_count(chunks[-1])

        if last_words < min_words:
            previous_words = word_count(chunks[-2])

            # Only merge if doing so doesn't create a chunk
            # larger than max_words.
            merged = f"{chunks[-2]} {chunks[-1]}"

            if word_count(merged) <= max_words:
                chunks[-2] = merged
                chunks.pop()

    # Add title only to first chunk.
    if title:
        title = str(title).strip()

        if title:
            chunks[0] = f"{title}. {chunks[0]}"

    return chunks


def create_chunk_records(review: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Convert one cleaned review into basic chunk records.

    Metadata enrichment is intentionally handled separately in
    metadata.py.
    """
    review_id = str(
        review.get("review_id")
        or review.get("doc_id")
        or review.get("id")
        or ""
    ).strip()

    text = (
        review.get("text_norm")
        or review.get("text")
        or ""
    )

    title = review.get("title") or ""

    chunks = chunk_review(
        text=text,
        title=title,
    )

    records = []

    for index, chunk_text in enumerate(chunks):
        records.append(
            {
                "chunk_id": f"{review_id}_{index}",
                "doc_id": review_id,
                "chunk_index": index,
                "text": chunk_text,
                "chunk_word_count": word_count(chunk_text),
            }
        )

    return records


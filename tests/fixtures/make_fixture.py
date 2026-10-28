"""Generate a tiny SYNTHETIC corpus in the corpus.jsonl schema for unit tests.

These reviews are hand-written test data, not real Amazon reviews. They only
exist so the retrieval code can be tested without the 459k-chunk corpus.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from chunking.metadata import detect_aspects  # noqa: E402

# (doc_id, brand, model, rating, [chunk texts])
REVIEWS = [
    ("s10a", "Samsung", "Galaxy S10", 5, ["Amazing camera. The night mode photos are sharp and the colors pop."]),
    ("s10b", "Samsung", "Galaxy S10", 2, ["The battery drains by noon even with light use. Very disappointed."]),
    ("s10c", "Samsung", "Galaxy S10", 4, ["Screen is gorgeous, the AMOLED display is bright outdoors.",
                                           "Screen is gorgeous. Gaming is smooth with no lag at all."]),
    ("s10d", "Samsung", "Galaxy S10", 1, ["It dies before lunch every single day and takes forever to charge."]),
    ("s10e", "Samsung", "Galaxy S10", 3, ["Pictures look washed out in low light compared to my old phone."]),
    ("s10f", "Samsung", "Galaxy S10", 5, ["Fast delivery and the seller was friendly."]),
    ("i11a", "Apple", "iPhone 11", 5, ["The camera on this iPhone is incredible, portrait photos look professional."]),
    ("i11b", "Apple", "iPhone 11", 4, ["Battery life easily lasts a full day of heavy use."]),
    ("i11c", "Apple", "iPhone 11", 2, ["The LCD screen resolution is low for the price."]),
    ("i11d", "Apple", "iPhone 11", 1, ["Customer service refused to honor the warranty replacement."]),
    ("i11e", "Apple", "iPhone 11", 5, ["Gets me through the whole day without needing a charger."]),
    ("i11p", "Apple", "iPhone 11 Pro", 5, ["The triple lens camera on the Pro is far better than the regular iPhone 11."]),
    ("i11q", "Apple", "iPhone 11 Pro", 3, ["Battery is fine but the phone gets hot while charging."]),
    ("motoa", "Motorola", "Moto G Power", 5, ["Battery lasts three days, best battery I have ever had."]),
    ("motob", "Motorola", "Moto G Power", 2, ["Apps are slow to open and the phone lags when multitasking."]),
    ("motoc", "Motorola", "Moto G Power", 4, ["Good signal on Verizon, no dropped calls."]),
    ("nomod", "Generic", None, 3, ["Works okay as a backup phone, nothing special."]),
]


def build(path: Path) -> int:
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for doc_id, brand, model, rating, chunks in REVIEWS:
            for i, text in enumerate(chunks):
                rec = {
                    "chunk_id": f"{doc_id}_{i}",
                    "doc_id": doc_id,
                    "chunk_index": i,
                    "text": text,
                    "chunk_word_count": len(text.split()),
                    "brand": brand,
                    "model": model,
                    "rating": rating,
                    "timestamp": "2020-01-01",
                    "verified": True,
                    "helpful_votes": 0,
                    "is_renewed": False,
                    "aspect_tags": detect_aspects(text),
                    "mentions_other_model": False,
                    "source_url": f"https://example.invalid/{doc_id}",
                }
                f.write(json.dumps(rec) + "\n")
                n += 1
    return n


if __name__ == "__main__":
    out = Path(__file__).with_name("mini_corpus.jsonl")
    print(f"wrote {build(out)} chunks to {out}")

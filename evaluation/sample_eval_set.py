"""
Sample chunks from corpus.jsonl for the manually labeled held-out eval set.

Stratified by star rating (35% 1-star, 10% each 2/3/4-star, 35% 5-star of
1,200 chunks) so positive and negative opinions are roughly balanced.
At most one chunk per review (doc_id) and no duplicate texts.

Usage (from the repository root):

    uv run python evaluation/sample_eval_set.py ../data/corpus.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict

QUOTA = {1: 420, 2: 120, 3: 120, 4: 120, 5: 420}
SEED = 6127


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", help="Path to corpus.jsonl")
    parser.add_argument("--out", default="evaluation/eval_sample.csv")
    args = parser.parse_args()

    by_rating = defaultdict(list)
    with open(args.corpus, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            by_rating[r["rating"]].append(r)

    rng = random.Random(SEED)
    seen_docs, seen_texts, rows = set(), set(), []
    for rating, quota in QUOTA.items():
        pool = by_rating[rating]
        rng.shuffle(pool)
        taken = 0
        for r in pool:
            text_key = " ".join(r["text"].lower().split())
            if r["doc_id"] in seen_docs or text_key in seen_texts:
                continue
            seen_docs.add(r["doc_id"])
            seen_texts.add(text_key)
            rows.append(r)
            taken += 1
            if taken == quota:
                break
        print(f"{rating}-star: {taken}")

    assert len(rows) == sum(QUOTA.values())
    assert len({r["doc_id"] for r in rows}) == len(rows)

    # Mix ratings so annotators don't see 420 one-star chunks in a row.
    rng.shuffle(rows)

    # Rating is left out of the annotation sheet so it can't bias the labels;
    # join back on chunk_id when needed.
    with open(args.out, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["chunk_id", "doc_id", "brand", "model", "aspect_tags",
                         "text", "subjectivity", "polarity"])
        for r in rows:
            writer.writerow([r["chunk_id"], r["doc_id"], r["brand"], r["model"],
                             ";".join(r["aspect_tags"]), r["text"], "", ""])

    print(f"Wrote {len(rows)} chunks to {args.out}")


if __name__ == "__main__":
    main()

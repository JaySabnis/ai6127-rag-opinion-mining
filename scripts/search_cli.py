"""Query the retrieval indexes from the command line.

    uv run python -m scripts.search_cli "How is the camera on the Galaxy S10?"
    uv run python -m scripts.search_cli "battery dies quickly" --mode all --k 5
    uv run python -m scripts.search_cli "screen" --model "iPhone 11" --mode bm25
    uv run python -m scripts.search_cli "camera quality" --json > out.json
"""

from __future__ import annotations

import argparse
import json
import textwrap

from retrieval import Retriever
from retrieval.retriever import MODES


def show(res, width: int) -> None:
    p = res.parsed
    print(f"\n### mode={res.mode}  detected model={p.model} aspects={p.aspects}  filters={res.filters}")
    if res.search_text != res.query:
        print(f"    searched for: {res.search_text!r}")
    print(f"    candidates={res.n_candidates or 'all'}  timings(ms)={res.timings_ms}")
    for h in res.hits:
        src = []
        if h.bm25_rank:
            src.append(f"bm25#{h.bm25_rank}")
        if h.dense_rank:
            src.append(f"dense#{h.dense_rank}")
        if h.aspect_boosted:
            src.append("aspect+")
        print(f"{h.rank:>3}. [{h.score:.4f}] {h.chunk_id}  {h.model} | {h.rating}* | {', '.join(src)}")
        print(textwrap.indent(textwrap.shorten(h.text, width), "      "))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--index-dir", default="indexes")
    ap.add_argument("--mode", default="hybrid", choices=MODES + ("all",))
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--candidate-k", type=int, default=100)
    ap.add_argument("--model", default=None, help="explicit model filter, e.g. 'Galaxy S10'")
    ap.add_argument("--brand", default=None)
    ap.add_argument("--min-rating", type=int, default=None)
    ap.add_argument("--max-rating", type=int, default=None)
    ap.add_argument("--no-auto-filter", action="store_true")
    ap.add_argument("--aspect-weight", type=float, default=0.5,
                    help="Retrieval aspect boost weight (default: 0.5)")
    ap.add_argument("--keep-all-chunks", action="store_true", help="do not de-duplicate by review")
    ap.add_argument("--no-strip-model", action="store_true", help="keep the model name in the searched text")
    ap.add_argument("--no-dense", action="store_true", help="load BM25 only")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--width", type=int, default=220)
    args = ap.parse_args()

    r = Retriever.load(args.index_dir, use_dense=not args.no_dense)
    modes = (("bm25",) if args.no_dense else MODES) if args.mode == "all" else (args.mode,)
    filters = {"model": args.model, "brand": args.brand, "min_rating": args.min_rating, "max_rating": args.max_rating}
    results = [
        r.retrieve(
            args.query,
            mode=m,
            k=args.k,
            candidate_k=args.candidate_k,
            filters={k: v for k, v in filters.items() if v is not None},
            auto_filter=not args.no_auto_filter,
            aspect_weight=args.aspect_weight,
            max_per_doc=None if args.keep_all_chunks else 1,
            strip_model=not args.no_strip_model,
        )
        for m in modes
    ]
    if args.json:
        print(json.dumps([res.to_dict() for res in results], indent=2, default=str))
    else:
        for res in results:
            show(res, args.width)


if __name__ == "__main__":
    main()

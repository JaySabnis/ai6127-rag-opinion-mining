"""Run the reranker on a saved candidate pool without the UI or rewrite API.

Input : candidate JSON/JSONL, query, and optional query aspects
Output: JSON with ranked candidates, score components, timings, and settings

Steps: read candidates -> configure MiniLM/aspect gate -> rerank -> save JSON.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from retrieval.rerank import CrossEncoderReranker, RerankerConfig
from chunking.metadata import ASPECT_KEYWORDS


# ---- candidate file loading
def read_candidates(path: Path, *, max_candidates: int | None = None) -> list[dict]:
    """Accept a JSON list, a {candidates: [...]} response, or chunk JSONL."""
    if path.suffix.lower() == ".jsonl":
        candidates = []
        with path.open(encoding="utf-8-sig") as stream:
            for line in stream:
                if not line.strip():
                    continue
                # Stop early if a full corpus was accidentally supplied as a pool.
                if max_candidates is not None and len(candidates) >= max_candidates:
                    raise ValueError(
                        f"Input exceeds {max_candidates} candidates; pass a small "
                        "retrieval pool instead of the full corpus"
                    )
                candidates.append(json.loads(line))
    else:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        candidates = data.get("candidates") if isinstance(data, dict) else data
    if not isinstance(candidates, list):
        raise ValueError("Input must contain a candidate list")
    if max_candidates is not None and len(candidates) > max_candidates:
        raise ValueError(f"Input exceeds {max_candidates} candidates")
    return candidates


# ---- command-line configuration and output
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--aspects", nargs="*", choices=tuple(ASPECT_KEYWORDS), default=[])
    parser.add_argument("--rrf-weight", type=float, default=1.0)
    parser.add_argument("--cross-encoder-weight", type=float, default=0.6)
    parser.add_argument("--aspect-weight", type=float, default=1.2)
    parser.add_argument("--aspect-threshold", type=float, default=-6.0)
    parser.add_argument("--aspect-temperature", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--max-candidates", type=int, default=1000,
        help="Reject larger input pools to catch accidental full-corpus scoring",
    )
    args = parser.parse_args()
    if args.max_candidates <= 0:
        parser.error("--max-candidates must be positive")
    if args.input.resolve() == args.output.resolve():
        parser.error("--output must not overwrite the input file")
    try:
        candidates = read_candidates(args.input, max_candidates=args.max_candidates)
    except ValueError as exc:
        parser.error(str(exc))
    reranker = CrossEncoderReranker(RerankerConfig(
        device=args.device,
        local_files_only=args.local_files_only,
        rrf_weight=args.rrf_weight,
        cross_encoder_weight=args.cross_encoder_weight,
        aspect_weight=args.aspect_weight,
        aspect_threshold=args.aspect_threshold,
        aspect_temperature=args.aspect_temperature,
    ))
    result = reranker.rerank_with_details(
        args.query, candidates, top_k=args.top_k, query_aspects=args.aspects,
    )
    # Include score components and settings so the saved ranking can be inspected.
    payload = {"query": args.query, "input_path": str(args.input), **asdict(result)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Ranked {result.input_count} candidates; returned {len(result.candidates)}")
    print(f"Timings (ms): {result.timing_ms}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()

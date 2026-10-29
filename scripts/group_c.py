"""Group C CLI; python -m scripts.group_c --help. No local LLM required."""
import argparse
import json
from pathlib import Path
import sys

from generation.client import DeepSeekClient, ROOT, write_json
from generation.io import read_chunks, write_predictions, write_run
from generation.pipeline import OpinionPipeline
from generation.schema import ASPECTS
from generation.sentiment import SentimentClassifier, aggregate_sentiment
from generation.verifier import FaithfulnessChecker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--env-file", type=Path)
    common.add_argument("--model")
    common.add_argument("--no-cache", action="store_true")
    common.add_argument("--cache-dir", type=Path, default=ROOT / ".cache/group_c")
    common.add_argument("--max-api-calls", type=int, default=32)
    classify = sub.add_parser("classify", parents=[common], help="Classify single text or JSONL/CSV chunks")
    source = classify.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path)
    source.add_argument("--text")
    classify.add_argument("--aspect", choices=ASPECTS)
    classify.add_argument("--brand")
    classify.add_argument("--phone-model")
    classify.add_argument("--limit", type=int)
    classify.add_argument("--output", type=Path, default=ROOT / "evaluation/predictions/deepseek_overall.csv")
    run = sub.add_parser("run", parents=[common], help="Classify -> ratios -> summarize -> check")
    run.add_argument("--input", type=Path, required=True)
    run.add_argument("--question", required=True)
    run.add_argument("--aspects", nargs="*", choices=ASPECTS, default=[])
    run.add_argument("--language", choices=["zh", "en"], default="zh")
    run.add_argument("--limit", type=int)
    run.add_argument("--no-check", action="store_true", help="Ablation: keep unverified draft")
    run.add_argument("--output", type=Path, default=ROOT / "results/group_c/run.json")
    check = sub.add_parser("check", parents=[common], help="Check an existing summary, including hallucinations")
    check.add_argument("--input", type=Path, required=True, help="Original retrieved evidence")
    summary = check.add_mutually_exclusive_group(required=True)
    summary.add_argument("--summary")
    summary.add_argument("--summary-file", type=Path, help="Plain text, or JSON with sentences")
    check.add_argument("--output", type=Path, default=ROOT / "results/group_c/check.json")
    args = parser.parse_args()
    try:
        client = DeepSeekClient(env_file=args.env_file, model=args.model,
            cache_dir=None if args.no_cache else args.cache_dir, max_calls=args.max_api_calls)
        if args.command == "classify":
            chunks = read_chunks(args.input, args.limit) if args.input else [{
                "chunk_id": "single_0", "doc_id": "single", "text": args.text,
                "brand": args.brand, "model": args.phone_model}]
            rows = SentimentClassifier(client).classify(chunks, args.aspect)
            write_predictions(args.output, rows)
            print(json.dumps({"predictions": rows if args.text else len(rows),
                              "statistics": aggregate_sentiment(rows), "api": client.telemetry()},
                             ensure_ascii=False, indent=2))
        elif args.command == "run":
            result = OpinionPipeline(client).run(args.question, read_chunks(args.input, args.limit),
                aspects=args.aspects, language=args.language, verify=not args.no_check)
            write_run(args.output, result)
            print(result["answer"])
            print(json.dumps({"statistics": result["statistics"],
                "flagged_sentences": len(result["flagged_sentences"]), "api": result["api"]},
                ensure_ascii=False, indent=2))
        else:
            text = args.summary
            if args.summary_file:
                text = args.summary_file.read_text(encoding="utf-8-sig")
                if args.summary_file.suffix == ".json":
                    text = json.loads(text)["sentences"]
            checked = FaithfulnessChecker(client).check(text, read_chunks(args.input))
            result = {"checks": checked, "flagged_sentences": [s for s in checked if s["flagged"]],
                      "api": client.telemetry()}
            write_json(args.output, result)
            print(json.dumps(result, ensure_ascii=False, indent=2))
    except (RuntimeError, ValueError, OSError) as exc:
        parser.exit(1, f"Group C failed: {exc}\n")
    print("Saved:", args.output)


if __name__ == "__main__":
    main()

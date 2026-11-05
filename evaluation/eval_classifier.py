"""
Score classifier predictions against the gold labels in eval.xls
(precision, recall and F1 per class, as required by Q4).

Predictions come from the generation side as a CSV with columns

    chunk_id,label        label in {positive, negative, mixed, neutral}

Put it in evaluation/predictions/<run_name>.csv, then run (from the
repository root):

    uv run python evaluation/eval_classifier.py evaluation/predictions/<run_name>.csv

Scores are reported for both stages of the task and overall:
  1. subjectivity: opinionated vs neutral, on all chunks
  2. polarity: positive / negative / mixed, on chunks that are opinionated in the gold set
  3. overall: the 4 labels
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import xlrd
from sklearn.metrics import classification_report, confusion_matrix

HERE = Path(__file__).parent
EVAL_XLS = HERE / "eval.xls"
RESULTS = HERE / "results"

LABELS = ["positive", "negative", "mixed", "neutral"]
OPINION = ["positive", "negative", "mixed"]


def subjectivity(label: str) -> str:
    return "neutral" if label == "neutral" else "opinionated"


def validated_labels(rows, *, source: str) -> dict[str, str]:
    """Reject ambiguous input instead of silently overwriting duplicate IDs."""
    result = {}
    for row in rows:
        cid, label = str(row["chunk_id"]).strip(), str(row["label"]).strip()
        if not cid or label not in LABELS:
            raise ValueError(f"{source}: blank ID or invalid label at record {len(result) + 1}")
        if cid in result:
            raise ValueError(f"{source}: duplicate chunk_id {cid}")
        result[cid] = label
    if not result:
        raise ValueError(f"{source}: no records")
    return result


def load_gold(path=EVAL_XLS) -> dict[str, str]:
    sheet = xlrd.open_workbook(str(path)).sheet_by_index(0)
    header = sheet.row_values(0)
    cid, lab = header.index("chunk_id"), header.index("label")
    return validated_labels(({"chunk_id": sheet.cell_value(r, cid), "label": sheet.cell_value(r, lab)}
                             for r in range(1, sheet.nrows)), source="gold")


def report(name: str, gold: list[str], pred: list[str], labels: list[str]) -> dict:
    print(f"\n== {name} (n={len(gold)})")
    print(classification_report(gold, pred, labels=labels, digits=3, zero_division=0))
    return classification_report(gold, pred, labels=labels, output_dict=True, zero_division=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", help="CSV with chunk_id,label")
    parser.add_argument("--gold", type=Path, default=EVAL_XLS)
    parser.add_argument("--output-dir", type=Path, default=RESULTS)
    args = parser.parse_args()

    gold = load_gold(args.gold)
    with open(args.predictions, encoding="utf-8-sig") as f:
        pred = validated_labels(csv.DictReader(f), source="predictions")

    bad = {l for l in pred.values() if l not in LABELS}
    if bad:
        raise SystemExit(f"Unknown predicted labels: {bad}. Use one of {LABELS}.")
    missing = gold.keys() - pred.keys()
    if missing:
        raise SystemExit(f"{len(missing)} gold chunks have no prediction, e.g. {sorted(missing)[:3]}")

    ids = sorted(gold)
    opinionated = [c for c in ids if gold[c] in OPINION]
    present = lambda labels, ids_: [l for l in labels if any(gold[c] == l for c in ids_)]

    results = {
        "run": Path(args.predictions).stem,
        "n": len(ids),
        "provenance": {
            "gold_file": args.gold.name,
            "gold_sha256": hashlib.sha256(args.gold.read_bytes()).hexdigest(),
            "predictions_file": Path(args.predictions).name,
            "predictions_sha256": hashlib.sha256(Path(args.predictions).read_bytes()).hexdigest(),
            "n_predictions": len(pred),
            "n_predictions_outside_gold": len(pred.keys() - gold.keys()),
            "scope": "existing human gold; labels and model predictions unchanged",
        },
        "subjectivity": report("Stage 1: subjectivity",
                               [subjectivity(gold[c]) for c in ids],
                               [subjectivity(pred[c]) for c in ids],
                               ["opinionated", "neutral"]),
        # A chunk predicted neutral here counts as a polarity miss.
        "polarity": report("Stage 2: polarity (gold-opinionated chunks)",
                           [gold[c] for c in opinionated], [pred[c] for c in opinionated],
                           present(OPINION, opinionated)),
        "overall": report("Overall: 4 labels",
                          [gold[c] for c in ids], [pred[c] for c in ids], present(LABELS, ids)),
    }

    print("Confusion matrix (rows = gold, columns = predicted):", LABELS)
    print(confusion_matrix([gold[c] for c in ids], [pred[c] for c in ids], labels=LABELS))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / f"classifier_{results['run']}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()

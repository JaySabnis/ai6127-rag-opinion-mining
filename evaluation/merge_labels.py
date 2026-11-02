"""
Merge the annotators' CSVs, report inter-annotator agreement, and build
the final eval.xls.

1. Put every downloaded labels_batch<b>_<name>.csv into evaluation/labels/
2. Run (from the repository root):

       uv run python evaluation/merge_labels.py

   It prints the agreement, writes results/agreement.json, and lists every
   chunk the annotators disagree on in labels/disagreements.csv.
3. Fill the `final_label` column of labels/disagreements.csv and run again.
   Once every disagreement is resolved it writes evaluation/eval.xls.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import xlwt
from sklearn.metrics import cohen_kappa_score

HERE = Path(__file__).parent
LABELS_DIR = HERE / "labels"
DISAGREEMENTS = LABELS_DIR / "disagreements.csv"
SAMPLE = HERE / "eval_sample.csv"
AGREEMENT_JSON = HERE / "results" / "agreement.json"
EVAL_XLS = HERE / "eval.xls"

LABELS = {"positive", "negative", "mixed", "neutral", "unusable"}
OPINION = {"positive", "negative", "mixed"}


def subjectivity(label: str) -> str:
    return "neutral" if label == "neutral" else "opinionated"


def load_annotations() -> dict[str, dict[str, dict[str, str]]]:
    """batch -> annotator -> chunk_id -> label (unlabeled rows skipped)."""
    ann = defaultdict(lambda: defaultdict(dict))
    for path in sorted(LABELS_DIR.glob("labels_batch*.csv")):
        with open(path, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                label = row["label"].strip()
                if not label:
                    continue
                if label not in LABELS:
                    raise SystemExit(f"{path.name}: unknown label {label!r} for {row['chunk_id']}")
                ann[row["batch"]][row["annotator"]][row["chunk_id"]] = label
    if not ann:
        raise SystemExit(f"No labels_batch*.csv files in {LABELS_DIR}")
    return ann


def agreement(a: list[str], b: list[str]) -> dict:
    if not a:
        return {"n": 0, "percent": None, "kappa": None}
    kappa = cohen_kappa_score(a, b)
    return {
        "n": len(a),
        "percent": round(100 * sum(x == y for x, y in zip(a, b)) / len(a), 1),
        # Kappa is undefined when both annotators used a single class.
        "kappa": None if math.isnan(kappa) else round(float(kappa), 3),
    }


def pair_agreement(x: dict[str, str], y: dict[str, str]) -> dict:
    common = sorted(x.keys() & y.keys())
    usable = [c for c in common if "unusable" not in (x[c], y[c])]
    both_opinion = [c for c in usable if x[c] in OPINION and y[c] in OPINION]
    return {
        "label": agreement([x[c] for c in common], [y[c] for c in common]),
        "subjectivity": agreement([subjectivity(x[c]) for c in usable],
                                  [subjectivity(y[c]) for c in usable]),
        "polarity": agreement([x[c] for c in both_opinion], [y[c] for c in both_opinion]),
    }


def report_agreement(ann) -> None:
    pairs = []
    for batch, by_annotator in sorted(ann.items()):
        names = sorted(by_annotator)
        if len(names) < 2:
            print(f"Batch {batch}: only {names}, no agreement yet")
        for a, b in combinations(names, 2):
            pairs.append({"batch": batch, "annotators": [a, b],
                          **pair_agreement(by_annotator[a], by_annotator[b])})

    print("\nInter-annotator agreement (percent / Cohen's kappa)")
    for p in pairs:
        cells = "  ".join(f"{k}: {p[k]['percent']}% / {p[k]['kappa']} (n={p[k]['n']})"
                          for k in ("label", "subjectivity", "polarity"))
        print(f"  Batch {p['batch']} {p['annotators'][0]} vs {p['annotators'][1]}  {cells}")

    # Mean of the pairwise kappas (Light's kappa) and pooled percent agreement.
    overall = {}
    for k in ("label", "subjectivity", "polarity"):
        kappas = [p[k]["kappa"] for p in pairs if p[k]["kappa"] is not None]
        n = sum(p[k]["n"] for p in pairs)
        agreed = sum(p[k]["n"] * p[k]["percent"] / 100 for p in pairs if p[k]["n"])
        overall[k] = {"n": n,
                      "percent": round(100 * agreed / n, 1) if n else None,
                      "mean_kappa": round(sum(kappas) / len(kappas), 3) if kappas else None}
        print(f"  Overall {k}: {overall[k]['percent']}% / mean kappa {overall[k]['mean_kappa']}")

    AGREEMENT_JSON.parent.mkdir(exist_ok=True)
    AGREEMENT_JSON.write_text(json.dumps({"pairs": pairs, "overall": overall}, indent=2))
    print(f"Wrote {AGREEMENT_JSON}")


def final_labels(ann, texts: dict[str, str]) -> dict[str, str]:
    """Agreed or majority label per chunk; disagreements go to DISAGREEMENTS."""
    by_chunk = defaultdict(dict)
    for by_annotator in ann.values():
        for name, labels in by_annotator.items():
            for chunk_id, label in labels.items():
                by_chunk[chunk_id][name] = label

    # Keep decisions already written into disagreements.csv.
    resolved = {}
    if DISAGREEMENTS.exists():
        with open(DISAGREEMENTS, encoding="utf-8-sig") as f:
            resolved = {r["chunk_id"]: r["final_label"].strip() for r in csv.DictReader(f)}

    final, open_rows = {}, []
    for chunk_id, votes in by_chunk.items():
        (top, top_n), *rest = Counter(votes.values()).most_common()
        if not rest or top_n > rest[0][1]:
            final[chunk_id] = top
            continue
        decision = resolved.get(chunk_id, "")
        if decision and decision not in LABELS:
            raise SystemExit(f"disagreements.csv: unknown final_label {decision!r} for {chunk_id}")
        if decision:
            final[chunk_id] = decision
        open_rows.append([chunk_id, "; ".join(f"{n}={l}" for n, l in sorted(votes.items())),
                          decision, texts.get(chunk_id, "")])

    with open(DISAGREEMENTS, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["chunk_id", "labels", "final_label", "text"])
        writer.writerows(sorted(open_rows))
    unresolved = sum(1 for r in open_rows if not r[2])
    print(f"\n{len(open_rows)} disagreements ({unresolved} unresolved) -> {DISAGREEMENTS}")
    return final if unresolved == 0 else {}


def write_eval_xls(final: dict[str, str], texts: dict[str, str]) -> None:
    rows = sorted((c, l) for c, l in final.items() if l != "unusable")
    book = xlwt.Workbook(encoding="utf-8")
    sheet = book.add_sheet("eval")
    header = ["chunk_id", "text", "label", "subjectivity", "polarity"]
    for col, name in enumerate(header):
        sheet.write(0, col, name)
    for i, (chunk_id, label) in enumerate(rows, start=1):
        values = [chunk_id, texts[chunk_id], label, subjectivity(label),
                  label if label in OPINION else ""]
        for col, value in enumerate(values):
            sheet.write(i, col, value)
    book.save(str(EVAL_XLS))
    print(f"Wrote {len(rows)} records to {EVAL_XLS}: {dict(Counter(l for _, l in rows))}")
    if len(rows) < 1000:
        print("Warning: fewer than 1,000 records (Q4 requires at least 1,000).")


def main() -> None:
    with open(SAMPLE, encoding="utf-8-sig") as f:
        texts = {r["chunk_id"]: r["text"] for r in csv.DictReader(f)}
    ann = load_annotations()
    report_agreement(ann)
    final = final_labels(ann, texts)
    if final:
        write_eval_xls(final, texts)
    else:
        print("Resolve every disagreement (final_label) to build eval.xls.")


if __name__ == "__main__":
    main()

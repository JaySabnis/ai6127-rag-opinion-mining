"""Lexicon baseline for the classifier: NLTK VADER on eval.xls.

VADER's compound score is mapped with the thresholds recommended in the
vaderSentiment README: >= 0.05 positive, <= -0.05 negative, otherwise neutral.
VADER has no "mixed" output, so every gold-mixed chunk counts as an error;
the binary comparison printed at the end leaves mixed out for a fairer view.

    uv run python -m nltk.downloader vader_lexicon
    uv run python evaluation/baseline_vader.py
    uv run python evaluation/eval_classifier.py evaluation/predictions/vader_baseline.csv
"""

from __future__ import annotations

import csv
from pathlib import Path

import xlrd
from nltk.sentiment import SentimentIntensityAnalyzer

HERE = Path(__file__).parent
OUT = HERE / "predictions" / "vader_baseline.csv"
LLM = HERE / "predictions" / "deepseek_overall.csv"


def vader_label(compound: float) -> str:
    if compound >= 0.05:
        return "positive"
    if compound <= -0.05:
        return "negative"
    return "neutral"


def main() -> None:
    sheet = xlrd.open_workbook(str(HERE / "eval.xls")).sheet_by_index(0)
    header = sheet.row_values(0)
    cid, txt, lab = (header.index(c) for c in ("chunk_id", "text", "label"))
    rows = [sheet.row_values(r) for r in range(1, sheet.nrows)]
    sia = SentimentIntensityAnalyzer()
    pred = {r[cid]: vader_label(sia.polarity_scores(r[txt])["compound"]) for r in rows}
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["chunk_id", "label"])
        w.writerows(pred.items())
    print(f"wrote {len(pred)} predictions to {OUT}")

    # Binary view: gold positive/negative chunks only; any other prediction is wrong.
    gold = {r[cid]: r[lab] for r in rows if r[lab] in ("positive", "negative")}
    with LLM.open(encoding="utf-8") as f:
        llm = {r["chunk_id"]: r["label"] for r in csv.DictReader(f)}
    for name, p in (("vader", pred), ("deepseek", llm)):
        correct = sum(p.get(k) == v for k, v in gold.items())
        print(f"binary pos/neg accuracy {name}: {correct}/{len(gold)} = {correct / len(gold):.3f}")


if __name__ == "__main__":
    vader_label_checks = {0.05: "positive", -0.05: "negative", 0.0: "neutral", 0.049: "neutral"}
    assert all(vader_label(k) == v for k, v in vader_label_checks.items())
    main()

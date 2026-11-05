"""Read Group A/B records and write files matching the evaluation branch."""
import csv
import json
from pathlib import Path
from generation.client import write_json
from generation.schema import normalize_chunks


def read_chunks(path, limit=None):
    path = Path(path)
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(data, dict):
            data = next((data[k] for k in ("chunks", "hits", "candidates") if k in data), None)
        if not isinstance(data, list):
            raise ValueError("JSON needs a list, or a chunks/hits/candidates list")
        rows = data[:limit] if limit else data
    else:
        rows = []
        with path.open(encoding="utf-8-sig", newline="") as file:
            source = csv.DictReader(file) if path.suffix.lower() == ".csv" else (
                json.loads(line) for line in file if line.strip())
            for row in source:
                rows.append(row)
                if limit and len(rows) >= limit:
                    break
    return normalize_chunks(rows)


def write_predictions(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if len({r["chunk_id"] for r in rows}) != len(rows):
        raise ValueError("Prediction CSV requires one aspect/scope per chunk_id")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["chunk_id", "label"])
        writer.writeheader()
        writer.writerows({"chunk_id": r["chunk_id"], "label": r["label"]} for r in rows)
    temporary.replace(path)
    write_json(path.with_suffix(".details.json"), rows)


def write_run(path, result):
    path = Path(path)
    write_json(path, result)
    # These files can be copied straight into evaluation/rag_outputs and timings.
    # Display placeholders are not model claims and must not inflate answer coverage.
    abstained = result.get("abstained", not bool(result["answer"].strip()))
    rag = {"question": result["question"], "contexts": result["contexts"],
           "answer": "" if abstained else result["answer"], "abstained": abstained,
           "display_answer": result["answer"], "status": result.get("status", "completed"),
           "checker_error": result.get("checker_error")}
    path.with_suffix(".rag.jsonl").write_text(json.dumps(rag, ensure_ascii=False) + "\n", encoding="utf-8")
    timing = {"query": result["question"], "n_chunks": len(result["chunks"]),
              "scope": result.get("timing_scope", "downstream"), **result["timing"]}
    path.with_suffix(".timings.jsonl").write_text(json.dumps(timing, ensure_ascii=False) + "\n", encoding="utf-8")
    zh = result.get("language") == "zh"
    lines = ["# 评论分析结果" if zh else "# Review analysis", "", result["answer"], "",
             "## 本次检索样本的好评比例" if zh else "## Positive ratio in this retrieved sample", ""]
    for s in result["statistics"]:
        product = " ".join(str(s[k] or "") for k in ("brand", "model")).strip() or ("未知机型" if zh else "Unknown model")
        ratio = ("无可用分母" if zh else "no denominator") if s["positive_ratio"] is None else f"{s['positive_ratio']:.1%}"
        n = s["n_relevant_reviews"]
        detail = f"，{n}条相关独立评论；" if zh else f", {n} relevant unique reviews; "
        lines.append(f"- {product} / {s['aspect']}: {ratio}{detail}{s['counts']}")
    lines += ["", result["statistics_note"], "", "## 原始草稿" if zh else "## Draft", "",
              result["draft_answer"] or ("（无草稿）" if zh else "(no draft)"),
              "", "## 逐句依据检查" if zh else "## Per-sentence evidence check", ""]
    for s in result["checks"]:
        lines.append(f"- {'⚠️' if s['flagged'] else '✅'} {s['text']} — {s['status']}{'：' if zh else ': '}{s['reason']}")
        for evidence in s["evidence"]:
            lines.append(f"  - [{evidence['chunk_id']}] {evidence['quote']}")
    path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")

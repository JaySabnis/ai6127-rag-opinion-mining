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
    rag = {"question": result["question"], "contexts": result["contexts"], "answer": result["answer"]}
    path.with_suffix(".rag.jsonl").write_text(json.dumps(rag, ensure_ascii=False) + "\n", encoding="utf-8")
    timing = {"query": result["question"], "n_chunks": len(result["chunks"]),
              **{k: v for k, v in result["timing"].items() if k != "total_s"}}
    path.with_suffix(".timings.jsonl").write_text(json.dumps(timing, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# 评论分析结果", "", result["answer"], "", "## 本次检索样本的好评比例", ""]
    for s in result["statistics"]:
        product = " ".join(str(s[k] or "") for k in ("brand", "model")).strip() or "未知机型"
        ratio = "无可用分母" if s["positive_ratio"] is None else f"{s['positive_ratio']:.1%}"
        lines.append(f"- {product} / {s['aspect']}: {ratio}，{s['n_relevant_reviews']}条相关独立评论；{s['counts']}")
    lines += ["", result["statistics_note"], "", "## 原始草稿", "", result["draft_answer"] or "（无草稿）",
              "", "## 逐句依据检查", ""]
    for s in result["checks"]:
        lines.append(f"- {'⚠️' if s['flagged'] else '✅'} {s['text']} — {s['status']}：{s['reason']}")
        for evidence in s["evidence"]:
            lines.append(f"  - [{evidence['chunk_id']}] {evidence['quote']}")
    path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")

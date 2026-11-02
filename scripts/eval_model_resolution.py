"""Experiment: how should the LLM's product names be matched to the corpus?

Compares four ways of turning a question into the model filter(s):

    parser        rule-based parser only, no LLM
    llm           DeepSeek rewrite (team prompt), then resolve_model() check   <- current design
    llm_full      same, plus the FULL corpus catalogue (~4,200 names) in the prompt
    llm_short     same, plus a SHORTLIST of ~20 catalogue names similar to the question

For every LLM method the final model is still passed through resolve_model(),
so the comparison is "what does adding a catalogue to the prompt buy on top
of the check". Reported per method:

    accuracy      final filter(s) equal the gold model(s)
    raw exact     LLM's own model string already exists in the corpus as written
    zero filter   LLM's own model string would match 0 rows without the check
    latency       median wall time of the LLM call (ms)
    prompt tok    median prompt tokens (and cache-hit tokens, from DeepSeek usage)

Only meta.pkl is loaded (no FAISS / torch), so this runs quickly.

    uv run python -m scripts.eval_model_resolution                      # all methods
    uv run python -m scripts.eval_model_resolution --methods parser llm llm_short
    uv run python -m scripts.eval_model_resolution --json results/model_resolution.json

Cost note: llm_full sends the whole catalogue with every question
(~27 questions x tens of thousands of tokens); check DeepSeek pricing first.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retrieval.config import IndexPaths  # noqa: E402
from retrieval.corpus import load_meta  # noqa: E402
from retrieval.query_parser import QueryParser, model_tokens  # noqa: E402
from retrieval.text import STOPWORDS  # noqa: E402


# ----------------------------------------------------------------------- cases
@dataclass
class Case:
    question: str
    gold: list[set[str]]  # one set of acceptable corpus spellings per product
    kind: str


# Gold names were checked against the corpus catalogue (exact spellings).
CASES = [
    Case("How is the screen on the Note 10+?", [{"Galaxy Note 10 Plus"}], "+ and missing family name"),
    Case("Is the Note 9 battery still good?", [{"Galaxy Note 9", "Galaxy Note9"}], "missing family name"),
    Case("三星 Note 10 的屏幕怎么样？", [{"Galaxy Note 10"}], "Chinese brand; 'Note 10' is also a Xiaomi model"),
    Case("三星 S10e 的相机怎么样？", [{"Galaxy S10e"}], "Chinese brand"),
    Case("苹果 11 的电池怎么样？", [{"iPhone 11"}], "Chinese brand, no 'iPhone'"),
    Case("iphone11promax battery drain", [{"iPhone 11 Pro Max"}], "no spaces"),
    Case("Is the 11 Pro Max camera worth it?", [{"iPhone 11 Pro Max"}], "missing family name"),
    Case("XR camera quality in low light", [{"iPhone XR"}], "one-word short name"),
    Case("Is the XS Max screen good?", [{"iPhone XS Max"}], "missing family name"),
    Case("S20 FE performance for gaming", [{"Galaxy S20 FE", "S20 FE"}], "missing family name"),
    Case("Galaxy S9+ camera", [{"Galaxy S9 Plus"}], "+ for Plus"),
    Case("Pixel 3 XL battery life", [{"Pixel 3 XL", "Pixel 3XL"}], "spacing variant"),
    Case("pixel 4a screen brightness", [{"Pixel 4a", "Pixel 4A"}], "case variant"),
    Case("Moto G Power 2020 battery", [{"Moto G Power (2020)"}], "year in brackets"),
    Case("Moto G7 Power battery life", [{"Moto G7 Power"}], "plain"),
    Case("Razr flip phone durability", [{"Motorola Razr", "Razr"}], "one-word name"),
    Case("OnePlus 7 Pro display", [{"Oneplus 7 Pro"}], "corpus spelling 'Oneplus'"),
    Case("LG G6 camera", [{"G6"}], "brand + short model"),
    Case("iPhone 12 mini battery", [{"iPhone 12 Mini"}], "case variant"),
    Case("Galaxy A71 performance", [{"Galaxy A71"}], "plain"),
    Case("Redmi Note 8 Pro camera", [{"Redmi Note 8 Pro"}], "plain"),
    Case("Galaxy S21 Ultra camera zoom", [{"Galaxy S21 Ultra"}], "plain"),
    Case("Which is better for photos, S10e or Pixel 3a?",
         [{"Galaxy S10e"}, {"Pixel 3a", "Pixel 3A"}], "comparison"),
    Case("iPhone 12 vs Galaxy S21 battery", [{"iPhone 12"}, {"Galaxy S21"}], "comparison"),
    Case("Are Samsung phones good for gaming?", [], "brand only (no model)"),
    Case("Which phone has the best battery?", [], "no product"),
    Case("Is the iPhone 99 camera good?", [], "model not in corpus"),
]


def is_correct(pred: list[str], gold: list[set[str]]) -> bool:
    pred = sorted(set(p for p in pred if p))
    if len(pred) != len(gold):
        return False
    remaining = list(gold)
    for p in pred:
        hit = next((g for g in remaining if p in g), None)
        if hit is None:
            return False
        remaining.remove(hit)
    return True


# ------------------------------------------------------------------- catalogue
def catalogue(parser: QueryParser) -> list[tuple[str, str]]:
    """(brand, model) for every model in the corpus, most frequent brand per model."""
    return sorted(((b or "", m) for m, b in parser.model_brand.items()), key=lambda x: (x[0].lower(), x[1].lower()))


_GENERIC = STOPWORDS | {"phone", "phones", "good", "best", "better", "which", "vs", "camera", "battery",
                        "screen", "display", "life", "quality", "performance", "worth", "still", "gaming",
                        "photos", "low", "light", "drain", "brightness", "zoom", "flip", "durability"}


def shortlist(question: str, cat: list[tuple[str, str]], counts: dict[str, int], k: int = 20) -> list[tuple[str, str]]:
    """Catalogue names sharing the most distinctive tokens with the question."""
    q = {t for t in model_tokens(question) if t not in _GENERIC}
    if not q:
        return []
    scored = []
    for brand, model in cat:
        toks = set(model_tokens(f"{brand} {model}"))
        overlap = q & toks
        if not overlap:
            continue
        mt = set(model_tokens(model))
        score = len(overlap & mt) / max(len(mt), 1) + 0.25 * len(overlap - mt)
        scored.append((score, math.log1p(counts.get(model, 0)), brand, model))
    scored.sort(reverse=True)
    return [(b, m) for _, _, b, m in scored[:k]]


def catalogue_block(names: list[tuple[str, str]], title: str) -> str:
    lines = "\n".join(f"{b} | {m}" for b, m in names) or "(no candidates)"
    return (f"\n\n{title}\nEach line is 'brand | model' exactly as written in the review corpus. "
            "When a product in the question matches a line, copy that brand and model EXACTLY into "
            "product. If no line matches, give your best normalized brand and model as usual.\n" + lines)


# --------------------------------------------------------------------- LLM call
def call_llm(system: str, question: str) -> tuple[dict, dict, float]:
    """Same request as retrieval/query_rewrite.py, but returns usage and latency too."""
    import httpx
    from dotenv import load_dotenv

    from retrieval.query_rewrite import validate_result

    load_dotenv(ROOT / ".env", override=False)
    key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not key:
        raise RuntimeError("Set DEEPSEEK_API_KEY in .env")
    base = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
    t = perf_counter()
    resp = httpx.post(
        f"{base}/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro"),
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": question}],
            "response_format": {"type": "json_object"},
            "thinking": {"type": "disabled"},
            "temperature": 0,
            "max_tokens": 1024,
        },
        timeout=120,
    )
    ms = (perf_counter() - t) * 1000
    resp.raise_for_status()
    body = resp.json()
    choice = body["choices"][0]
    if choice.get("finish_reason") != "stop":
        raise ValueError("incomplete response")
    return validate_result(json.loads(choice["message"]["content"])), body.get("usage", {}), ms


# ---------------------------------------------------------------------- methods
def run(method: str, case: Case, parser: QueryParser, cat, counts, base_prompt: str, llm=call_llm) -> dict:
    row = {"method": method, "question": case.question}
    if method == "parser":
        pred = [parser.detect_model(case.question)]
        row.update(pred=[p for p in pred if p], llm_models=None)
    else:
        prompt = base_prompt
        if method == "llm_full":
            prompt += catalogue_block(cat, "Full corpus catalogue:")
        elif method == "llm_short":
            prompt += catalogue_block(shortlist(case.question, cat, counts), "Candidate models from the corpus:")
        try:
            result, usage, ms = llm(prompt, case.question)
        except Exception as exc:  # report and keep going
            row.update(pred=[], error=f"{type(exc).__name__}: {exc}", correct=False)
            return row
        raw = [p.get("model") for p in result["product"]]
        pred = [parser.resolve_model(p.get("brand"), p.get("model")) for p in result["product"]]
        row.update(
            pred=[p for p in pred if p],
            llm_models=raw,
            raw_exact=[m in parser.model_brand for m in raw if m],
            ms=ms,
            prompt_tokens=usage.get("prompt_tokens"),
            cache_hit_tokens=usage.get("prompt_cache_hit_tokens"),
        )
    row["correct"] = is_correct(row["pred"], case.gold)
    return row


def summarize(rows: list[dict], method: str) -> dict:
    rs = [r for r in rows if r["method"] == method]
    out = {"method": method, "n": len(rs), "accuracy": sum(r["correct"] for r in rs) / len(rs),
           "errors": sum("error" in r for r in rs)}
    flags = [x for r in rs for x in (r.get("raw_exact") or [])]
    if flags:
        out["raw_exact"] = sum(flags) / len(flags)
        out["zero_filter"] = sum(not x for x in flags)
        out["n_llm_models"] = len(flags)
    for key in ("ms", "prompt_tokens", "cache_hit_tokens"):
        vals = [r[key] for r in rs if r.get(key) is not None]
        if vals:
            out[f"median_{key}"] = statistics.median(vals)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index-dir", default="indexes")
    ap.add_argument("--methods", nargs="+", default=["parser", "llm", "llm_short", "llm_full"],
                    choices=["parser", "llm", "llm_short", "llm_full"])
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    meta = load_meta(IndexPaths(args.index_dir).meta)
    parser = QueryParser(meta)
    counts = meta["model"].value_counts().to_dict()
    cat = catalogue(parser)
    base_prompt = ""
    if any(m != "parser" for m in args.methods):
        from retrieval.query_rewrite import PROMPT

        base_prompt = PROMPT
    print(f"catalogue: {len(cat):,} models; cases: {len(CASES)}; methods: {args.methods}\n")

    rows = []
    for case in CASES:
        print(f"- {case.question}   [{case.kind}]  gold={[sorted(g) for g in case.gold]}")
        for method in args.methods:
            r = run(method, case, parser, cat, counts, base_prompt)
            rows.append(r)
            mark = "ok " if r["correct"] else "ERR"
            extra = f"  llm said {r['llm_models']}" if r.get("llm_models") is not None else ""
            extra += f"  ({r['error']})" if r.get("error") else ""
            print(f"    {mark} {method:<10} -> {r['pred']}{extra}")

    print("\n" + "=" * 92)
    print(f"{'method':<11}{'accuracy':>10}{'raw exact':>11}{'0-row filters':>15}{'median ms':>11}"
          f"{'prompt tok':>12}{'cache hit':>11}{'errors':>8}")
    summaries = []
    for m in args.methods:
        s = summarize(rows, m)
        summaries.append(s)
        raw = f"{s['raw_exact']:.0%}" if "raw_exact" in s else "-"
        zero = f"{s['zero_filter']}/{s['n_llm_models']}" if "zero_filter" in s else "-"
        print(f"{m:<11}{s['accuracy']:>10.0%}{raw:>11}{zero:>15}"
              f"{s.get('median_ms', 0):>11.0f}{s.get('median_prompt_tokens', 0):>12.0f}"
              f"{s.get('median_cache_hit_tokens', 0):>11.0f}{s['errors']:>8}")
    print("\naccuracy = final filter(s) correct after resolve_model(); raw exact = LLM's own model string "
          "is a corpus name; 0-row filters = LLM names that would match nothing without the check")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"summary": summaries, "rows": rows}, ensure_ascii=False, indent=2,
                                        default=list), encoding="utf-8")
        print(f"saved {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

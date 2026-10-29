"""Group B -> C adapter. No retriever, embedding model or corpus download needed."""
from __future__ import annotations

from time import perf_counter
from generation.schema import ASPECTS, normalize_chunks
from generation.sentiment import SentimentClassifier, aggregate_sentiment
from generation.summary import GroundedSummarizer
from generation.verifier import FaithfulnessChecker


def render_sentences(sentences):
    return "\n".join(s["text"] + " " + " ".join(f"[{c}]" for c in s["citations"])
                     for s in sentences)


class OpinionPipeline:
    def __init__(self, client, *, max_chunks=64, max_context_chars=60000):
        self.client = client
        self.classifier = SentimentClassifier(client)
        self.summarizer = GroundedSummarizer(client)
        self.checker = FaithfulnessChecker(client)
        self.max_chunks, self.max_context_chars = max_chunks, max_context_chars

    def run(self, question, chunks, *, aspects=(), language="zh", verify=True):
        if not isinstance(question, str) or not question.strip():
            raise ValueError("A nonempty question is required")
        chunks = normalize_chunks(chunks)
        aspects = list(dict.fromkeys(aspects))
        if any(a not in ASPECTS for a in aspects):
            raise ValueError("Use the six corpus aspect names, or an empty list for overall")
        if len(chunks) > self.max_chunks or sum(len(r["text"]) for r in chunks) > self.max_context_chars:
            raise ValueError("Evidence exceeds the bounded context budget; reduce retrieval top_k")
        timings, started = {}, perf_counter()
        t = perf_counter()
        predictions = []
        for aspect in aspects or [None]:
            predictions.extend(self.classifier.classify(chunks, aspect))
        stats = aggregate_sentiment(predictions)
        timings["classification_s"] = perf_counter() - t
        t = perf_counter()
        sentences = self.summarizer.summarize(question, chunks, aspects=aspects, language=language)
        timings["generation_s"] = perf_counter() - t
        t = perf_counter()
        checker_error = None
        if verify:
            try:
                checks = self.checker.check(sentences, chunks)
            except RuntimeError as exc:
                # Retain the draft but do not accidentally approve it during an outage.
                checker_error = str(exc)
                checks = [dict(s, sentence_id=f"s{i}", status="unverifiable", flagged=True,
                               reason="核查服务失败，尚未验证", evidence=[])
                          for i, s in enumerate(sentences)]
        else:
            checks = [dict(s, sentence_id=f"s{i}", status="unverifiable", flagged=True,
                           reason="消融实验关闭了检查器", evidence=[])
                      for i, s in enumerate(sentences)]
        timings["verification_s"] = perf_counter() - t
        draft = render_sentences(sentences)
        if verify:
            supported = [s for s in checks if s["status"] == "supported"]
            answer = render_sentences(supported)
            if not answer:
                answer = "本次检索证据不足以形成通过核查的总结。" if language == "zh" else "No summary claims passed the evidence check."
        else:
            answer = draft or ("本次检索证据不足以形成总结。" if language == "zh" else "Insufficient retrieved evidence.")
        if not chunks:
            answer = "没有检索到评论，无法计算好评比例或生成总结。" if language == "zh" else "No reviews retrieved; no ratio or summary is available."
        timings["total_s"] = perf_counter() - started
        # Stage files intentionally omit total_s to avoid double-counting in
        # evaluation/summarize_latency.py, which sums all *_s fields.
        return {"question": question, "answer": answer, "draft_answer": draft,
                "sentences": sentences, "checks": checks,
                "flagged_sentences": [s for s in checks if s["flagged"]],
                "verification_enabled": verify, "checker_error": checker_error,
                "predictions": predictions, "statistics": stats, "chunks": chunks,
                "contexts": [r["text"] for r in chunks], "timing": timings,
                "api": self.client.telemetry(),
                "statistics_note": "本次检索样本的去重评论比例，不代表全体买家；缺失的完整评论片段可能改变判断。"}


def enrich_reply(reply, question, *, client=None, aspects=None, language="zh", verify=True):
    """Accept app.pipeline.search_full_corpus() output and preserve all its fields."""
    if client is None:
        from generation.client import DeepSeekClient
        client = DeepSeekClient()
    if aspects is None:
        aspects = (reply.get("rewrite") or {}).get("aspects", [])
    result = OpinionPipeline(client).run(question, reply.get("chunks", []),
                                        aspects=aspects, language=language, verify=verify)
    return {**reply, **{k: v for k, v in result.items() if k != "chunks"},
            "chunks": result["chunks"],
            "timing_ms": {**reply.get("timing_ms", {}),
                "classification": result["timing"]["classification_s"] * 1000,
                "generation": result["timing"]["generation_s"] * 1000,
                "verification": result["timing"]["verification_s"] * 1000,
                "total": reply.get("timing_ms", {}).get("total", 0) + result["timing"]["total_s"] * 1000}}

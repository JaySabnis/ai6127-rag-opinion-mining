import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.error

from generation.client import DeepSeekClient
from generation.io import read_chunks, write_predictions, write_run
from generation.pipeline import OpinionPipeline, enrich_reply
from generation.schema import normalize_chunks
from generation.sentiment import SentimentClassifier, aggregate_sentiment
from generation.verifier import FaithfulnessChecker, parse_summary


def chunk(cid="a_0", doc="a", text="The battery lasts all day.", model="TestPhone"):
    return {"chunk_id": cid, "doc_id": doc, "text": text, "brand": "Test", "model": model}


def pred(cid, doc, label, present=True, model="TestPhone"):
    return {"chunk_id": cid, "doc_id": doc, "label": label, "aspect_present": present,
            "brand": "Test", "model": model, "aspect": "battery"}


def classification(cid="a_0", label="positive", quote="lasts all day", present=True):
    return {"chunk_id": cid, "label": label, "aspect_present": present,
            "subjectivity": "neutral" if label == "neutral" else "opinionated",
            "evidence_quotes": [quote] if quote else [], "reason": "Text evidence"}


def completion(value, reason="stop"):
    return {"model": "test", "usage": {"prompt_tokens": 3, "completion_tokens": 2},
            "choices": [{"finish_reason": reason, "message": {"content": json.dumps(value)}}]}


class QueueClient:
    def __init__(self, values):
        self.values, self.payloads = iter(values), []

    def json(self, system, payload, *, validate, **kwargs):
        self.payloads.append(payload)
        value = next(self.values)
        if isinstance(value, Exception):
            raise value
        return validate(value)

    def telemetry(self):
        return {"api_calls": len(self.payloads), "cache_hits": 0}


class StatisticsTests(unittest.TestCase):
    def test_overlap_counts_once_and_opposing_chunks_become_mixed(self):
        rows = [pred("a_0", "a", "positive"), pred("a_1", "a", "positive"),
                pred("b_0", "b", "negative"), pred("b_1", "b", "positive"),
                pred("c_0", "c", "neutral"), pred("d_0", "d", "negative"),
                pred("e_0", "e", "neutral", False)]
        s = aggregate_sentiment(rows)[0]
        self.assertEqual(s["counts"], {"positive": 1, "negative": 1, "mixed": 1, "neutral": 1})
        self.assertEqual(s["n_relevant_reviews"], 4)
        self.assertEqual(s["n_excluded_aspect_absent"], 1)
        self.assertEqual(s["positive_ratio"], 0.25)
        self.assertEqual(s["polarized_positive_ratio"], 0.5)

    def test_neutral_only_and_no_relevant_evidence(self):
        s = aggregate_sentiment([pred("a", "a", "neutral")])[0]
        self.assertEqual(s["positive_ratio"], 0)
        self.assertIsNone(s["polarized_positive_ratio"])
        s = aggregate_sentiment([pred("a", "a", "neutral", False)])[0]
        self.assertIsNone(s["positive_ratio"])
        self.assertEqual(aggregate_sentiment([]), [])

    def test_products_never_share_a_percentage(self):
        s = aggregate_sentiment([pred("a", "a", "positive"), pred("b", "b", "negative", model="Other")])
        self.assertEqual(len(s), 2)
        self.assertEqual([r["positive_ratio"] for r in s], [1, 0])

    def test_conflicting_product_metadata_is_an_error(self):
        with self.assertRaises(ValueError):
            aggregate_sentiment([pred("a_0", "a", "positive"), pred("a_1", "a", "negative", model="Other")])

    def test_exact_duplicate_prediction_does_not_add_vote(self):
        p = pred("a", "a", "positive")
        self.assertEqual(aggregate_sentiment([p, p])[0]["n_relevant_reviews"], 1)

    def test_no_binary_ratio_when_only_mixed(self):
        self.assertIsNone(aggregate_sentiment([pred("a", "a", "mixed")])[0]["polarized_positive_ratio"])


class ClassificationTests(unittest.TestCase):
    def test_rating_and_gold_labels_are_not_sent_to_model(self):
        client = QueueClient([{"results": [classification()]}])
        rows = SentimentClassifier(client).classify([chunk() | {"rating": 1, "label": "negative"}], "battery")
        self.assertEqual(rows[0]["label"], "positive")
        self.assertNotIn("rating", client.payloads[0]["chunks"][0])
        self.assertNotIn("label", client.payloads[0]["chunks"][0])

    def test_aspect_absence_is_distinct_from_neutral(self):
        client = QueueClient([{"results": [classification(label="neutral", quote=None, present=False)]}])
        row = SentimentClassifier(client).classify([chunk()], "camera")[0]
        self.assertFalse(row["aspect_present"])

    def test_invented_quote_is_rejected(self):
        client = QueueClient([{"results": [classification(quote="satellite communication")]}])
        with self.assertRaises(ValueError):
            SentimentClassifier(client).classify([chunk()])

    def test_incomplete_predictions_do_not_silently_become_neutral(self):
        client = QueueClient([{"results": []}])
        with self.assertRaises(ValueError):
            SentimentClassifier(client).classify([chunk()])

    def test_inconsistent_stage_labels_are_rejected(self):
        value = classification() | {"subjectivity": "neutral"}
        with self.assertRaises(ValueError):
            SentimentClassifier(QueueClient([{"results": [value]}])).classify([chunk()])


class CheckerTests(unittest.TestCase):
    def test_missing_and_unknown_citations_flag_without_api(self):
        client = QueueClient([])
        result = FaithfulnessChecker(client).check([
            {"text": "Satellite phone.", "citations": []},
            {"text": "Satellite phone.", "citations": ["bogus"]}], [chunk()])
        self.assertTrue(all(s["flagged"] for s in result))
        self.assertEqual(client.payloads, [])

    def test_valid_id_but_unsupported_content_is_flagged(self):
        value = {"judgments": [{"sentence_id": "s0", "status": "unsupported",
            "reason": "No satellite evidence", "evidence": []}]}
        checked = FaithfulnessChecker(QueueClient([value])).check([
            {"text": "It supports satellite calls.", "citations": ["a_0"]}], [chunk()])
        self.assertTrue(checked[0]["flagged"])

    def test_judge_cannot_invent_a_supporting_quote(self):
        value = {"judgments": [{"sentence_id": "s0", "status": "supported", "reason": "Yes",
            "evidence": [{"chunk_id": "a_0", "quote": "It supports satellite calls"}]}]}
        with self.assertRaises(ValueError):
            FaithfulnessChecker(QueueClient([value])).check([
                {"text": "Satellite calls.", "citations": ["a_0"]}], [chunk()])

    def test_partial_support_is_always_flagged(self):
        value = {"judgments": [{"sentence_id": "s0", "status": "partially_supported", "reason": "Only battery",
            "evidence": [{"chunk_id": "a_0", "quote": "lasts all day"}]}]}
        result = FaithfulnessChecker(QueueClient([value])).check([
            {"text": "Good battery and satellite calls.", "citations": ["a_0"]}], [chunk()])
        self.assertTrue(result[0]["flagged"])

    def test_checker_must_cover_every_sentence(self):
        with self.assertRaises(ValueError):
            FaithfulnessChecker(QueueClient([{"judgments": []}])).check([
                {"text": "Good battery.", "citations": ["a_0"]}], [chunk()])

    def test_plain_summary_retains_uncited_claims(self):
        parsed = parse_summary("Good battery [a_0].\n有卫星通信。\nWorks [a_0][b_0]!")
        self.assertEqual(len(parsed), 3)
        self.assertEqual(parsed[1]["citations"], [])
        self.assertEqual(parsed[2]["citations"], ["a_0", "b_0"])


class IntegrationTests(unittest.TestCase):
    def client(self, checker=None):
        return QueueClient([
            {"results": [classification()]},
            {"sentences": [{"text": "A reviewer praises all-day battery life.", "citations": ["a_0"]}]},
            checker if checker is not None else {"judgments": [{"sentence_id": "s0", "status": "supported",
                "reason": "Review says all day", "evidence": [{"chunk_id": "a_0", "quote": "lasts all day"}]}]},
        ])

    def test_end_to_end_preserves_group_b_fields_and_writes_eval_exports(self):
        reply = {"chunks": [chunk()], "reranked": True, "rewrite": {"aspects": ["battery"]},
                 "timing_ms": {"total": 20}, "candidate_count": 50}
        result = enrich_reply(reply, "Battery?", client=self.client())
        self.assertIn("[a_0]", result["answer"])
        self.assertEqual(result["candidate_count"], 50)
        self.assertFalse(result["flagged_sentences"])
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / "run.json"
            write_run(p, result)
            rag = json.loads(p.with_suffix(".rag.jsonl").read_text())
            self.assertEqual(set(rag), {"question", "contexts", "answer"})
            timing = json.loads(p.with_suffix(".timings.jsonl").read_text())
            self.assertNotIn("total_s", timing)

    def test_checker_outage_keeps_draft_but_does_not_approve_it(self):
        result = OpinionPipeline(self.client(RuntimeError("HTTP 503"))).run("Battery?", [chunk()])
        self.assertTrue(result["draft_answer"])
        self.assertTrue(result["flagged_sentences"])
        self.assertEqual(result["checks"][0]["status"], "unverifiable")
        self.assertNotIn("[a_0]", result["answer"])

    def test_empty_retrieval_makes_no_calls_and_no_ratio(self):
        client = QueueClient([])
        result = OpinionPipeline(client).run("Battery?", [])
        self.assertEqual(result["statistics"], [])
        self.assertEqual(client.payloads, [])

    def test_no_check_ablation_exposes_unverified_draft(self):
        result = OpinionPipeline(self.client()).run("Battery?", [chunk()], verify=False)
        self.assertEqual(result["answer"], result["draft_answer"])
        self.assertFalse(result["verification_enabled"])

    def test_oversize_context_is_not_silently_truncated(self):
        with self.assertRaises(ValueError):
            OpinionPipeline(QueueClient([]), max_context_chars=2).run("Battery?", [chunk()])

    def test_duplicate_and_nested_group_b_candidates(self):
        rows = [{"chunk_id": "a_0", "doc_id": "a", "text": "Good battery",
                 "metadata": {"brand": "Test", "model": "TestPhone"}}]
        self.assertEqual(len(normalize_chunks(rows + rows)), 1)
        self.assertEqual(normalize_chunks(rows)[0]["model"], "TestPhone")
        with self.assertRaises(ValueError):
            normalize_chunks(rows + [rows[0] | {"text": "Different"}])

    def test_csv_predictions_match_evaluation_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "predictions.csv"
            write_predictions(path, [pred("a", "a", "positive")])
            self.assertEqual(path.read_text().splitlines(), ["chunk_id,label", "a,positive"])

    def test_existing_group_b_fixture_loads_directly(self):
        path = Path(__file__).resolve().parents[1] / "fixtures/rerank_candidates.json"
        self.assertEqual(len(read_chunks(path)), 6)


class ClientTests(unittest.TestCase):
    def test_invalid_json_schema_retries_then_fails_not_neutral(self):
        client = DeepSeekClient(transport=lambda body: completion({"bad": True}))
        def validate(value):
            if "good" not in value:
                raise ValueError("bad")
            return value
        with self.assertRaises(RuntimeError):
            client.json("JSON", {}, validate=validate)
        self.assertEqual(client.calls, 2)

    def test_truncated_response_retried(self):
        values = iter([completion({}, "length"), completion({"ok": True})])
        client = DeepSeekClient(transport=lambda body: next(values))
        self.assertTrue(client.json("JSON", {}, validate=lambda v: v)["ok"])
        self.assertEqual(client.calls, 2)

    def test_valid_response_cache_avoids_repeated_paid_call(self):
        with tempfile.TemporaryDirectory() as temp:
            client = DeepSeekClient(cache_dir=temp, transport=lambda body: completion({"ok": True}))
            first = client.json("JSON", {}, validate=lambda v: v)
            self.assertEqual(first, client.json("JSON", {}, validate=lambda v: v))
            self.assertEqual(client.calls, 1)
            self.assertEqual(client.cache_hits, 1)

    def test_call_budget_is_enforced(self):
        client = DeepSeekClient(max_calls=0, transport=lambda body: completion({}))
        with self.assertRaises(RuntimeError):
            client.json("JSON", {}, validate=lambda v: v)

    def test_auth_failure_not_retried_or_exposed(self):
        def transport(body):
            raise urllib.error.HTTPError("https://api.deepseek.com", 401, "secret detail", {}, None)
        client = DeepSeekClient(transport=transport)
        with self.assertRaisesRegex(RuntimeError, "HTTP 401") as error:
            client.json("JSON", {}, validate=lambda v: v)
        self.assertNotIn("secret detail", str(error.exception))
        self.assertEqual(client.calls, 1)


if __name__ == "__main__":
    unittest.main()

"""Frozen-input and resumability contracts for the checker-only experiment."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evaluation.report.run_paired_checker import run_paired, telemetry_delta


class RecordingClient:
    model = "test-model"
    base_url = "https://example.test"

    def __init__(self, fail=False, unsupported=False):
        self.calls = 0
        self.usage = []
        self.fail = fail
        self.unsupported = unsupported
        self.payloads = []

    def telemetry(self):
        return {"api_calls": self.calls, "cache_hits": 0, "model": self.model, "usage": self.usage}

    def json(self, prompt, payload, *, validate, **_kwargs):
        self.calls += 1
        self.payloads.append(copy.deepcopy(payload))
        if self.fail:
            raise RuntimeError("test service unavailable")
        self.usage.append({"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120,
                           "prompt_cache_hit_tokens": 30})
        judgments = [{"sentence_id": s["sentence_id"], "status": "unsupported" if self.unsupported else "supported",
                      "reason": "Test judgment", "evidence": [] if self.unsupported else
                      [{"chunk_id": s["sources"][0]["chunk_id"], "quote": s["sources"][0]["text"]}]}
                     for s in payload["sentences"]]
        return validate({"judgments": judgments})


class PairedCheckerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.inputs, self.output = self.root / "inputs", self.root / "output"
        self.inputs.mkdir()
        self.original_bytes = {}
        for index in (1, 2):
            qid = f"Q{index:03d}"
            chunk = {"chunk_id": f"c{index}", "doc_id": f"d{index}", "text": "Great camera."}
            value = {"question": f"Camera question {index}?", "chunks": [chunk],
                     "sentences": [{"text": "One reviewer praises the camera.", "citations": [chunk["chunk_id"]]}],
                     "predictions": [{"chunk_id": chunk["chunk_id"], "label": "positive"}],
                     "statistics": [{"positive": 1}], "verification_enabled": False,
                     "draft_answer": f"One reviewer praises the camera. [{chunk['chunk_id']}]"}
            raw = (json.dumps(value, indent=3) + "\n").encode()
            (self.inputs / (qid+".json")).write_bytes(raw)
            self.original_bytes[qid] = raw

    def tearDown(self):
        self.temp.cleanup()

    def pair(self, qid="Q001"):
        return json.loads((self.output / "pairs" / (qid+".json")).read_text())

    def test_only_checker_runs_and_off_new_share_frozen_inputs(self):
        client = RecordingClient()
        with patch("generation.sentiment.SentimentClassifier.classify", side_effect=AssertionError("classification must stay frozen")), \
             patch("generation.summary.GroundedSummarizer.summarize", side_effect=AssertionError("summary must stay frozen")):
            summary = run_paired(self.inputs, self.output, client)
        self.assertEqual(2, client.calls)
        self.assertEqual("completed", summary["status"])
        for qid, raw in self.original_bytes.items():
            self.assertEqual(raw, (self.inputs / (qid+".json")).read_bytes())
            self.assertEqual(raw, (self.output / "frozen_inputs" / (qid+".json")).read_bytes())
            pair = self.pair(qid)
            original = json.loads(raw)
            self.assertEqual(original["sentences"], pair["frozen"]["sentences"])
            self.assertEqual(original["chunks"], pair["frozen"]["chunks"])
            self.assertEqual(original["predictions"], pair["frozen"]["predictions"])
            self.assertEqual(original["draft_answer"], pair["off"]["answer"])
            self.assertEqual(pair["off"]["answer"], pair["new"]["answer"])

    def test_resume_is_idempotent_and_limit_can_expand(self):
        first = RecordingClient()
        summary = run_paired(self.inputs, self.output, first, limit=1)
        self.assertEqual("incomplete", summary["status"])
        self.assertTrue(summary["selected_queries_completed"])
        self.assertEqual(["Q002"], summary["pending_query_ids"])
        second = RecordingClient()
        summary = run_paired(self.inputs, self.output, second, resume=True)
        self.assertEqual(1, second.calls)
        self.assertEqual("completed", summary["status"])
        third = RecordingClient()
        run_paired(self.inputs, self.output, third, resume=True)
        self.assertEqual(0, third.calls)
        self.assertEqual(1, len(self.pair()["attempts"]))

    def test_existing_output_requires_resume(self):
        run_paired(self.inputs, self.output, RecordingClient(), limit=1)
        client = RecordingClient()
        with self.assertRaisesRegex(ValueError, "--resume"):
            run_paired(self.inputs, self.output, client)
        self.assertEqual(0, client.calls)

    def test_changed_source_or_code_refuses_resume_before_calling(self):
        run_paired(self.inputs, self.output, RecordingClient(), limit=1)
        path = self.inputs / "Q002.json"
        path.write_bytes(path.read_bytes()+b" ")
        client = RecordingClient()
        with self.assertRaisesRegex(ValueError, "Resume refused"):
            run_paired(self.inputs, self.output, client, resume=True)
        path.write_bytes(self.original_bytes["Q002"])
        with patch("evaluation.report.run_paired_checker.code_provenance", return_value={"source_sha256": {"changed": "hash"}}):
            with self.assertRaisesRegex(ValueError, "Resume refused"):
                run_paired(self.inputs, self.output, client, resume=True)
        self.assertEqual(0, client.calls)

    def test_tampered_frozen_copy_is_rejected(self):
        run_paired(self.inputs, self.output, RecordingClient(), limit=1)
        (self.output / "frozen_inputs/Q002.json").write_text("{}")
        client = RecordingClient()
        with self.assertRaisesRegex(ValueError, "Frozen input changed"):
            run_paired(self.inputs, self.output, client, resume=True)
        self.assertEqual(0, client.calls)

    def test_api_failure_is_recorded_and_resume_retries_without_losing_attempt(self):
        client = RecordingClient(fail=True)
        summary = run_paired(self.inputs, self.output, client, limit=1)
        self.assertFalse(summary["selected_queries_completed"])
        self.assertEqual("incomplete", summary["status"])
        pair = self.pair()
        self.assertEqual("failed", pair["status"])
        self.assertIsNone(pair["new"]["answer"])
        self.assertEqual(1, pair["attempts"][0]["api"]["requests_without_usage"])
        self.assertEqual("", (self.output / "rag_outputs/checker_new.jsonl").read_text())
        self.assertEqual("", (self.output / "rag_outputs/checker_off.jsonl").read_text())
        retried = RecordingClient()
        run_paired(self.inputs, self.output, retried, limit=1, resume=True)
        self.assertEqual(["failed", "completed"], [a["status"] for a in self.pair()["attempts"]])

    def test_abstention_is_empty_answer_not_a_claim_or_failure(self):
        result = run_paired(self.inputs, self.output, RecordingClient(unsupported=True), limit=1)
        pair = self.pair()
        self.assertEqual("completed", pair["status"])
        self.assertEqual("", pair["new"]["answer"])
        self.assertTrue(pair["new"]["abstained"])
        self.assertEqual(1, result["new_abstentions"])
        self.assertEqual(0, result["retained_sentences"])
        self.assertEqual(0.5, result["completed_pair_coverage"])
        self.assertEqual(0, result["new_answer_coverage_on_completed_pairs"])
        self.assertEqual(0, result["new_sentence_retention_ratio"])
        exported = json.loads((self.output / "rag_outputs/checker_new.jsonl").read_text())
        self.assertTrue(exported["abstained"])

    def test_all_inputs_are_validated_before_any_api_call(self):
        path = self.inputs / "Q002.json"
        value = json.loads(path.read_text())
        value["draft_answer"] = "A silently regenerated draft."
        path.write_text(json.dumps(value))
        client = RecordingClient()
        with self.assertRaisesRegex(ValueError, "draft does not match"):
            run_paired(self.inputs, self.output, client, limit=1)
        self.assertEqual(0, client.calls)
        self.assertFalse(self.output.exists())

    def test_paid_usage_delta_counts_tokens_and_documented_price_only(self):
        pricing = {"input_per_million": 1, "cached_input_per_million": 0.1, "output_per_million": 2,
                   "currency": "USD", "source_url": "https://example.test/pricing", "as_of": "test"}
        client = RecordingClient()
        result = run_paired(self.inputs, self.output, client, limit=1, pricing=pricing)
        api = self.pair()["attempts"][0]["api"]
        self.assertEqual(100, api["prompt_tokens"])
        self.assertEqual(30, api["cached_prompt_tokens"])
        self.assertAlmostEqual(0.000113, api["estimated_cost"])
        self.assertAlmostEqual(0.000113, result["estimated_cost_recorded_attempts"])
        self.assertIsNone(telemetry_delta({"usage": []}, client.telemetry())["estimated_cost"])


if __name__ == "__main__":
    unittest.main()

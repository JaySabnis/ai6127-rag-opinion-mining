"""Regression tests for evidence removal invalidating a whole-sentence verdict."""
import copy
import json
import tempfile
import unittest

from generation.client import DeepSeekClient
from generation.verifier import FaithfulnessChecker


CHUNKS = [
    {"chunk_id": "a_0", "doc_id": "a", "text": "The camera is excellent. Battery lasts one day."},
    {"chunk_id": "b_0", "doc_id": "b", "text": "The display is bright."},
]
SENTENCES = [
    {"text": "The camera is excellent and supports satellite calls.", "citations": ["a_0"]},
    {"text": "One reviewer calls the display bright.", "citations": ["b_0"]},
]


def judgment(sid="s0", status="supported", evidence=None):
    return {"sentence_id": sid, "status": status, "reason": "The cited source supports the claim.",
            "evidence": [{"chunk_id": "a_0", "quote": "The camera is excellent."}] if evidence is None else evidence}


class StubClient:
    def __init__(self, judgments):
        self.judgments = judgments
        self.calls = 0

    def json(self, _prompt, _payload, *, validate, **_kwargs):
        self.calls += 1
        return validate(copy.deepcopy({"judgments": self.judgments}))


class VerifierRegressionTests(unittest.TestCase):
    def test_one_true_quote_does_not_rescue_a_fabricated_clause(self):
        evidence = [{"chunk_id": "a_0", "quote": "The camera is excellent."},
                    {"chunk_id": "a_0", "quote": "It supports satellite calls."}]
        for status in ("supported", "partially_supported", "contradicted"):
            with self.subTest(status=status):
                check = FaithfulnessChecker(StubClient([judgment(status=status, evidence=evidence)]))
                result = check.check(SENTENCES[:1], CHUNKS)[0]
                self.assertEqual("unverifiable", result["status"])
                self.assertTrue(result["flagged"])
                self.assertEqual(evidence[:1], result["evidence"])
                self.assertIn("original_status=" + status, result["reason"])
                self.assertIn("evidence[1]", result["reason"])

    def test_invalid_evidence_does_not_remove_an_independent_good_sentence(self):
        bad = judgment(evidence=[{"chunk_id": "a_0", "quote": "invented"}])
        good = judgment("s1", evidence=[{"chunk_id": "b_0", "quote": "The display is bright."}])
        results = FaithfulnessChecker(StubClient([bad, good])).check(SENTENCES, CHUNKS)
        self.assertEqual(["unverifiable", "supported"], [r["status"] for r in results])
        self.assertFalse(results[1]["flagged"])

    def test_quote_from_existing_but_uncited_chunk_invalidates_verdict(self):
        evidence = [{"chunk_id": "a_0", "quote": "The camera is excellent."},
                    {"chunk_id": "b_0", "quote": "The display is bright."}]
        result = FaithfulnessChecker(StubClient([judgment(evidence=evidence)])).check(SENTENCES[:1], CHUNKS)[0]
        self.assertEqual("unverifiable", result["status"])
        self.assertIn("chunk_not_cited", result["reason"])

    def test_unknown_summary_citation_is_structural_failure_only(self):
        sentences = [{"text": "Bad citation.", "citations": ["absent"]}, SENTENCES[1]]
        good = judgment("s1", evidence=[{"chunk_id": "b_0", "quote": "The display is bright."}])
        client = StubClient([good])
        results = FaithfulnessChecker(client).check(sentences, CHUNKS)
        self.assertEqual(["unsupported", "supported"], [r["status"] for r in results])
        self.assertEqual(1, client.calls)

    def test_empty_type_invalid_or_malformed_evidence_stays_local(self):
        invalid_items = [None, "quote", {}, {"chunk_id": "a_0"},
                         {"chunk_id": [], "quote": "The camera is excellent."},
                         {"chunk_id": "a_0", "quote": None},
                         {"chunk_id": "a_0", "quote": 7},
                         {"chunk_id": "a_0", "quote": " \u00a0\u2003 "}]
        valid = {"chunk_id": "a_0", "quote": "The camera is excellent."}
        good = judgment("s1", evidence=[{"chunk_id": "b_0", "quote": "The display is bright."}])
        for invalid in invalid_items:
            with self.subTest(invalid=invalid):
                result = FaithfulnessChecker(StubClient([judgment(evidence=[valid, invalid]), good])).check(SENTENCES, CHUNKS)
                self.assertEqual(["unverifiable", "supported"], [r["status"] for r in result])

    def test_empty_or_non_list_evidence_cannot_confirm_a_claim(self):
        for evidence in ([], None, "text", {}, 1):
            with self.subTest(evidence=evidence):
                value = judgment()
                value["evidence"] = evidence
                result = FaithfulnessChecker(StubClient([value])).check(SENTENCES[:1], CHUNKS)[0]
                self.assertEqual("unverifiable", result["status"])
                self.assertEqual([], result["evidence"])

    def test_unicode_composition_and_spaces_remain_valid(self):
        chunks = [{"chunk_id": "a_0", "doc_id": "a", "text": "Cafe\u0301\u00a0camera\u2003is\nexcellent."}]
        quote = {"chunk_id": "a_0", "quote": "Café camera is excellent."}
        result = FaithfulnessChecker(StubClient([judgment(evidence=[quote])])).check(SENTENCES[:1], chunks)[0]
        self.assertEqual("supported", result["status"])
        self.assertFalse(result["flagged"])

    def test_api_failure_propagates_without_inventing_approval(self):
        class Down:
            def json(self, *_args, **_kwargs):
                raise RuntimeError("service unavailable")
        with self.assertRaisesRegex(RuntimeError, "service unavailable"):
            FaithfulnessChecker(Down()).check(SENTENCES, CHUNKS)

    def test_missing_or_duplicate_judgment_remains_a_response_failure(self):
        for values in ([judgment()], [judgment(), judgment()]):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    FaithfulnessChecker(StubClient(values)).check(SENTENCES, CHUNKS)

    def test_cached_sanitized_verdict_never_regains_supported_status(self):
        response = {"judgments": [judgment(evidence=[
            {"chunk_id": "a_0", "quote": "The camera is excellent."},
            {"chunk_id": "a_0", "quote": "satellite calls"}])]}
        requests = []
        def transport(body):
            requests.append(body)
            return {"model": "test", "usage": {}, "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(response)}}]}
        with tempfile.TemporaryDirectory() as cache:
            client = DeepSeekClient(model="test", cache_dir=cache, transport=transport)
            first = FaithfulnessChecker(client).check(SENTENCES[:1], CHUNKS)
            second = FaithfulnessChecker(client).check(SENTENCES[:1], CHUNKS)
            self.assertEqual(first, second)
            self.assertEqual("unverifiable", second[0]["status"])
            self.assertEqual(1, len(requests))
            self.assertEqual(1, client.cache_hits)

    def test_legacy_cache_with_deleted_fake_quote_is_not_reused(self):
        response = {"judgments": [judgment(evidence=[
            {"chunk_id": "a_0", "quote": "The camera is excellent."},
            {"chunk_id": "a_0", "quote": "satellite calls"}])]}
        requests = []
        def transport(body):
            requests.append(copy.deepcopy(body))
            return {"model": "test", "usage": {}, "choices": [{"finish_reason": "stop",
                "message": {"content": json.dumps(response)}}]}
        with tempfile.TemporaryDirectory() as cache:
            client = DeepSeekClient(model="test", cache_dir=cache, transport=transport)
            class Legacy:
                def json(self, system, payload, *, validate, **kwargs):
                    kwargs.pop("cache_namespace", None)
                    def old_sanitizer(value):
                        value["judgments"][0]["evidence"] = value["judgments"][0]["evidence"][:1]
                        return value
                    return client.json(system, payload, validate=old_sanitizer, **kwargs)
            old = FaithfulnessChecker(Legacy()).check(SENTENCES[:1], CHUNKS)
            self.assertEqual(old[0]["status"], "supported")
            corrected = FaithfulnessChecker(client).check(SENTENCES[:1], CHUNKS)
            self.assertEqual(corrected[0]["status"], "unverifiable")
            self.assertEqual(2, len(requests))
            self.assertEqual(requests[0], requests[1])  # Namespace does not alter inference.
            self.assertEqual(client.cache_hits, 0)
            self.assertEqual(corrected, FaithfulnessChecker(client).check(SENTENCES[:1], CHUNKS))
            self.assertEqual(client.cache_hits, 1)


if __name__ == "__main__":
    unittest.main()

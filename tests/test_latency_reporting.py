"""Offline regression tests; no model downloads, real indexes or API requests."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evaluation.summarize_latency import main as summarize_main, read_timings, summarize
from scripts.benchmark_pipeline import main as benchmark_main, prerequisites, read_queries, timing_row


class LatencyReportingTests(unittest.TestCase):
    def test_authoritative_total_is_not_added_again(self):
        row = {"scope": "downstream", "classification_s": 2, "generation_s": 3,
               "verification_s": 1, "total_s": 6.5}
        result = summarize([row])
        self.assertEqual(result["total_s"]["mean"], 6.5)
        self.assertEqual(result["total_basis"], "measured_total")
        self.assertEqual(row["total_s"], 6.5)

    def test_legacy_exports_derive_downstream_total_without_mutation(self):
        rows = [{"n_chunks": 10, "classification_s": 2, "generation_s": 3, "verification_s": 1}]
        result = summarize(rows)
        self.assertEqual(result["total_s"]["mean"], 6)
        self.assertEqual(result["timing_scope"], "downstream")
        self.assertEqual(result["chunks_classified_per_second"], 5)
        self.assertNotIn("total_s", rows[0])

    def test_startup_and_unknown_intervals_are_not_summed(self):
        result = summarize([{"classification_s": 2, "generation_s": 3,
                             "index_load_s": 100, "custom_overlap_s": 20}])
        self.assertEqual(result["total_s"]["mean"], 5)
        self.assertEqual(result["fields_excluded_from_derived_total"], ["custom_overlap_s", "index_load_s"])

    def test_end_to_end_requires_scope_metadata(self):
        row = {"retrieve_s": 1, "classification_s": 2, "generation_s": 3, "total_s": 7}
        self.assertEqual(summarize([row])["scope"], "unspecified")
        self.assertEqual(summarize([{**row, "scope": "end_to_end"}])["scope"], "end_to_end")
        self.assertEqual(summarize([{"total_s": 7}])["scope"], "unspecified")

    def test_scope_conflicts_and_mixed_runs_are_rejected(self):
        invalid = [
            [{"total_s": 1, "scope": "downstream", "timing_scope": "end_to_end"}],
            [{"retrieve_s": 1, "scope": "downstream"}],
            [{"classification_s": 1, "scope": "retrieval"}],
            [{"total_s": 1, "scope": []}],
            [{"classification_s": 1}, {"retrieve_s": 1}],
        ]
        for rows in invalid:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                summarize(rows)

    def test_first_query_is_not_a_cold_start(self):
        result = summarize([{"generation_s": 8}, {"generation_s": 2}])
        self.assertEqual(result["first_query_total_s"], 8)
        self.assertEqual(result["subsequent_query_total_mean_s"], 2)
        self.assertNotIn("cold_total_s", result)
        self.assertNotIn("warm_total_mean_s", result)
        self.assertFalse(result["concurrent_throughput_measured"])
        self.assertEqual(result["serial_queries_per_minute_estimate"], 12)
        self.assertEqual(result["queries_per_minute"], 12)

    def test_zero_duration_is_valid_but_has_no_throughput(self):
        result = summarize([{"total_s": 0}])
        self.assertEqual(result["total_s"]["p95"], 0)
        self.assertIsNone(result["queries_per_minute"])
        json.dumps(result, allow_nan=False)

    def test_invalid_times_and_chunk_counts_are_rejected(self):
        for value in (-1, "2", None, True, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                summarize([{"total_s": value}])
        for value in (-1, 1.5, "2", True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                summarize([{"total_s": 1, "n_chunks": value}])
        with self.assertRaisesRegex(ValueError, "summed stage durations"):
            summarize([{"classification_s": 1e308, "generation_s": 1e308}])

    def test_empty_or_unrecognized_data_is_rejected(self):
        for rows in ([], [{}], [None], [{"index_load_s": 2}], [{"retrieve_s": 2, "retrieval_s": 2}]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                summarize(rows)

    def test_missing_stage_counts_do_not_claim_complete_observations(self):
        result = summarize([{"classification_s": 2, "generation_s": 2}, {"generation_s": 3}])
        self.assertEqual(result["classification_s"]["n"], 1)
        self.assertEqual(result["generation_s"]["n"], 2)
        self.assertEqual(result["total_s"]["mean"], 3.5)

    def test_reader_and_cli_keep_legacy_inputs_usable(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "run.jsonl"
            output = Path(directory) / "nested/summary.json"
            source.write_text('\ufeff\n{"classification_s": 2, "generation_s": 3}\n', encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                summarize_main([str(source), "--output", str(output)])
            self.assertEqual(json.loads(output.read_text())["total_s"]["mean"], 5)
            source.write_text('{}\nnot json\n')
            with self.assertRaisesRegex(ValueError, "line 2"):
                read_timings(source)
            source.write_text('[]\n')
            with self.assertRaisesRegex(ValueError, "must be a JSON object"):
                read_timings(source)


class BenchmarkInputTests(unittest.TestCase):
    def test_query_file_requires_five_distinct_questions_and_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "queries.json"
            path.write_text(json.dumps([f"Question {i}" for i in range(5)]))
            self.assertEqual(len(read_queries(path)), 5)
            path.write_text(json.dumps([f"Question {i}" for i in range(4)]))
            with self.assertRaisesRegex(ValueError, "at least five"):
                read_queries(path)
            path.write_text(json.dumps([" same ", "SAME", "q2", "q3", "q4"]))
            with self.assertRaisesRegex(ValueError, "duplicates"):
                read_queries(path)
            path.write_text(json.dumps([{"query_id": "x", "query": f"Question {i}"} for i in range(5)]))
            with self.assertRaisesRegex(ValueError, "unique"):
                read_queries(path)

    def test_jsonl_accepts_question_alias(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "queries.jsonl"
            path.write_text("\n".join(json.dumps({"question": f"Q{i}"}) for i in range(5)))
            self.assertEqual(read_queries(path)[4], {"query_id": "q005", "query": "Q4"})

    def test_missing_dependencies_and_indexes_report_no_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("scripts.benchmark_pipeline.importlib.util.find_spec", return_value=None):
                errors = prerequisites(Path(directory))
            self.assertTrue(any("Missing Python packages" in error for error in errors))
            self.assertTrue(any("no fixture fallback" in error for error in errors))

    def test_outer_timing_does_not_add_nested_totals(self):
        searched = {"timing_ms": {"total": 9000, "rewrite": 1000, "retrieve": 2000, "rerank": 3000}}
        reply = {"chunks": [1, 2], "timing": {"total_s": 15, "classification_s": 4,
                                                   "generation_s": 5, "verification_s": 6}}
        row = timing_row({"query": "Q"}, searched, reply, 22, position=1)
        self.assertEqual(row["total_s"], 22)
        self.assertEqual(row["retrieve_s"], 2)
        self.assertEqual(row["n_chunks"], 2)
        self.assertEqual(summarize([row])["total_s"]["mean"], 22)
        self.assertEqual(summarize([row])["scope"], "end_to_end")

    def test_preflight_failure_never_creates_results_or_an_api_client(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "queries.json", Path(directory) / "new-results"
            source.write_text(json.dumps([f"Question {i}" for i in range(5)]))
            with patch("scripts.benchmark_pipeline.load_env"), \
                    patch("scripts.benchmark_pipeline.prerequisites", return_value=["Missing retrieval index files"]), \
                    patch("scripts.benchmark_pipeline.DeepSeekClient") as client, \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                benchmark_main(["--queries", str(source), "--output-dir", str(output), "--check-only"])
            client.assert_not_called()
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()

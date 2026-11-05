"""Offline evaluation contracts; no model downloads, credentials or API calls."""
import asyncio
from contextlib import redirect_stdout, redirect_stderr
import csv
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from evaluation import eval_rag as ev


class FakeMetric:
    def __init__(self, value=0.75, error=None):
        self.value, self.error, self.calls = value, error, []

    async def ascore(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(value=self.value)


class RagEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "checker_new.jsonl"
        self.output = self.root / "scores"
        self.env = self.root / "local.env"
        self.env.write_text("DEEPSEEK_API_KEY=test-secret-never-save\nDEEPSEEK_MODEL=test-model\n")
        self.cfg = ev.load_config(self.env, environ={})
        self.original = [
            {"query_id": "Q001", "question": "Camera?", "contexts": ["The camera is great."],
             "answer": "One reviewer likes the camera.", "abstained": False},
            {"query_id": "Q002", "question": "Battery?", "contexts": ["It lasts all day."],
             "answer": "", "abstained": True},
        ]
        self.write_rows(self.original)

    def tearDown(self):
        self.temp.cleanup()

    def write_rows(self, rows):
        self.source.write_text("".join(json.dumps(row)+"\n" for row in rows))

    def loaded(self, limit=None):
        rows, source = ev.read_rows(self.source, limit)
        return rows, ev.provenance(rows, source, self.cfg)

    def metrics(self, names, value=0.75):
        return {name: (FakeMetric(value), lambda row: {"response": row["answer"]}) for name in names}

    def run_cli(self, names=None, factory=None, extra=()):
        args = [str(self.source), "--env-file", str(self.env), "--output-dir", str(self.output), *extra]
        if names is not None:
            args += ["--metrics", *names]
        factory = factory or (lambda cfg, selected: self.metrics(selected))
        with patch.dict("os.environ", {}, clear=True), patch.object(ev, "build_metrics", side_effect=factory), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return ev.main(args)

    def test_deepseek_env_aliases_defaults_and_no_secret_provenance(self):
        self.assertEqual("https://api.deepseek.com", self.cfg["LLM_BASE_URL"])
        self.assertEqual("test-model", self.cfg["LLM_MODEL"])
        _, identity = self.loaded()
        text = json.dumps(identity)
        self.assertNotIn("test-secret-never-save", text)
        self.assertNotIn("API_KEY", text)
        self.assertEqual(["Q001", "Q002"], identity["row_ids"])
        self.assertEqual(ev.sha256(self.source.read_bytes()), identity["input_sha256"])

    def test_legacy_env_and_explicit_file_precedence(self):
        self.env.write_text('export LLM_API_KEY="legacy-secret"\nLLM_MODEL=legacy-model\nLLM_BASE_URL=https://judge.example/v1/\n')
        cfg = ev.load_config(self.env, environ={"DEEPSEEK_API_KEY": "ambient-secret", "DEEPSEEK_MODEL": "ambient"})
        self.assertEqual("legacy-secret", cfg["LLM_API_KEY"])
        self.assertEqual("legacy-model", cfg["LLM_MODEL"])
        self.assertEqual("https://judge.example/v1", cfg["LLM_BASE_URL"])
        cfg = ev.load_config(self.root / "local.env", environ={})
        self.assertNotIn("answer_relevancy", ev.available_metrics(cfg))

    def test_embedding_configuration_can_come_from_environment_or_file(self):
        env = {"EMBEDDING_PROVIDER": "huggingface", "EMBEDDING_MODEL": "BAAI/bge-small-en-v1.5"}
        cfg = ev.load_config(self.env, environ=env)
        self.assertEqual("huggingface", cfg["EMBEDDING_PROVIDER"])
        self.assertEqual("BAAI/bge-small-en-v1.5", cfg["EMBEDDING_MODEL"])
        self.assertEqual(set(ev.METRIC_NAMES), set(ev.available_metrics(cfg)))
        self.env.write_text(self.env.read_text()+"EMBEDDING_PROVIDER=huggingface\nEMBEDDING_MODEL=local-override\n")
        self.assertEqual("local-override", ev.load_config(self.env, environ=env)["EMBEDDING_MODEL"])

    def test_secret_bearing_endpoint_rejected(self):
        self.env.write_text("DEEPSEEK_API_KEY=test\nDEEPSEEK_BASE_URL=https://host.example?api_key=secret\n")
        with self.assertRaisesRegex(ValueError, "without credentials"):
            ev.load_config(self.env, environ={})

    def test_abstention_skips_only_answer_dependent_calls(self):
        rows, identity = self.loaded()
        metrics = self.metrics(ev.METRIC_NAMES)
        scored = asyncio.run(ev.score_all(rows, metrics))
        summary = ev.summarize(scored, "test", self.cfg, identity)
        for name in ev.ANSWER_DEPENDENT:
            self.assertEqual(1, len(metrics[name][0].calls))
            self.assertEqual("skipped_abstention", scored[1][name+"_status"])
            self.assertIsNone(scored[1][name])
            self.assertEqual(1, summary[name+"_valid_n"])
            self.assertEqual(0, summary[name+"_failed"])
            self.assertEqual(0.75, summary[name])
        self.assertEqual(2, len(metrics["context_relevance"][0].calls))
        self.assertEqual(2, summary["context_relevance_valid_n"])
        self.assertEqual(0.5, summary["answer_coverage"])

    def test_empty_legacy_answer_and_explicit_fallback_are_abstentions(self):
        rows = [dict(self.original[0], answer=" \n", abstained=False),
                dict(self.original[1], answer="No supported answer.", abstained=True)]
        self.write_rows(rows)
        loaded, identity = self.loaded()
        metrics = self.metrics(["faithfulness", "context_relevance"])
        scored = asyncio.run(ev.score_all(loaded, metrics))
        summary = ev.summarize(scored, "test", self.cfg, identity)
        self.assertEqual(0, len(metrics["faithfulness"][0].calls))
        self.assertIsNone(summary["faithfulness"])
        self.assertEqual(0, summary["faithfulness_failed"])
        self.assertEqual(2, summary["faithfulness_skipped_abstention"])
        self.assertEqual(0, summary["answer_coverage"])

    def test_failures_and_nonfinite_values_not_scored_or_leaked(self):
        rows, identity = self.loaded()
        for value in (None, float("nan"), float("inf"), True, "0.5"):
            with self.subTest(value=value):
                scored = asyncio.run(ev.score_all(rows, self.metrics(["faithfulness"], value)))
                summary = ev.summarize(scored, "test", self.cfg, identity)
                self.assertEqual(1, summary["faithfulness_failed"])
                self.assertEqual(0, summary["faithfulness_valid_n"])
                self.assertIsNone(summary["faithfulness"])
        metrics = {"faithfulness": (FakeMetric(error=RuntimeError("test-secret-never-save")), lambda row: {})}
        scored = asyncio.run(ev.score_all(rows, metrics))
        self.assertNotIn("test-secret-never-save", json.dumps(scored))
        self.assertIn("RuntimeError", scored[0]["faithfulness_error"])

    def test_partial_merge_preserves_other_numeric_metrics_and_clears_old_error(self):
        def failing(cfg, selected):
            return {name: (FakeMetric(error=RuntimeError("unavailable")), lambda r: {}) for name in selected}
        self.assertEqual(1, self.run_cli(["faithfulness"], factory=failing))
        self.assertEqual(0, self.run_cli(["faithfulness"]))
        self.assertEqual(0, self.run_cli(["context_relevance"]))
        summary = json.loads((self.output / "rag_checker_new.json").read_text())
        self.assertEqual(0.75, summary["faithfulness"])
        self.assertEqual(0.75, summary["context_relevance"])
        self.assertEqual(0, summary["faithfulness_failed"])
        self.assertNotIn("test-secret-never-save", json.dumps(summary))
        with (self.output / "rag_checker_new.csv").open(encoding="utf-8-sig") as file:
            scored = list(csv.DictReader(file))
        self.assertEqual("", scored[0]["faithfulness_error"])

    def test_changed_answer_same_question_refuses_before_initialization(self):
        self.run_cli(["faithfulness"])
        altered = [dict(self.original[0], answer="Different claim."), self.original[1]]
        self.write_rows(altered)
        with self.assertRaises(SystemExit), patch.object(ev, "build_metrics") as factory:
            with redirect_stderr(io.StringIO()):
                ev.main([str(self.source), "--env-file", str(self.env), "--output-dir", str(self.output),
                         "--metrics", "context_relevance"])
        factory.assert_not_called()

    def test_changed_row_order_or_judge_or_selection_refuses_merge(self):
        self.run_cli(["faithfulness"])
        rows, identity = self.loaded()
        stem = self.output / "rag_checker_new"
        for field, changed in (("row_ids", ["Q002", "Q001"]),
                               ("judge_config", {**identity["judge_config"], "LLM_MODEL": "different"}),
                               ("row_sha256", identity["row_sha256"][:1])):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "Partial merge refused"):
                ev.load_prior(stem, {**identity, field: changed})
        self.write_rows(list(reversed(self.original)))
        _, reordered = self.loaded()
        with self.assertRaisesRegex(ValueError, "Partial merge refused"):
            ev.load_prior(stem, reordered)

    def test_old_untracked_results_and_tampered_csv_refused(self):
        self.run_cli(["faithfulness"])
        rows, identity = self.loaded()
        stem = self.output / "rag_checker_new"
        path = Path(f"{stem}.csv")
        path.write_bytes(path.read_bytes()+b"\n")
        with self.assertRaisesRegex(ValueError, "CSV changed"):
            ev.load_prior(stem, identity)
        Path(f"{stem}.json").write_text('{"faithfulness": 1.0}')
        with self.assertRaisesRegex(ValueError, "Partial merge refused"):
            ev.load_prior(stem, identity)

    def test_same_input_and_ids_but_csv_reordered_refused(self):
        self.run_cli(["faithfulness"])
        _, identity = self.loaded()
        stem = self.output / "rag_checker_new"
        csv_path = Path(f"{stem}.csv")
        with csv_path.open(encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            fields, rows = reader.fieldnames, list(reader)
        with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(reversed(rows))
        path = Path(f"{stem}.json")
        saved = json.loads(path.read_text())
        saved["csv_sha256"] = ev.sha256(csv_path.read_bytes())
        path.write_text(json.dumps(saved))
        with self.assertRaisesRegex(ValueError, "ordering differs"):
            ev.load_prior(stem, identity)

    def test_invalid_inputs_rejected_even_beyond_trial_limit(self):
        for changed in (dict(self.original[1], contexts=[12]), dict(self.original[1], answer=None),
                        dict(self.original[1], abstained="true"), dict(self.original[1], status="failed"),
                        dict(self.original[1], query_id="Q001")):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                self.write_rows([self.original[0], changed])
                ev.read_rows(self.source, limit=1)

    def test_unavailable_embedding_metric_rejected_before_initialization(self):
        with self.assertRaises(SystemExit), patch.object(ev, "build_metrics") as factory:
            with redirect_stderr(io.StringIO()):
                ev.main([str(self.source), "--env-file", str(self.env), "--output-dir", str(self.output),
                         "--metrics", "answer_relevancy"])
        factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()

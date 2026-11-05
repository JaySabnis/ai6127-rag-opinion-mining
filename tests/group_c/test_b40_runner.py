"""Offline regression tests; no network, API client, or index is loaded."""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from evaluation.report import run_b40


class B40RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.query = {"query_id": "Q001", "query": "How is the battery?", "aspects": ["battery"]}
        self.queries = self.root / "queries.jsonl"
        self.queries.write_text(json.dumps(self.query) + "\n", encoding="utf-8")
        self.inputs = self.root / "inputs"
        for config in run_b40.VARIANTS.values():
            path = self.inputs / config / "Q001.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps([{"chunk_id": "c1", "doc_id": "d1", "text": "Great battery"}]),
                            encoding="utf-8")
        self.args = argparse.Namespace(output_dir=self.root / "fresh-run", env_file=None,
                                       variants=["B40_learned"])
        for target, value in (("QUERIES", self.queries), ("INPUTS", self.inputs)):
            patcher = patch.object(run_b40, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        provenance = patch.object(run_b40, "code_provenance", return_value={
            "git_head": "test-commit", "source_fingerprint": "same-code"})
        provenance.start()
        self.addCleanup(provenance.stop)

    def fake_success(self, cmd, **kwargs):
        self.assertIn("--no-cache", cmd)
        self.assertIn("--no-check", cmd)
        result = Path(cmd[cmd.index("--output") + 1])
        question = cmd[cmd.index("--question") + 1]
        result.write_text(json.dumps({"question": question, "verification_enabled": False,
                                      "api": {"model": "test-model"}}), encoding="utf-8")
        result.with_suffix(".rag.jsonl").write_text(json.dumps({
            "question": question, "contexts": ["Great battery"], "answer": "A positive review."}) + "\n",
            encoding="utf-8")
        result.with_suffix(".timings.jsonl").write_text(json.dumps({
            "query": question, "classify_s": 1.0, "summarize_s": 2.0}) + "\n", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    def manifest(self):
        return json.loads((self.args.output_dir / "run_manifest.json").read_text(encoding="utf-8"))

    def test_selected_run_records_sources_and_keeps_exports_inside_destination(self):
        env = self.root / "private.env"
        env.write_text("DEEPSEEK_API_KEY=not-a-real-key", encoding="utf-8")
        self.args.env_file = env
        with patch.object(run_b40.subprocess, "run", side_effect=self.fake_success) as child:
            self.assertEqual(run_b40.execute(self.args), 0)
        child.assert_called_once()
        command = child.call_args.args[0]
        self.assertEqual(command[command.index("--env-file") + 1], str(env.resolve()))
        manifest = self.manifest()
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["code"]["git_head"], "test-commit")
        source = self.inputs / "08_learned_v2" / "Q001.json"
        self.assertEqual(manifest["inputs"][0]["sha256"], hashlib.sha256(source.read_bytes()).hexdigest())
        self.assertNotIn("not-a-real-key", json.dumps(manifest))
        self.assertTrue((self.args.output_dir / "rag_outputs" / "B40_learned.jsonl").is_file())
        self.assertTrue((self.args.output_dir / "timings" / "B40_learned.jsonl").is_file())
        self.assertFalse((self.args.output_dir / "runs" / "B40_bm25").exists())

    def test_nonempty_destination_is_rejected_before_any_child_call(self):
        self.args.output_dir.mkdir()
        previous = self.args.output_dir / "historical.json"
        previous.write_text("original", encoding="utf-8")
        with patch.object(run_b40.subprocess, "run") as child:
            with self.assertRaisesRegex(ValueError, "new or empty"):
                run_b40.execute(self.args)
        child.assert_not_called()
        self.assertEqual(previous.read_text(), "original")

    def test_child_failure_returns_nonzero_and_does_not_export_partial_batch(self):
        def failed(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, 7, "", "private provider output")
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(run_b40.execute(self.args, run_command=failed), 1)
        manifest = self.manifest()
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(len(manifest["failures"]), 1)
        self.assertNotIn("private provider output", json.dumps(manifest))
        self.assertFalse((self.args.output_dir / "rag_outputs").exists())

    def test_missing_sidecar_is_failure_even_if_child_exits_zero(self):
        def incomplete(cmd, **kwargs):
            done = self.fake_success(cmd, **kwargs)
            Path(cmd[cmd.index("--output") + 1]).with_suffix(".timings.jsonl").unlink()
            return done
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(run_b40.execute(self.args, run_command=incomplete), 1)
        self.assertEqual(self.manifest()["status"], "failed")
        self.assertFalse((self.args.output_dir / "rag_outputs").exists())

    def test_cli_requires_destination_and_rejects_historical_checker_variants(self):
        self.assertEqual(set(run_b40.VARIANTS), {"B40_bm25", "B40_dense", "B40_learned"})
        for argv in ([], ["--output-dir", str(self.args.output_dir), "--variants", "B40_learned_checker"],
                     ["--output-dir", str(self.args.output_dir), "--variants", "B40_learned_checker_fixed"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                run_b40.main(argv)
            self.assertNotEqual(raised.exception.code, 0)

    def test_missing_later_input_fails_preflight_before_first_api_call(self):
        self.args.variants = ["B40_learned", "B40_dense"]
        (self.inputs / "02_dense" / "Q001.json").unlink()
        with patch.object(run_b40.subprocess, "run") as child:
            with self.assertRaises(OSError):
                run_b40.execute(self.args)
        child.assert_not_called()
        self.assertFalse(self.args.output_dir.exists())

    def test_input_changed_during_run_cannot_be_exported_as_reproducible(self):
        def changed(cmd, **kwargs):
            done = self.fake_success(cmd, **kwargs)
            source = Path(cmd[cmd.index("--input") + 1])
            source.write_text("[]", encoding="utf-8")
            return done
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(run_b40.execute(self.args, run_command=changed), 1)
        self.assertIn("Input changed", self.manifest()["failures"][0]["error"])
        self.assertFalse((self.args.output_dir / "rag_outputs").exists())


if __name__ == "__main__":
    unittest.main()

"""Historical predictions remain immutable when ratios are repaired."""
import json
from pathlib import Path
import tempfile
import unittest

from evaluation.report.recompute_saved_statistics import recompute

ROOT = Path(__file__).resolve().parents[2]


class RecomputeTests(unittest.TestCase):
    def test_corrected_ratio_and_immutable_source(self):
        original = (ROOT / "results/group_c/b40/B40_learned/Q012.json").read_bytes()
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp) / "source", Path(tmp) / "corrected"
            saved = source / "B40_learned/Q012.json"
            saved.parent.mkdir(parents=True)
            saved.write_bytes(original)
            summary = recompute(source, output)
            corrected = json.loads((output / "B40_learned/Q012.json").read_text())
            self.assertEqual(saved.read_bytes(), original)
            self.assertEqual(summary["n_numerical_changes"], 1)
            self.assertEqual(len(corrected["statistics"]), 1)
            self.assertEqual(corrected["statistics"][0]["positive_ratio"], 0.3)
            with self.assertRaisesRegex(ValueError, "empty"):
                recompute(source, output)

    def test_cannot_write_into_historical_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            with self.assertRaisesRegex(ValueError, "outside"):
                recompute(source, source / "corrected")


if __name__ == "__main__":
    unittest.main()

"""Guard against silent changes to the scored sample; never modify gold files."""
import unittest
from evaluation.eval_classifier import validated_labels


class EvaluationInputTests(unittest.TestCase):
    def test_duplicate_ids_are_not_silently_overwritten(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validated_labels([{"chunk_id": "a", "label": "positive"},
                              {"chunk_id": "a", "label": "negative"}], source="test")

    def test_unknown_labels_and_empty_input_fail(self):
        for rows in ([], [{"chunk_id": "a", "label": "unknown"}],
                     [{"chunk_id": "", "label": "neutral"}]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                validated_labels(rows, source="test")

    def test_whitespace_is_normalized(self):
        self.assertEqual(validated_labels([{"chunk_id": " a ", "label": " positive "}],
                                          source="test"), {"a": "positive"})


if __name__ == "__main__":
    unittest.main()

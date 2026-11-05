"""Product identity regressions, including the two audited saved runs.

Core cases use only the standard library. Catalogue/boundary cases additionally
run when the repository's retrieval dependencies are installed; no models load.
"""
import copy
import importlib.util
import json
from pathlib import Path
import unittest

from generation.schema import normalize_chunks
from generation.sentiment import SentimentClassifier, aggregate_sentiment
from retrieval.product_identity import make_product_id, variant_key

ROOT = Path(__file__).resolve().parents[2]


def prediction(doc, model, label="positive", *, chunk=None, brand="Samsung", **extra):
    return {"chunk_id": chunk or doc + "_0", "doc_id": doc, "brand": brand,
            "model": model, "aspect": "battery", "aspect_present": True,
            "label": label, **extra}


class ProductIdentityTests(unittest.TestCase):
    def test_saved_note9_run_reaggregates_three_of_ten(self):
        path = ROOT / "results/group_c/b40/B40_learned/Q012.json"
        saved = json.loads(path.read_text(encoding="utf-8"))
        before = copy.deepcopy(saved["predictions"])
        stats = aggregate_sentiment(saved["predictions"])
        self.assertEqual(saved["predictions"], before)
        self.assertEqual(len(stats), 1)
        self.assertEqual(stats[0]["counts"], {"positive": 3, "negative": 6, "mixed": 0, "neutral": 1})
        self.assertEqual(stats[0]["n_relevant_reviews"], 10)
        self.assertAlmostEqual(stats[0]["positive_ratio"], 3 / 10)
        self.assertEqual(set(stats[0]["model_aliases"]), {"Galaxy Note 9", "Galaxy Note9"})

    def test_saved_s20fe_run_reaggregates_three_of_seven(self):
        saved = json.loads((ROOT / "results/group_c/b40/B40_learned/Q030.json").read_text(encoding="utf-8"))
        stats = aggregate_sentiment(saved["predictions"])
        self.assertEqual(len(stats), 1)
        self.assertEqual(stats[0]["counts"], {"positive": 3, "negative": 4, "mixed": 0, "neutral": 0})
        self.assertEqual(stats[0]["n_relevant_reviews"], 7)
        self.assertEqual(stats[0]["n_excluded_aspect_absent"], 3)
        self.assertAlmostEqual(stats[0]["positive_ratio"], 3 / 7)

    def test_actual_model_suffixes_do_not_merge(self):
        models = ["Galaxy S10", "Galaxy S10+", "Galaxy S10 5G", "Galaxy S10e",
                  "Galaxy S20", "Galaxy S20 FE", "Galaxy S20 Ultra"]
        self.assertEqual(len(aggregate_sentiment([prediction(str(i), m) for i, m in enumerate(models)])), len(models))
        self.assertEqual(make_product_id("Samsung", "Galaxy S10+"), make_product_id("Samsung", "S10 Plus"))
        self.assertNotEqual(make_product_id("Motorola", "Moto G Power"), make_product_id("Motorola", "Moto G Power (2020)"))

    def test_brand_scope_and_unknown_aliases_are_conservative(self):
        self.assertNotEqual(make_product_id("Samsung", "Galaxy Note 10"), make_product_id("Xiaomi", "Note 10"))
        self.assertNotEqual(variant_key("Galaxy X1", "Other"), variant_key("X1", "Other"))
        self.assertNotEqual(make_product_id("Other", "AB12"), make_product_id("Other", "A B12"))

    def test_alias_chunks_of_one_review_still_count_once(self):
        rows = [prediction("same", "Galaxy Note9", chunk="same_0"),
                prediction("same", "Galaxy Note 9", "negative", chunk="same_1")]
        stats = aggregate_sentiment(rows)
        self.assertEqual(len(stats), 1)
        self.assertEqual(stats[0]["n_relevant_reviews"], 1)
        self.assertEqual(stats[0]["counts"]["mixed"], 1)

    def test_unknown_models_are_scoped_to_source_review(self):
        rows = [prediction("one", None), prediction("two", "", "negative"),
                prediction("one", None, chunk="one_1")]
        stats = aggregate_sentiment(rows)
        self.assertEqual(len(stats), 2)
        self.assertEqual([s["n_relevant_reviews"] for s in stats], [1, 1])
        self.assertNotEqual(stats[0]["product_id"], stats[1]["product_id"])

    def test_explicit_identity_is_preserved_and_takes_precedence(self):
        rows = [prediction("one", "Vendor code A", product_id="catalogue:42", canonical_model="Known phone"),
                prediction("two", "Another spelling", "negative", product_id="catalogue:42", canonical_model="Known phone")]
        stats = aggregate_sentiment(rows)
        self.assertEqual(len(stats), 1)
        self.assertEqual(stats[0]["product_id"], "catalogue:42")
        self.assertEqual(stats[0]["model"], "Known phone")
        self.assertEqual(stats[0]["positive_ratio"], .5)
        self.assertEqual(set(stats[0]["model_aliases"]), {"Vendor code A", "Another spelling"})

    def test_conflicting_identity_metadata_is_rejected(self):
        a = prediction("a", "Galaxy S10", product_id="catalogue:42", canonical_model="Galaxy S10")
        for other in [prediction("b", "Galaxy S10+", product_id="catalogue:42", canonical_model="Galaxy S10 Plus"),
                      prediction("b", "Galaxy S10", brand="Other", product_id="catalogue:42", canonical_model="Galaxy S10"),
                      prediction("a", "Galaxy S10", chunk="a_1", product_id="catalogue:99", canonical_model="Galaxy S10")]:
            with self.subTest(other=other), self.assertRaises(ValueError):
                aggregate_sentiment([a, other])

    def test_duplicate_chunk_cannot_change_explicit_id_or_canonical_model(self):
        row = {"chunk_id": "a_0", "doc_id": "a", "text": "Good battery.",
               "brand": "Samsung", "model": "Galaxy S10", "product_id": "catalogue:42",
               "canonical_model": "Galaxy S10"}
        for altered in [dict(row, product_id="catalogue:99"), dict(row, canonical_model="Galaxy S10 Plus")]:
            with self.subTest(altered=altered), self.assertRaises(ValueError):
                normalize_chunks([row, altered])
        self.assertEqual(len(normalize_chunks([row, row])), 1)

    def test_nested_input_and_classifier_keep_identity_and_raw_model(self):
        row = {"chunk_id": "a_0", "doc_id": "a", "text": "Great battery.",
               "metadata": {"brand": "Samsung", "model": "Galaxy Note9",
                            "canonical_model": "Galaxy Note 9", "product_id": "catalogue:note9"}}
        original = copy.deepcopy(row)
        class Client:
            def json(self, prompt, payload, validate, **kwargs):
                return validate({"results": [{"chunk_id": "a_0", "aspect_present": True,
                    "subjectivity": "opinionated", "label": "positive", "reason": "Praise",
                    "evidence_quotes": ["Great battery."]}]})
        result = SentimentClassifier(Client()).classify([row], "battery")[0]
        self.assertEqual(row, original)
        self.assertEqual(result["model"], "Galaxy Note9")
        self.assertEqual(result["canonical_model"], "Galaxy Note 9")
        self.assertEqual(result["product_id"], "catalogue:note9")


@unittest.skipUnless(importlib.util.find_spec("pandas"), "retrieval catalogue tests require pandas")
class CatalogueIdentityTests(unittest.TestCase):
    def test_catalogue_and_generation_share_alias_identity(self):
        import pandas as pd
        from retrieval.query_parser import QueryParser
        rows = [("Galaxy Note 9", "Samsung"), ("Galaxy Note 9", "Samsung"),
                ("Galaxy Note9", "Samsung"), ("Galaxy S20 FE", "Samsung"),
                ("S20 FE", "Samsung"), ("Galaxy S10", "Samsung"),
                ("Galaxy S10+", "Samsung"), ("Galaxy S10 5G", "Samsung")]
        parser = QueryParser(pd.DataFrame(rows, columns=["model", "brand"]))
        for model, brand in rows:
            self.assertEqual(make_product_id(brand, model), make_product_id(brand, parser.canonical_of(model)))
        self.assertEqual(parser.canonical_of("Galaxy Note9"), "Galaxy Note 9")
        self.assertNotEqual(parser.canonical_of("Galaxy S10"), parser.canonical_of("Galaxy S10+"))

    @unittest.skipUnless(all(importlib.util.find_spec(m) for m in ("scipy", "nltk")), "retriever boundary tests require scipy/nltk")
    def test_hit_exports_catalogue_identity_without_overwriting_source(self):
        import pandas as pd
        from retrieval.retriever import Retriever
        rows = []
        for i, model in enumerate(["Galaxy Note 9", "Galaxy Note 9", "Galaxy Note9"]):
            rows.append({"chunk_id": f"c{i}", "doc_id": f"d{i}", "text": "Great battery.",
                         "brand": "Samsung", "model": model, "rating": 5, "timestamp": None,
                         "aspect_tags": ["battery"], "helpful_votes": 0, "verified": True,
                         "is_renewed": False, "source_url": None})
        retriever = Retriever(pd.DataFrame(rows), None, None)
        hit = retriever._make_hit(1, 2, 1.0, {}).to_dict()
        self.assertEqual(hit["model"], "Galaxy Note9")
        self.assertEqual(hit["canonical_model"], "Galaxy Note 9")
        self.assertEqual(hit["product_id"], make_product_id("Samsung", "Galaxy Note 9"))
        self.assertEqual(normalize_chunks([hit])[0]["product_id"], hit["product_id"])


if __name__ == "__main__":
    unittest.main()

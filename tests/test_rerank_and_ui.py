"""Core checks without real API calls or model downloads."""

from copy import deepcopy
import math
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from retrieval.query_rewrite import validate_result
from retrieval.rerank import CrossEncoderReranker, RerankerConfig


class RetrievalTests(unittest.TestCase):
    def test_rerank_raw_text_aspect_bonus_and_original_data(self):
        rows = [
            {"chunk_id": "a", "doc_id": "1", "text": "Good battery and camera.",
             "metadata": {"brand": "Samsung", "model": "Galaxy S10",
                          "aspect_tags": ["battery", "camera", "camera"]}},
            {"chunk_id": "b", "doc_id": "2", "text": "Good battery.",
             "aspect_tags": ["battery"]},
            {"chunk_id": "c", "doc_id": "3", "text": "Good phone."},
        ]
        before = deepcopy(rows)
        model = Mock()
        model.predict.return_value = [0.0, 0.05, 0.1]
        reranker = CrossEncoderReranker()
        with patch.object(reranker, "_load_model", return_value=model) as load:
            result = reranker.rerank("Battery and camera?", rows,
                                    query_aspects=["battery", "camera"], top_k=2)
            self.assertEqual([row["chunk_id"] for row in result], ["a", "b"])
            self.assertAlmostEqual(result[0]["rerank_score"], 1.348516426)
            self.assertEqual(result[0]["aspect_match_count"], 2)
            self.assertEqual(result[0]["aspect_coverage"], 1)
            self.assertEqual(result[1]["aspect_coverage"], 0.5)
            self.assertAlmostEqual(result[1]["aspect_bonus"], 0.3 + 0.3 * result[1]["aspect_gate"])
            self.assertEqual(model.predict.call_args.args[0],
                             [("Battery and camera?", row["text"]) for row in rows])
            self.assertEqual(rows, before)
            self.assertEqual(reranker.rerank("Battery?", rows, top_k=0), [])
            reranker.rerank("Battery?", rows)
            load.assert_called_once()

    def test_gate_reduces_bonus_for_low_scores(self):
        rows = [{"chunk_id": str(i), "text": "Battery review.", "aspect_tags": ["battery"]}
                for i in range(4)]
        rows[3]["aspect_tags"] = []
        model = Mock()
        model.predict.return_value = [-12, -6, 0, 0]
        reranker = CrossEncoderReranker()
        with patch.object(reranker, "_load_model", return_value=model):
            result = {r["chunk_id"]: r for r in reranker.rerank(
                "Battery?", rows, query_aspects=["battery"])}
            self.assertAlmostEqual(result["0"]["aspect_bonus"], 0.601483574)
            self.assertEqual(result["1"]["aspect_gate"], 0.5)
            self.assertAlmostEqual(result["1"]["aspect_bonus"], 0.9)
            self.assertAlmostEqual(result["2"]["aspect_bonus"], 1.198516426)
            self.assertEqual(result["3"]["aspect_bonus"], 0)
            for row in result.values():
                self.assertEqual(row["cross_encoder_contribution"], 0.6 * row["cross_encoder_score"])
                self.assertEqual(row["rerank_score"], row["cross_encoder_contribution"] + row["aspect_bonus"])
            reranker.config.aspect_weight = 0
            self.assertTrue(all(r["aspect_bonus"] == 0 for r in reranker.rerank(
                "Battery?", rows, query_aspects=["battery"])))
        for temperature in (0, -1, float("nan")):
            with self.assertRaises(ValueError):
                CrossEncoderReranker(RerankerConfig(aspect_temperature=temperature)).rerank("Battery?", [])

    def test_fixed_aspect_reward_respects_coverage_and_zero_matches(self):
        rows = [
            {"chunk_id": "partial", "text": "A", "aspect_tags": ["display"]},
            {"chunk_id": "full_low", "text": "B", "aspect_tags": ["display", "battery"]},
            {"chunk_id": "full_high", "text": "C", "aspect_tags": ["display", "battery"]},
            {"chunk_id": "none", "text": "D", "aspect_tags": []},
        ]
        model = Mock()
        model.predict.return_value = [-1e300, -1e300, 1e300, 0]
        reranker = CrossEncoderReranker()
        with patch.object(reranker, "_load_model", return_value=model):
            result = {r["chunk_id"]: r for r in reranker.rerank(
                "Display and battery?", rows, query_aspects=["display", "battery"])}
            for cid, expected in {"partial": 0.3, "full_low": 0.6, "full_high": 1.2, "none": 0}.items():
                self.assertAlmostEqual(result[cid]["aspect_bonus"], expected)
                self.assertAlmostEqual(result[cid]["aspect_bonus"],
                                       result[cid]["aspect_fixed_bonus"] + result[cid]["aspect_gated_bonus"])
            self.assertTrue(all(r["aspect_bonus"] == 0 for r in reranker.rerank("query", rows)))

    def test_rrf_can_break_close_scores_but_not_large_semantic_gaps(self):
        rows = [
            {"chunk_id": "semantic", "text": "A", "rrf_normalized": 0.0},
            {"chunk_id": "consensus", "text": "B", "rrf_normalized": 0.04},
            {"chunk_id": "strong", "text": "C", "score": 999999},
        ]
        before = deepcopy(rows)
        model = Mock()
        model.predict.return_value = [1.0, 0.6, 3.0]
        reranker = CrossEncoderReranker()
        with patch.object(reranker, "_load_model", return_value=model):
            result = reranker.rerank("query", rows)
            self.assertEqual([r["chunk_id"] for r in result], ["strong", "consensus", "semantic"])
            self.assertAlmostEqual(result[1]["rrf_bonus"], 0.04)
            self.assertAlmostEqual(result[1]["rerank_score"], 0.6 * 0.25 * (1 + math.tanh(0.6)) + 0.04)
            self.assertEqual(result[0]["rrf_bonus"], 0)  # generic score is not RRF
            reranker.config.rrf_weight = 0
            self.assertEqual([r["chunk_id"] for r in reranker.rerank("query", rows)],
                             ["strong", "semantic", "consensus"])
        self.assertEqual(rows, before)

    def test_invalid_rrf_rejected_before_model_loading(self):
        for value in (-0.01, 1.01, float("nan"), float("inf"), True, "bad"):
            reranker = CrossEncoderReranker()
            with patch.object(reranker, "_load_model") as load:
                with self.assertRaises(ValueError):
                    reranker.rerank("query", [{"text": "x", "rrf_normalized": value}])
                load.assert_not_called()
        for weight in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                CrossEncoderReranker(RerankerConfig(rrf_weight=weight)).rerank("query", [])
            with self.assertRaises(ValueError):
                CrossEncoderReranker(RerankerConfig(cross_encoder_weight=weight)).rerank("query", [])

    def test_minilm_is_bounded_but_gate_uses_raw_score(self):
        raw = [-1e300, -6, -0.2, 0, 0.2, 6, 1e300]
        rows = [{"chunk_id": str(i), "text": "Review."} for i in range(len(raw))]
        model = Mock()
        model.predict.return_value = raw
        reranker = CrossEncoderReranker()
        with patch.object(reranker, "_load_model", return_value=model):
            result = reranker.rerank("query", rows)
        by_id = {r["chunk_id"]: r for r in result}
        for i, score in enumerate(raw):
            row = by_id[str(i)]
            self.assertEqual(row["cross_encoder_raw_score"], score)
            self.assertEqual(row["cross_encoder_score"], 0.25 * (1 + math.tanh(score)))
            self.assertTrue(0 <= row["cross_encoder_score"] <= 0.5)
            self.assertTrue(0 <= row["cross_encoder_contribution"] <= 0.3)
        self.assertEqual(by_id["1"]["aspect_gate"], 0.5)
        self.assertEqual([r["chunk_id"] for r in result], [str(i) for i in reversed(range(len(raw)))])

    def test_rewrite_schema_and_aspect_whitelist(self):
        result = {"query": "Battery life?", "product": [], "aspects": ["battery"]}
        self.assertEqual(validate_result(result), result)
        for change in ({"aspects": ["price"]}, {"query": ""}, {"product": "Samsung"}):
            with self.assertRaises(ValueError):
                validate_result(result | change)

    def test_pipeline_defaults_to_retrieval_boost_and_preserves_reranker_aspects(self):
        from app.pipeline import search_full_corpus

        for rewrite in (False, True):
            for weight in (0.0, 0.5):
                candidate = {"chunk_id": "a", "doc_id": "1", "text": "Battery lasts all day."}
                hit = Mock()
                hit.to_dict.return_value = candidate
                res = Mock(parsed=Mock(aspects=["battery"]), hits=[hit], filters={},
                           model_variants=None, n_candidates=1, search_text="battery")
                retriever = Mock()
                retriever.retrieve.return_value = res
                reranker = Mock()
                reranker.rerank.return_value = [candidate]
                kwargs = {} if weight == 0.5 else {"retrieval_aspect_weight": weight}
                reply = search_full_corpus(
                    "Battery?", retriever, top_k=10, rerank=True, reranker=reranker,
                    use_rewrite=rewrite, rewrite_fn=lambda q: {
                        "query": "Battery life?", "product": [], "aspects": ["battery"]},
                    **kwargs)
                self.assertEqual(retriever.retrieve.call_args.kwargs["aspect_weight"], weight)
                self.assertEqual(reranker.rerank.call_args.kwargs["query_aspects"], ["battery"])
                self.assertEqual(reply["retrieval"][0]["aspect_weight"], weight)

    def test_ui_rewrite_rerank_history_and_error(self):
        from streamlit.testing.v1 import AppTest

        rewrite = {"query": "Galaxy S10 battery life?", "product": [], "aspects": ["battery"]}
        scorer = Mock()
        scorer.rerank_with_details.return_value = Mock(candidates=[{
            "chunk_id": "a", "doc_id": "1", "text": "Good battery.",
            "rating": 2, "aspect_tags": ["battery", "service"],
            "rerank_score": 1.0, "aspect_bonus": 0.5,
        }], timing_ms={"model_load": 0, "rerank": 1})
        path = Path(__file__).resolve().parents[1] / "app/streamlit_app.py"
        with patch("retrieval.query_rewrite.rewrite_query", return_value=rewrite) as rewrite_call, \
                patch("retrieval.rerank.CrossEncoderReranker", return_value=scorer):
            app = AppTest.from_file(str(path)).run()
            self.assertEqual(len(app.expander), 0)
            app.chat_input[0].set_value("Battery?").run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(len(app.chat_message), 2)
            markup = "\n".join(item.value for item in app.markdown)
            self.assertIn("★★", markup)
            self.assertIn('class="aspect-tag">battery</span>', markup)
            self.assertFalse(any("Chunk a" in item.value for item in app.caption))
            self.assertEqual(scorer.rerank_with_details.call_args.args[0], rewrite["query"])
            self.assertEqual(scorer.rerank_with_details.call_args.kwargs["query_aspects"], ["battery"])
            app.select_slider(key="top_k").set_value(3).run()
            rewrite_call.assert_called_once()
            app.chat_input[0].set_value("Another battery question?").run()
            self.assertEqual(len(app.chat_message), 4)
            self.assertEqual(app.selectbox(key="evidence_turn").value, 3)
            app.selectbox(key="evidence_turn").set_value(1).run()
            self.assertEqual(rewrite_call.call_count, 2)
            self.assertEqual(len(app.expander), 1)
            rewrite_call.side_effect = RuntimeError("API unavailable")
            app.chat_input[0].set_value("Camera?").run()
            self.assertIn("API unavailable", app.error[0].value)
            app.button(key="clear_chat").click().run()
            self.assertEqual(app.session_state.messages, [])
            self.assertEqual(len(app.selectbox), 0)
            self.assertEqual(len(app.expander), 0)

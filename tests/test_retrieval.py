"""Core checks without real API calls or model downloads."""

from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from retrieval.query_rewrite import validate_result
from retrieval.rerank import CrossEncoderReranker, RerankerConfig


class RetrievalTests(unittest.TestCase):
    def test_rerank_product_context_aspect_bonus_and_original_data(self):
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
        model.predict.return_value = [0.0, 0.2, 0.4]
        reranker = CrossEncoderReranker()
        with patch.object(reranker, "_load_model", return_value=model) as load:
            result = reranker.rerank("Battery and camera?", rows,
                                    query_aspects=["battery", "camera"], top_k=2)
            self.assertEqual([row["chunk_id"] for row in result], ["a", "b"])
            self.assertAlmostEqual(result[0]["rerank_score"], 0.993307149)
            self.assertEqual(result[0]["aspect_match_count"], 2)
            self.assertAlmostEqual(result[1]["aspect_bonus"], 0.497256851)
            text = model.predict.call_args.args[0][0][1]
            self.assertEqual(text, "This is a review of Samsung Galaxy S10.\n\n" + rows[0]["text"])
            self.assertEqual(rows, before)
            self.assertEqual(reranker.rerank("Battery?", rows, top_k=0), [])
            reranker.rerank("Battery?", rows)
            load.assert_called_once()

    def test_gate_reduces_bonus_for_low_scores(self):
        rows = [{"chunk_id": str(i), "text": "Battery review.", "aspect_tags": ["battery"]}
                for i in range(4)]
        rows[3]["aspect_tags"] = []
        model = Mock()
        model.predict.return_value = [-11, -5, 0, 0]
        reranker = CrossEncoderReranker()
        with patch.object(reranker, "_load_model", return_value=model):
            result = {r["chunk_id"]: r for r in reranker.rerank(
                "Battery?", rows, query_aspects=["battery"])}
            self.assertAlmostEqual(result["0"]["aspect_bonus"], 0.001236312)
            self.assertEqual(result["1"]["aspect_gate"], 0.5)
            self.assertEqual(result["1"]["aspect_bonus"], 0.25)
            self.assertAlmostEqual(result["2"]["aspect_bonus"], 0.496653575)
            self.assertEqual(result["3"]["aspect_bonus"], 0)
            for row in result.values():
                self.assertEqual(row["rerank_score"], row["cross_encoder_score"] + row["aspect_bonus"])
            reranker.config.aspect_weight = 0
            self.assertTrue(all(r["aspect_bonus"] == 0 for r in reranker.rerank(
                "Battery?", rows, query_aspects=["battery"])))
        for temperature in (0, -1, float("nan")):
            with self.assertRaises(ValueError):
                CrossEncoderReranker(RerankerConfig(aspect_temperature=temperature)).rerank("Battery?", [])

    def test_rewrite_schema_and_aspect_whitelist(self):
        result = {"query": "Battery life?", "product": [], "aspects": ["battery"]}
        self.assertEqual(validate_result(result), result)
        for change in ({"aspects": ["price"]}, {"query": ""}, {"product": "Samsung"}):
            with self.assertRaises(ValueError):
                validate_result(result | change)

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

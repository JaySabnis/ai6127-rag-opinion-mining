"""Core checks without real API calls or model downloads."""

from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from retrieval.query_rewrite import validate_result
from retrieval.rerank import CrossEncoderReranker, RerankerConfig


class RetrievalTests(unittest.TestCase):
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

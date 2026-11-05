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
        import streamlit as st
        from streamlit.testing.v1 import AppTest

        # Resource caches outlive an individual AppTest. Never reuse a previous
        # test's mock, or let an existing real model/index leak into this test.
        st.cache_resource.clear()
        self.addCleanup(st.cache_resource.clear)
        rewrite = {"query": "Galaxy S10 battery life?", "product": [], "aspects": ["battery"]}
        scorer = Mock()
        first = {
            "chunk_id": "a", "doc_id": "1", "text": "Good battery.",
            "brand": "Samsung", "model": "Galaxy S10",
            "rating": 2, "aspect_tags": ["battery", "service"],
            "rerank_score": 1.0, "aspect_bonus": 0.5,
        }
        second = {**first, "chunk_id": "b", "doc_id": "2", "rating": 5,
                  "text": "Battery lasts two days."}
        scorer.rerank_with_details.side_effect = [
            Mock(candidates=[chunk], timing_ms={"model_load": 0, "rerank": 1})
            for chunk in (first, second)
        ]
        path = Path(__file__).resolve().parents[1] / "app/streamlit_app.py"
        with patch("retrieval.query_rewrite.rewrite_query", return_value=rewrite) as rewrite_call, \
                patch("retrieval.rerank.CrossEncoderReranker", return_value=scorer), \
                patch("retrieval.dense.pick_device", return_value="cpu"), \
                patch("retrieval.Retriever.load") as load_index, \
                patch("generation.client.DeepSeekClient") as client, \
                patch("generation.pipeline.enrich_reply") as enrich:
            app = AppTest.from_file(str(path)).run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual([item.label for item in app.expander], ["Advanced settings"])
            self.assertFalse(app.toggle(key="generate_answer").value)
            self.assertTrue(app.toggle(key="verify_answer").disabled)
            app.chat_input[0].set_value("Battery?").run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(len(app.chat_message), 2)
            self.assertEqual([item.label for item in app.expander],
                             ["Retrieval details", "[1] Samsung Galaxy S10 · ★★☆☆☆", "Advanced settings"])
            markup = "\n".join(item.value for item in app.markdown)
            self.assertIn('class="aspect-tag">battery</span>', markup)
            self.assertIn("Good battery.", [item.value for item in app.text])
            self.assertFalse(any("Chunk a" in item.value for item in app.caption))
            self.assertEqual(scorer.rerank_with_details.call_args.args[0], rewrite["query"])
            self.assertEqual(scorer.rerank_with_details.call_args.kwargs["query_aspects"], ["battery"])
            app.select_slider(key="top_k").set_value(3).run()
            rewrite_call.assert_called_once()
            scorer.rerank_with_details.assert_called_once()
            app.chat_input[0].set_value("Another battery question?").run()
            self.assertEqual(len(app.chat_message), 4)
            self.assertEqual(scorer.rerank_with_details.call_args.kwargs["top_k"], 3)
            self.assertEqual(app.selectbox(key="evidence_turn").value, 3)
            self.assertIn("Battery lasts two days.", [item.value for item in app.text])
            saved = deepcopy(app.session_state.messages)
            app.selectbox(key="evidence_turn").set_value(1).run()
            self.assertEqual(rewrite_call.call_count, 2)
            self.assertEqual(scorer.rerank_with_details.call_count, 2)
            self.assertEqual(app.session_state.messages, saved)
            self.assertIn("Good battery.", [item.value for item in app.text])
            self.assertNotIn("Battery lasts two days.", [item.value for item in app.text])
            rewrite_call.side_effect = RuntimeError("API unavailable")
            app.chat_input[0].set_value("Camera?").run()
            self.assertEqual(len(app.exception), 0)
            self.assertIn("API unavailable", app.error[0].value)
            self.assertEqual(len(app.chat_message), 6)
            self.assertEqual(rewrite_call.call_count, 3)
            self.assertEqual(scorer.rerank_with_details.call_count, 2)
            self.assertEqual(app.selectbox(key="evidence_turn").value, 1)
            load_index.assert_not_called()
            client.assert_not_called()
            enrich.assert_not_called()
            app.button(key="clear_chat").click().run()
            self.assertEqual(app.session_state.messages, [])
            self.assertEqual(len(app.selectbox), 0)
            self.assertEqual(len(app.chat_message), 0)
            self.assertEqual(len(app.error), 0)
            self.assertEqual([item.label for item in app.expander], ["Advanced settings"])

    def test_full_corpus_ui_group_c_switches_and_history(self):
        import streamlit as st
        from streamlit.testing.v1 import AppTest

        st.cache_resource.clear()
        self.addCleanup(st.cache_resource.clear)
        root = Path(__file__).resolve().parents[1]
        index_meta = root / "indexes/meta.pkl"
        real_exists = Path.exists
        candidate = {"chunk_id": "a", "doc_id": "1", "text": "Good battery.",
                     "brand": "Samsung", "model": "Galaxy S10", "rating": 5,
                     "aspect_tags": ["battery"], "rerank_score": 1.0}
        hit = Mock()
        hit.to_dict.side_effect = lambda: deepcopy(candidate)
        retrieval_result = Mock(parsed=Mock(aspects=["battery"]), hits=[hit], filters={},
                                model_variants=None, n_candidates=20, search_text="battery")
        retriever = Mock()
        retriever.retrieve.return_value = retrieval_result
        scorer = Mock()
        scorer.rerank.side_effect = lambda *args, **kwargs: [deepcopy(candidate)]
        rewrite = {"query": "Galaxy S10 battery life?", "product": [], "aspects": ["battery"]}

        def enriched(reply, question, **kwargs):
            # UI contract fixture only; pipeline/model correctness has separate
            # tests. Keep realistic evidence-check states when the switch is off.
            verified = kwargs["verify"]
            check = {"text": "Owners praise the battery.", "citations": ["a"],
                     "status": "supported" if verified else "unverifiable",
                     "flagged": not verified,
                     "reason": "Evidence supports the claim." if verified else "Checker disabled for ablation"}
            return {**reply, "answer": "Owners praise the battery. [a]", "checks": [check],
                    "flagged_sentences": [] if verified else [check],
                    "statistics_note": "Ratios cover this retrieved sample only.",
                    "statistics": [{"brand": "Samsung", "model": "Galaxy S10", "aspect": "battery",
                                    "positive_ratio": 0.75, "n_relevant_reviews": 4,
                                    "counts": {"positive": 3, "negative": 1, "mixed": 0, "neutral": 0}}]}

        # Only the index-presence guard is faked. The real search_full_corpus
        # adapter executes against injected retrieval/reranking dependencies.
        with patch.object(Path, "exists", autospec=True,
                          side_effect=lambda path: path == index_meta or real_exists(path)), \
                patch("retrieval.Retriever.load", return_value=retriever) as load_index, \
                patch("retrieval.rerank.CrossEncoderReranker", return_value=scorer), \
                patch("retrieval.dense.pick_device", return_value="cpu"), \
                patch("retrieval.query_rewrite.rewrite_query", return_value=rewrite) as rewrite_call, \
                patch("generation.client.DeepSeekClient") as client, \
                patch("generation.pipeline.enrich_reply", side_effect=enriched) as enrich:
            app = AppTest.from_file(str(root / "app/streamlit_app.py")).run()
            app.radio(key="source").set_value("Full corpus").run()
            self.assertEqual(len(app.exception), 0)
            self.assertFalse(app.chat_input[0].disabled)
            self.assertTrue(app.toggle(key="use_rewrite").value)
            load_index.assert_not_called()
            app.toggle(key="generate_answer").set_value(True).run()
            self.assertFalse(app.toggle(key="verify_answer").disabled)
            enrich.assert_not_called()

            app.chat_input[0].set_value("How is the Galaxy S10 battery?").run()
            self.assertEqual(len(app.exception), 0)
            load_index.assert_called_once_with(root / "indexes")
            rewrite_call.assert_called_once()
            retriever.retrieve.assert_called_once()
            scorer.rerank.assert_called_once()
            self.assertEqual(scorer.rerank.call_args.args[0], rewrite["query"])
            self.assertEqual(enrich.call_args.kwargs["language"], "en")
            self.assertTrue(enrich.call_args.kwargs["verify"])
            self.assertIs(enrich.call_args.kwargs["client"], client.return_value)
            self.assertEqual(client.call_args.kwargs["cache_dir"], root / ".cache/group_c")
            markup = "\n".join(item.value for item in app.markdown)
            self.assertIn("Owners praise the battery. [1]", markup)
            self.assertIn("75% positive", markup)
            self.assertIn("Summary evidence check · 0 flagged", [item.label for item in app.expander])
            self.assertEqual(app.session_state.messages[1]["content"]["source"], "Full corpus (hybrid retrieval)")

            saved = deepcopy(app.session_state.messages)
            app.toggle(key="verify_answer").set_value(False).run()
            app.toggle(key="rerank").set_value(False).run()
            app.toggle(key="use_rewrite").set_value(False).run()
            self.assertEqual(app.session_state.messages, saved)
            enrich.assert_called_once()
            retriever.retrieve.assert_called_once()
            app.chat_input[0].set_value("Galaxy S10 的电池怎么样？").run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(enrich.call_count, 2)
            self.assertFalse(enrich.call_args.kwargs["verify"])
            self.assertEqual(enrich.call_args.kwargs["language"], "zh")
            self.assertFalse(enrich.call_args.args[0]["reranked"])
            self.assertEqual(retriever.retrieve.call_args.args[0], "Galaxy S10 的电池怎么样？")
            rewrite_call.assert_called_once()
            scorer.rerank.assert_called_once()
            self.assertIn("Summary evidence check · 1 flagged", [item.label for item in app.expander])

            app.toggle(key="generate_answer").set_value(False).run()
            self.assertTrue(app.toggle(key="verify_answer").disabled)
            app.chat_input[0].set_value("How is the camera?").run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(retriever.retrieve.call_count, 3)
            self.assertEqual(enrich.call_count, 2)
            self.assertEqual(client.call_count, 2)
            self.assertNotIn("answer", app.session_state.messages[-1]["content"])
            app.selectbox(key="evidence_turn").set_value(1).run()
            self.assertEqual(app.selectbox(key="evidence_turn").value, 1)
            self.assertEqual(retriever.retrieve.call_count, 3)
            self.assertEqual(enrich.call_count, 2)

            app.toggle(key="generate_answer").set_value(True).run()
            enrich.side_effect = RuntimeError("Summary service unavailable")
            app.chat_input[0].set_value("How about battery charging?").run()
            self.assertEqual(len(app.exception), 0)
            self.assertIn("Summary service unavailable", app.error[0].value)
            self.assertIn("error", app.session_state.messages[-1]["content"])
            self.assertEqual(app.selectbox(key="evidence_turn").value, 1)
            load_index.assert_called_once()

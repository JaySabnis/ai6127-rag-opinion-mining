import unittest
from unittest.mock import Mock
from copy import deepcopy
import numpy as np
from retrieval.rerank import CrossEncoderReranker, RerankerConfig
from retrieval.learned_rank import features

class LearnedRankTests(unittest.TestCase):
    def test_labels_and_ids_do_not_affect_rank_features(self):
        rows=[{'text':'Screen is bright and sharp.', 'aspect_tags':['display']},
              {'text':'Good phone.'}]
        a=features('How is the screen?',rows,[-2,-3])
        altered=[dict(r,chunk_id=str(i),relevance=3-i,query_id='anything') for i,r in enumerate(rows)]
        np.testing.assert_array_equal(a,features('How is the screen?',altered,[-2,-3]))

    def test_head_and_tail_preserve_inputs_and_have_finite_scores(self):
        rows=[{'chunk_id':str(i),'text':'Screen is bright.' if i%2 else 'Good phone.',
               'aspect_tags':['display'] if i%2 else []} for i in range(24)]
        before=deepcopy(rows)
        ranker=CrossEncoderReranker();ranker._model=Mock()
        ranker._model.predict.return_value=[-i*.3 for i in range(20)]
        result=ranker.rerank('How is the screen?',rows,top_k=24)
        self.assertEqual(rows,before)
        self.assertEqual([r['chunk_id'] for r in result[20:]],['20','21','22','23'])
        self.assertTrue(all(np.isfinite(r['rerank_score']) for r in result))
        self.assertTrue(all(result[i]['rerank_score']>=result[i+1]['rerank_score'] for i in range(23)))
        self.assertEqual(result[0]['rerank_strategy'],'learned')
        self.assertEqual(len(ranker._model.predict.call_args.args[0]),20)

    def test_single_candidate_and_top_zero(self):
        ranker=CrossEncoderReranker();ranker._model=Mock();ranker._model.predict.return_value=[-3]
        self.assertEqual(ranker.rerank('battery',[{'text':'Lasts a day'}])[0]['rerank_score'],0)
        self.assertEqual(ranker.rerank('battery',[{'text':'Lasts a day'}],top_k=0),[])

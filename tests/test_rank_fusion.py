import unittest
from unittest.mock import Mock
from copy import deepcopy
from retrieval.rerank import CrossEncoderReranker, RerankerConfig

class RankFusionTests(unittest.TestCase):
    def test_stripped_query_window_tail_and_input_immutability(self):
        rows = [dict(chunk_id=str(i), text='Review', model='Mate 20', brand='Huawei') for i in range(5)]
        original = deepcopy(rows)
        r = CrossEncoderReranker(RerankerConfig(strategy="rank_fusion", rerank_window=3))
        r._model = Mock()
        r._model.predict.return_value = [-10., -8., -6.]
        result = r.rerank('How is the display of mate20', rows)
        self.assertEqual([x['chunk_id'] for x in result], ['2','1','0','3','4'])
        self.assertEqual(r._model.predict.call_args.args[0][0][0], 'How is the display of')
        self.assertEqual(rows, original)
        self.assertEqual([x['final_rank'] for x in result], [1,2,3,4,5])
        self.assertEqual(result[0]['aspect_bonus'],0)

    def test_zero_weight_preserves_retrieval_and_ties_are_stable(self):
        r=CrossEncoderReranker(RerankerConfig(strategy="rank_fusion", semantic_rank_weight=0))
        r._model=Mock()
        r._model.predict.return_value=[0.,0.,1.]
        rows=[{'text':'x','chunk_id':str(i)} for i in range(3)]
        self.assertEqual([x['chunk_id'] for x in r.rerank('q',rows)],['0','1','2'])
        self.assertEqual(r.rerank('q',rows,top_k=0),[])

if __name__=='__main__': unittest.main()

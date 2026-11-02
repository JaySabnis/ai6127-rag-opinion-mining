"""Frozen semantic caches must reject changed inputs rather than reuse wrong scores."""
import hashlib
import unittest
from scripts.run_retrieval_ablation import CachedLogits
from retrieval.rerank import DEFAULT_REVISION

class SemanticCacheTests(unittest.TestCase):
    def setUp(self):
        self.rows=[{'chunk_id':'one','text':'Bright screen.'}]
        self.cache={'chunk_ids':['one'],'query':'screen','logits':[-2.],
            'text_sha256':[hashlib.sha256(b'Bright screen.').hexdigest()],
            'model_revision':DEFAULT_REVISION}

    def test_changed_candidate_and_text_are_rejected(self):
        for rows in ([{'chunk_id':'other','text':'Bright screen.'}],
                     [{'chunk_id':'one','text':'Battery lasts a day.'}]):
            with self.assertRaises(ValueError):CachedLogits(self.cache,rows)

    def test_changed_query_and_revision_are_rejected(self):
        cache=CachedLogits(self.cache,self.rows)
        with self.assertRaises(ValueError):cache.predict([('battery','Bright screen.')])
        with self.assertRaises(ValueError):CachedLogits({**self.cache,'model_revision':'changed'},self.rows)
        self.assertEqual(cache.predict([('screen','Bright screen.')]),[-2.])

if __name__=='__main__':unittest.main()

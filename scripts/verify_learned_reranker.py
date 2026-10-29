"""Verify deployed implementation against trained development model and export run."""
import json
from unittest.mock import Mock
import numpy as np
from scripts.design_reranker import ROOT, metrics
from retrieval.learned_rank import predict
from retrieval.rerank import CrossEncoderReranker, RerankerConfig

def main():
    out=ROOT/'results/reranker_learned'
    data=json.loads((ROOT/'results/reranker_design/scores.json').read_text(encoding='utf8'))
    model=json.loads((ROOT/'retrieval/models/review_ranker_v1.json').read_text())
    labels={}
    for s in (ROOT/'evaluation/retrieval_ablation_40/qrels.jsonl').read_text().splitlines():
        r=json.loads(s); labels.setdefault(r['query_id'],{})[r['doc_id']]=r['relevance']
    ranker=CrossEncoderReranker(RerankerConfig(learned_model_path=str(ROOT/'retrieval/models/review_ranker_v1.json')));ranker._model=Mock()
    values=[];lines=[]
    for qid,d in data.items():
        ranker._model.predict.return_value=d['scores']
        result=ranker.rerank(d['query'],d['rows'],scoring_query=d['query'])
        scores=predict(d['query'],d['rows'],d['scores'],model)
        expected=sorted(range(len(scores)),key=lambda i:scores[i],reverse=True)
        assert [r['chunk_id'] for r in result]==[d['rows'][i]['chunk_id'] for i in expected]
        values.append(metrics([r['chunk_id'] for r in result],labels[qid]))
        lines.extend(f"{qid} Q0 {r['chunk_id']} {i} {21-i} learned" for i,r in enumerate(result,1))
    (out/'run.trec').write_text('\n'.join(lines)+'\n',encoding='utf8')
    (out/'production_check.json').write_text(json.dumps({'queries':len(data),'metrics':['P@10','MRR@10','nDCG@10'],'development_training_fit':np.mean(values,axis=0).tolist(),'model_consistency':True},indent=2)+'\n',encoding='utf8')
    print('All 40 production rankings match saved model; mean:',np.mean(values,axis=0))

if __name__=='__main__':main()

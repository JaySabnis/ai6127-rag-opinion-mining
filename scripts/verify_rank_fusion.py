"""Verify production ranking on all development queries with cached model logits."""
import csv
import json
from unittest.mock import Mock
from scripts.design_reranker import ROOT, OUT, metrics, order
from retrieval.rerank import CrossEncoderReranker, RerankerConfig

def main():
    data=json.loads((OUT/'scores.json').read_text(encoding='utf8'))
    labels={}
    for s in (ROOT/'evaluation/retrieval_ablation_40/qrels.jsonl').read_text().splitlines():
        r=json.loads(s); labels.setdefault(r['query_id'],{})[r['doc_id']]=r['relevance']
    rows=[]; runs=[]
    model=CrossEncoderReranker(RerankerConfig(strategy="rank_fusion"))
    model._model=Mock()
    for qid, d in data.items():
        model._model.predict.return_value=d['scores']
        ranked=model.rerank(d['query'],d['rows'],scoring_query=d['query'])
        ids=[r['chunk_id'] for r in ranked]
        assert ids==[d['rows'][i]['chunk_id'] for i in order(d['rows'],d['scores'],.8,10)]
        assert all(cid in labels[qid] for cid in ids)
        rows.append([qid,*metrics(ids,labels[qid])])
        runs.extend(f'{qid} Q0 {cid} {i} {21-i} rank_fusion' for i,cid in enumerate(ids,1))
    with (OUT/'per_query.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.writer(f);w.writerow(['query_id','P@10','MRR@10','nDCG@10']);w.writerows(rows)
    (OUT/'run.trec').write_text('\n'.join(runs)+'\n',encoding='utf8')
    print('40 production rankings match development implementation; all Top-20 judged.')

if __name__=='__main__': main()

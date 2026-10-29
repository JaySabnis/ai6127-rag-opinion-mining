"""Offline development: fixed judged Top-20, five-fold selection, no API calls."""
import json
import os
from pathlib import Path
import numpy as np
import torch
from retrieval.rerank import CrossEncoderReranker, RerankerConfig

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/reranker_design'

def metrics(ids, labels):
    grades = [labels[x] for x in ids[:10]]
    ideal = sorted(labels.values(), reverse=True)[:10]
    dcg = lambda a: sum(g / np.log2(i+2) for i,g in enumerate(a))
    return [sum(g>=2 for g in grades)/10,
            next((1/(i+1) for i,g in enumerate(grades) if g>=2),0),
            dcg(grades)/dcg(ideal)]

def order(rows, scores, weight, offset):
    semantic = sorted(range(len(rows)), key=lambda i: scores[i], reverse=True)
    ranks = {i:r+1 for r,i in enumerate(semantic)}
    return sorted(range(len(rows)), key=lambda i: (1-weight)/(offset+i+1)+weight/(offset+ranks[i]), reverse=True)

def main():
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['TRANSFORMERS_OFFLINE']='1'
    torch.set_num_threads(8)
    OUT.mkdir(parents=True,exist_ok=True)
    cache = OUT/'scores.json'
    queries = [json.loads(s) for s in (ROOT/'evaluation/retrieval_ablation_40/queries.jsonl').read_text(encoding='utf8').splitlines()]
    labels = {}
    for line in (ROOT/'evaluation/retrieval_ablation_40/qrels.jsonl').read_text().splitlines():
        r=json.loads(line); labels.setdefault(r['query_id'],{})[r['doc_id']]=r['relevance']
    model=CrossEncoderReranker(RerankerConfig(local_files_only=True))
    data=json.loads(cache.read_text()) if cache.exists() else {}
    for q in queries:
        qid=q['query_id']
        if qid in data: continue
        c=json.loads((ROOT/f'results/ablation_40/collection/{qid}.json').read_text(encoding='utf8'))
        rows=c['runs']['05_aspect']
        query=c['details']['05_aspect']['search_text'] or q['query']
        scored=model.rerank(query,rows,query_aspects=[],top_k=None)
        scores={r['chunk_id']:r['cross_encoder_raw_score'] for r in scored}
        data[qid]={'rows':rows,'query':query,'scores':[scores[r['chunk_id']] for r in rows]}
        cache.write_text(json.dumps(data,ensure_ascii=False),encoding='utf8')
        print(qid,flush=True)
    grid=[(w,k) for w in [0,.1,.2,.3,.4,.5,.6,.7,.8,1] for k in [1,10,60]]
    values=[]
    for w,k in grid:
        values.append([metrics([r['chunk_id'] for r in [data[q['query_id']]['rows'][i] for i in order(data[q['query_id']]['rows'],data[q['query_id']]['scores'],w,k)]],labels[q['query_id']]) for q in queries])
    a=np.array(values)
    # Deterministic folds, assigned before parameter selection.
    folds=np.arange(len(queries))%5
    selected=[]; held=[]
    for f in range(5):
        best=int(np.argmax(a[:,folds!=f,2].mean(axis=1)))
        selected.append({'fold':f,'weight':grid[best][0],'offset':grid[best][1]})
        held.extend(a[best,folds==f].tolist())
    best=int(np.argmax(a[:,:,2].mean(axis=1)))
    baseline=a[0].mean(axis=0)
    report={'protocol':'Fixed retrieval Top-20; model scores model-stripped query and raw text. Five-fold parameter selection; same queries previously inspected, not an untouched test set.',
            'metrics':['P@10','MRR@10','nDCG@10'],
            'baseline':baseline.tolist(),'cross_validated':np.mean(held,axis=0).tolist(),
            'folds':selected,'deployment':{'weight':grid[best][0],'offset':grid[best][1]},
            'development':a[best].mean(axis=0).tolist(),
            'grid':[{'weight':w,'offset':k,'mean':a[i].mean(axis=0).tolist()} for i,(w,k) in enumerate(grid)]}
    (OUT/'selection.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in report.items() if k!='grid'},indent=2))

if __name__=='__main__': main()

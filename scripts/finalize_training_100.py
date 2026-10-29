"""Validate fixed training candidates and export ID-keyed relevance labels."""
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'evaluation/reranker_training_100'

def write_jsonl(path,rows):
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf8',newline='\n')

def main():
    queries=list(map(json.loads,(OUT/'queries.jsonl').read_text(encoding='utf8').splitlines()))
    cached={r['query_id']:r for r in map(json.loads,(OUT/'scored_candidates.jsonl').read_text(encoding='utf8').splitlines())}
    labels={}
    for r in map(json.loads,(OUT/'qrels.jsonl').read_text().splitlines()):
        key=(r['query_id'],r['doc_id']);assert key not in labels;labels[key]=r['relevance']
    assert len(queries)==len(cached)==100 and len(labels)==2000
    evaluation=ROOT/'evaluation/retrieval_ablation_40_v2'
    eval_q=list(map(json.loads,(evaluation/'queries.jsonl').read_text(encoding='utf8').splitlines()))
    judged_ids={r['doc_id'] for r in map(json.loads,(evaluation/'qrels.jsonl').read_text().splitlines())}
    eval_c=[r for r in map(json.loads,(evaluation/'corpus.jsonl').read_text(encoding='utf8').splitlines()) if r['chunk_id'] in judged_ids]
    assert not ({q['query'].casefold() for q in queries}&{q['query'].casefold() for q in eval_q})
    assert not ({(p['brand'],p['model']) for q in queries for p in q['target_products']}&{(p['brand'],p['model']) for q in eval_q for p in q['target_products']})
    corpus={};labeled=[];qrels=[]
    for q in queries:
        d=cached[q['query_id']];assert len(d['rows'])==len(d['scores'])==20
        assert len({r['chunk_id'] for r in d['rows']})==len({r['doc_id'] for r in d['rows']})==20
        candidates=[]
        for rank,r in enumerate(d['rows'],1):
            g=labels[q['query_id'],r['chunk_id']];assert type(g) is int and g in (0,1,2,3)
            corpus[r['chunk_id']]=r
            qrels.append({'query_id':q['query_id'],'doc_id':r['chunk_id'],'relevance':g})
            candidates.append({'chunk_id':r['chunk_id'],'source_doc_id':r['doc_id'],'brand':r['brand'],
                'model':r['model'],'text':r['text'],'retrieval_rank':rank,'relevance':g})
        labeled.append({**q,'candidates':candidates})
    for field in ['chunk_id','doc_id','text']:
        assert not ({r[field] for r in corpus.values()}&{r[field] for r in eval_c})
    found={}
    with (ROOT/'data/corpus.jsonl').open(encoding='utf8') as f:
        for line in f:
            r=json.loads(line)
            if r['chunk_id'] in corpus:
                h=corpus[r['chunk_id']];assert (r['text'],r['doc_id'],r['model'])==(h['text'],h['doc_id'],h['model'])
                found[r['chunk_id']]=r
    assert len(found)==len(corpus)
    write_jsonl(OUT/'dataset_labeled.jsonl',labeled);write_jsonl(OUT/'qrels.jsonl',qrels)
    write_jsonl(OUT/'corpus.jsonl',[found[c] for c in sorted(found)])
    (OUT/'qrels.trec').write_text(''.join(f'{r["query_id"]} 0 {r["doc_id"]} {r["relevance"]}\n' for r in qrels),encoding='utf8',newline='\n')
    manifest={'name':'reranker_training_100','queries':100,'pairs':2000,'candidates_per_query':20,
        'products':16,'unique_chunks':len(corpus),'unique_reviews':len({r['doc_id'] for r in corpus.values()}),
        'label_distribution':dict(sorted(Counter(str(r['relevance']) for r in qrels).items())),
        'validation':{'source_text_verified':True,'query_product_chunk_review_text_disjoint_from_evaluation':True},
        'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='manifest.json'}}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8',newline='\n')
    print(json.dumps({k:v for k,v in manifest.items() if k!='files'},indent=2))

if __name__=='__main__':main()

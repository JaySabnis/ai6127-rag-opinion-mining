"""Reproduce eight ablations from shared candidate text and frozen score caches.

Default: use saved retrieval inputs and MiniLM logits.
--rescore: repeat MiniLM inference on the saved retrieval pools.
--refresh-retrieval: repeat retrieval using shared indexes and cached rewrites.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import torch
from retrieval.rerank import CrossEncoderReranker, RerankerConfig, DEFAULT_REVISION

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'evaluation/retrieval_ablation_40_v2'
OUT=ROOT/'results/ablation_latest'
IDS=['01_bm25','02_dense','03_rrf','04_strip_model','05_aspect','06_rerank','07_rewrite','08_learned_v2']
FEATURE_FIELDS={'chunk_id','rank','row_id','score','bm25_score','bm25_rank','dense_score','dense_rank','aspect_boosted','rrf_score','rrf_normalized'}

def write_jsonl(path,rows):
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf8',newline='\n')

def refresh_retrieval(queries,corpus):
    from retrieval import Retriever
    from app.pipeline import product_filter
    from retrieval.query_rewrite import validate_result
    retriever=Retriever.load(ROOT/'indexes')
    rewrites={r['query_id']:validate_result({k:r[k] for k in ('query','product','aspects')})
              for r in map(json.loads,(DATA/'rewrites.jsonl').read_text(encoding='utf8').splitlines())}
    snapshots=[]
    def compact(rows):
        for r in rows:
            if r['chunk_id'] not in corpus:corpus[r['chunk_id']]={k:v for k,v in r.items() if k not in FEATURE_FIELDS or k=='chunk_id'}
        return [{k:v for k,v in r.items() if k in FEATURE_FIELDS} for r in rows]
    for q in queries:
        runs,details={},{}
        for name,mode,strip,boost in [('01_bm25','bm25',False,0),('02_dense','dense',False,0),('03_rrf','hybrid',False,0),('04_strip_model','hybrid',True,0),('05_aspect','hybrid',True,.5)]:
            r=retriever.retrieve(q['query'],mode=mode,k=100,candidate_k=150,strip_model=strip,aspect_weight=boost,max_per_doc=1)
            runs[name]=compact([h.to_dict() for h in r.hits])
            details[name]={'filters':r.filters,'search_text':r.search_text,'aspects':r.parsed.aspects,'candidate_count':len(r.hits)}
        rw=rewrites[q['query_id']];filters=[product_filter(retriever,p) for p in rw['product']] or [None]
        merged={};groups=[];scoring_query=rw['query']
        for f in filters:
            r=retriever.retrieve(rw['query'],k=100,candidate_k=150,filters=f,auto_filter=False,strip_model=True,aspects=rw['aspects'],aspect_weight=.5,max_per_doc=1)
            groups.append({'filters':r.filters,'search_text':r.search_text,'candidate_count':len(r.hits)})
            if f and f.get('model'):scoring_query=retriever.parser.remove_model(scoring_query,f['model'])
            for h in r.hits:
                row=h.to_dict()
                if row['chunk_id'] not in merged or row['score']>merged[row['chunk_id']]['score']:merged[row['chunk_id']]=row
        snapshots.append({'query_id':q['query_id'],'retrieval_runs':runs,'retrieval_details':details,
            'rewritten_candidates':compact(sorted(merged.values(),key=lambda r:r['score'],reverse=True)[:100]),
            'rewrite_details':{'rewrite':rw,'groups':groups,'scoring_query':scoring_query},'semantic_cache':{}})
        print(q['query_id'],'retrieved',flush=True)
    return snapshots

class CachedLogits:
    """Use the same production ranking path with verified frozen input pairs."""
    def __init__(self,cache,rows):
        if cache['chunk_ids']!=[r['chunk_id'] for r in rows[:20]]:raise ValueError('Semantic cache candidate order differs; use --rescore')
        if cache['text_sha256']!=[hashlib.sha256(r['text'].encode()).hexdigest() for r in rows[:20]]:raise ValueError('Semantic cache passage text differs; use --rescore')
        if cache['model_revision']!=DEFAULT_REVISION:raise ValueError('Semantic cache model revision differs; use --rescore')
        self.pairs=[(cache['query'],r['text']) for r in rows[:20]];self.logits=cache['logits']
    def predict(self,pairs,**kwargs):
        if pairs!=self.pairs:raise ValueError('Semantic cache query/text differs; use --rescore')
        return self.logits

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rescore',action='store_true');parser.add_argument('--refresh-retrieval',action='store_true')
    args=parser.parse_args()
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    torch.set_num_threads(min(8,os.cpu_count() or 1))
    OUT.mkdir(exist_ok=True);(OUT/'runs').mkdir(exist_ok=True)
    queries=list(map(json.loads,(DATA/'queries.jsonl').read_text(encoding='utf8').splitlines()))
    corpus={r['chunk_id']:r for r in map(json.loads,(DATA/'corpus.jsonl').read_text(encoding='utf8').splitlines())}
    inputs=refresh_retrieval(queries,corpus) if args.refresh_retrieval else list(map(json.loads,(OUT/'inputs.jsonl').read_text(encoding='utf8').splitlines()))
    snapshots={r['query_id']:r for r in inputs}
    assert len(snapshots)==len(queries)==40
    fusion=CrossEncoderReranker(RerankerConfig(strategy='rank_fusion',local_files_only=True))
    learned=CrossEncoderReranker(RerankerConfig(strategy='learned',local_files_only=True))
    lines={k:[] for k in IDS};pending={}
    known={(r['query_id'],r['doc_id']) for r in map(json.loads,(DATA/'qrels.jsonl').read_text().splitlines())}
    for q in queries:
        qid=q['query_id'];d=snapshots[qid]
        hydrate=lambda rows:[{**corpus[r['chunk_id']],**r} for r in rows]
        runs={name:hydrate(rows) for name,rows in d['retrieval_runs'].items()}
        settings=[('06_rerank',q['query'],runs['05_aspect'],d['retrieval_details']['05_aspect']['search_text'],d['retrieval_details']['05_aspect']['aspects']),
                  ('07_rewrite',d['rewrite_details']['rewrite']['query'],hydrate(d['rewritten_candidates']),d['rewrite_details']['scoring_query'],d['rewrite_details']['rewrite']['aspects'])]
        for name,query,rows,scoring_query,aspects in settings:
            scoring_query=scoring_query.strip() or query.strip()
            if not (args.rescore or args.refresh_retrieval):fusion._model=CachedLogits(d['semantic_cache'][name],rows)
            ranked=fusion.rerank(query,rows,scoring_query=scoring_query,query_aspects=aspects,top_k=100)
            head=sorted(ranked[:20],key=lambda r:r['retrieval_rank'])
            d['semantic_cache'][name]={'query':scoring_query,'chunk_ids':[r['chunk_id'] for r in head],
                'logits':[r['cross_encoder_raw_score'] for r in head],
                'text_sha256':[hashlib.sha256(r['text'].encode()).hexdigest() for r in head],
                'model_revision':DEFAULT_REVISION}
            runs[name]=ranked
            assert [r['chunk_id'] for r in ranked[20:]]==[r['chunk_id'] for r in rows[20:]]
            if name=='07_rewrite':
                learned._model=CachedLogits(d['semantic_cache'][name],rows)
                runs['08_learned_v2']=learned.rerank(query,rows,scoring_query=scoring_query,query_aspects=aspects,top_k=100)
        for name in IDS:
            rows=runs[name]
            assert len(rows)==len({r['chunk_id'] for r in rows})==len({r['doc_id'] for r in rows})==100
            lines[name].extend(f'{qid} Q0 {r["chunk_id"]} {i} {101-i} {name}' for i,r in enumerate(rows,1))
            for r in rows[:10]:
                if (qid,r['chunk_id']) not in known:pending[qid+':'+r['chunk_id']]={'query_id':qid,'query':q['query'],'candidate':r}
        print(qid,'eight runs reconstructed',flush=True)
    for name,rows in lines.items():(OUT/'runs'/f'{name}.trec').write_text('\n'.join(rows)+'\n',encoding='utf8',newline='\n')
    write_jsonl(OUT/'inputs.jsonl',inputs)
    if args.refresh_retrieval:write_jsonl(DATA/'corpus.jsonl',[corpus[k] for k in sorted(corpus)])
    if pending:(OUT/'pending_top10.json').write_text(json.dumps(list(pending.values()),ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    elif (OUT/'pending_top10.json').exists():(OUT/'pending_top10.json').unlink()
    manifest={'systems':IDS,'queries':40,'top_k':100,'rerank_window':20,'algorithm':'rank_fusion: semantic_weight=.8, offset=10',
        'trained_configuration':'08_learned_v2','retrieval_rerun':args.refresh_retrieval,
        'minilm_inference_rerun':args.rescore or args.refresh_retrieval,'minilm_revision':DEFAULT_REVISION,
        'inputs_sha256':hashlib.sha256((OUT/'inputs.jsonl').read_bytes()).hexdigest(),
        'model_sha256':hashlib.sha256((ROOT/'retrieval/models/review_ranker_v2.json').read_bytes()).hexdigest(),
        'unjudged_top10_pairs':len(pending)}
    (OUT/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8',newline='\n')
    if pending:raise SystemExit('Grade pending Top10 pairs, export evaluation labels, then rerun reporting.')

if __name__=='__main__':main()

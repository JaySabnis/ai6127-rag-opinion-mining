"""Export canonical evaluation labels and verify complete Top10 coverage."""
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'evaluation/retrieval_ablation_40_v2'
OUT=ROOT/'results/ablation_latest'

def main():
    queries=list(map(json.loads,(DATA/'queries.jsonl').read_text(encoding='utf8').splitlines()))
    labeled=list(map(json.loads,(DATA/'dataset_labeled.jsonl').read_text(encoding='utf8').splitlines()))
    corpus={r['chunk_id']:r for r in map(json.loads,(DATA/'corpus.jsonl').read_text(encoding='utf8').splitlines())}
    assert {q['query_id'] for q in labeled}=={q['query_id'] for q in queries}
    rows=[];known={};reviews=set()
    for q in labeled:
        assert len({c['chunk_id'] for c in q['candidates']})==len(q['candidates'])
        for c in q['candidates']:
            cid=c['chunk_id'];r=corpus[cid];g=c['relevance']
            assert type(g) is int and g in (0,1,2,3)
            assert (c['text'],c['source_doc_id'],c['model'])==(r['text'],r['doc_id'],r['model'])
            known[q['query_id'],cid]=g;reviews.add(r['doc_id'])
            rows.append({'query_id':q['query_id'],'doc_id':cid,'relevance':g})
    (DATA/'qrels.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf8',newline='\n')
    (DATA/'qrels.trec').write_text(''.join(f'{r["query_id"]} 0 {r["doc_id"]} {r["relevance"]}\n' for r in rows),encoding='utf8',newline='\n')
    coverage={}
    for p in sorted((OUT/'runs').glob('*.trec')):
        ranks={}
        for line in p.read_text().splitlines():
            q,_,cid,rank,*_=line.split();ranks.setdefault(q,[]).append((int(rank),cid))
        assert set(ranks)=={q['query_id'] for q in queries}
        for q,items in ranks.items():
            assert len(items)==len({cid for _,cid in items})==100
            assert all((q,cid) in known for _,cid in sorted(items)[:10]),(p.name,q)
        coverage[p.stem]=True
    manifest={'name':'retrieval_evaluation40','queries':len(queries),'judged_pairs':len(rows),
        'judged_unique_chunks':len({r['doc_id'] for r in rows}),'judged_source_reviews':len(reviews),
        'candidate_corpus_chunks':len(corpus),'label_distribution':dict(sorted(Counter(str(r['relevance']) for r in rows).items())),
        'validation':{'top10_fully_judged':coverage,'same_qrels_for_all_metrics':True},
        'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(DATA.iterdir()) if p.is_file() and p.name!='manifest.json'}}
    (DATA/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8',newline='\n')
    print(json.dumps({k:v for k,v in manifest.items() if k not in ('files','validation')},indent=2))

if __name__=='__main__':main()

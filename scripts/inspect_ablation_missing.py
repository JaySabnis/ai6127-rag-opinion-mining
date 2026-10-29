import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/ablation_8'
def main():
    known={}
    for s in (ROOT/'evaluation/retrieval_ablation_40/qrels.jsonl').read_text().splitlines():
        r=json.loads(s);known.setdefault(r['query_id'],{})[r['doc_id']]=r['relevance']
    extra=json.loads((OUT/'supplemental_labels.json').read_text())
    queries={r['query_id']:r['query'] for r in map(json.loads,(ROOT/'evaluation/retrieval_ablation_40/queries.jsonl').read_text().splitlines())}
    missing={}
    for p in sorted((OUT/'collection').glob('*.json')):
        d=json.loads(p.read_text(encoding='utf8'));qid=d['query_id']
        for name,hits in d['runs'].items():
            for h in hits[:10]:
                if h['chunk_id'] not in known[qid] and h['chunk_id'] not in extra.get(qid,{}):
                    missing[qid,h['chunk_id']]={'qid':qid,'query':queries[qid],'id':h['chunk_id'],'model':h['model'],'text':h['text']}
    (OUT/'pending_top10.json').write_text(json.dumps(list(missing.values()),ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(list(missing.values()),ensure_ascii=False))
if __name__=='__main__':main()

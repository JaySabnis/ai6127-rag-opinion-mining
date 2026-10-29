"""Print complete, deduplicated candidate text for direct relevance judgment."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]/'evaluation/reranker_training_100'
def main():
    group=int(sys.argv[1]);start=(group-1)*6+1
    ids=[f'TR{i:03d}' for i in range(start,start+6)]
    extra={1:'TR097',6:'TR098',12:'TR099',16:'TR100'}
    if group in extra:ids.append(extra[group])
    rows={};order=[]
    for qid in ids:
        q=json.loads((ROOT/'collection'/f'{qid}.json').read_text(encoding='utf8'))
        mapping=[]
        for r in q['rows']:
            cid=r['chunk_id']
            if cid not in rows:rows[cid]=r;order.append(cid)
            mapping.append(order.index(cid)+1)
        print(qid,q['aspects'],'candidates=',mapping)
    for i,cid in enumerate(order,1):
        if len(sys.argv)>2 and not int(sys.argv[2]) <= i <= int(sys.argv[3]):continue
        print(f"[{i}] {cid}: {rows[cid]['text']}")
if __name__=='__main__':main()

"""Validate and export the directly judged training set."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'evaluation/reranker_training_100'

def write_jsonl(path, rows):
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf8')

def main():
    queries=[json.loads(s) for s in (OUT/'queries.jsonl').read_text(encoding='utf8').splitlines()]
    labels=json.loads((OUT/'direct_labels.json').read_text())
    assert len(queries)==len(labels)==100
    old=ROOT/'evaluation/retrieval_ablation_40_v2'
    old_q=[json.loads(s) for s in (old/'queries.jsonl').read_text(encoding='utf8').splitlines()]
    old_c=[json.loads(s) for s in (old/'corpus.jsonl').read_text(encoding='utf8').splitlines()]
    assert not ({q['query'].casefold() for q in queries}&{q['query'].casefold() for q in old_q})
    assert not ({(p['brand'],p['model']) for q in queries for p in q['target_products']}&{(p['brand'],p['model']) for q in old_q for p in q['target_products']})
    corpus={};labeled=[];qrels=[];features=[]
    for q in queries:
        d=json.loads((OUT/'collection'/f"{q['query_id']}.json").read_text(encoding='utf8'))
        grades=labels[q['query_id']]
        assert len(d['rows'])==len(grades)==len(d['scores'])==20
        assert len({h['doc_id'] for h in d['rows']})==20
        candidates=[]
        for rank,(h,g) in enumerate(zip(d['rows'],grades,strict=True),1):
            assert g in (0,1,2,3)
            corpus[h['chunk_id']]=h
            qrels.append({'query_id':q['query_id'],'doc_id':h['chunk_id'],'relevance':g})
            candidates.append({'chunk_id':h['chunk_id'],'source_doc_id':h['doc_id'],
                'brand':h['brand'],'model':h['model'],'text':h['text'],
                'retrieval_rank':rank,'relevance':g})
        labeled.append({**q,'candidates':candidates})
        features.append({'query_id':q['query_id'],'query':d['search_text'],
                         'product_group':q['product_group'],'rows':d['rows'],'scores':d['scores']})
    assert not (set(corpus)&{r['chunk_id'] for r in old_c})
    assert not ({r['doc_id'] for r in corpus.values()}&{r['doc_id'] for r in old_c})
    assert not ({r['text'].strip() for r in corpus.values()}&{r['text'].strip() for r in old_c})
    # Verify every stored candidate against the actual full source corpus.
    found={}
    with (ROOT/'data/corpus.jsonl').open(encoding='utf8') as f:
        for line in f:
            r=json.loads(line)
            if r['chunk_id'] in corpus:
                h=corpus[r['chunk_id']]
                assert (r['text'],r['doc_id'],r['model'])==(h['text'],h['doc_id'],h['model'])
                found[r['chunk_id']]=r
    assert len(found)==len(corpus)
    write_jsonl(OUT/'dataset_labeled.jsonl',labeled)
    write_jsonl(OUT/'qrels.jsonl',qrels)
    write_jsonl(OUT/'corpus.jsonl',[found[c] for c in sorted(found)])
    write_jsonl(OUT/'scored_candidates.jsonl',features)
    (OUT/'qrels.trec').write_text(''.join(f"{r['query_id']} 0 {r['doc_id']} {r['relevance']}\n" for r in qrels),encoding='utf8')
    (OUT/'qrels.tsv').write_text('query_id\tchunk_id\trelevance\n'+''.join(f"{r['query_id']}\t{r['doc_id']}\t{r['relevance']}\n" for r in qrels),encoding='utf8')
    manifest={'name':'reranker_training_100','queries':100,'pairs':2000,'candidates_per_query':20,
        'products':16,'unique_chunks':len(corpus),'unique_reviews':len({r['doc_id'] for r in corpus.values()}),
        'label_distribution':dict(sorted(Counter(r['relevance'] for r in qrels).items())),
        'query_types':dict(Counter(q['query_type'] for q in queries)),
        'collection':{'scope':'full existing indexes','mode':'hybrid','retrieval_k':50,
                      'candidate_k_per_channel':150,'aspect_weight':.5,'strip_model':True,'max_per_doc':1,
                      'saved_candidates':'fixed retrieval Top20, before reranking'},
        'validation':{'query_disjoint_from_eval40':True,'target_product_disjoint_from_eval40':True,
                      'chunk_disjoint_from_eval40':True,'review_disjoint_from_eval40':True,
                      'exact_text_disjoint_from_eval40':True,
                      'source_text_verified':True,'all_2000_pairs_labeled':True},
        'grading':'0 irrelevant/wrong product; 1 weak/indirect; 2 clear partial/brief; 3 detailed direct evidence; multi-aspect one-aspect support capped at 2',
        'miniLM_frozen':True,'scoring_api_used':False,
        'files':{n:hashlib.sha256((OUT/n).read_bytes()).hexdigest() for n in ['queries.jsonl','qrels.jsonl','dataset_labeled.jsonl','corpus.jsonl','scored_candidates.jsonl']}}
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    (OUT/'README.md').write_text('# Dedicated Reranker Training Set: 100 Queries\n\nThe set contains 100 English queries covering 16 products: 96 single-aspect queries (16 each for camera, battery, display, performance, service and carrier) and four camera-and-battery queries. Query text, target products, chunks, source reviews and exact text are disjoint from evaluation40.\n\n## Fixed candidate collection\n\nSearch the full shared indexes with product filtering, model-name removal, BM25/dense RRF and retrieval aspect boosting. Retrieve 50 results at depth 150 per channel, keeping at most one chunk per source review. Freeze the first 20 candidates before judging. No candidates are replaced to balance the labels.\n\nThere are 2,000 query-chunk judgments, 1,797 unique chunks and 1,703 source reviews. Grade counts are 0: 225; 1: 209; 2: 592; 3: 974.\n\n## Relevance grades\n\n- 0: wrong product or no answer-bearing content.\n- 1: vague, indirect or weak relevance.\n- 2: clear but brief or partial support.\n- 3: detailed direct experience, context or comparison.\n\nPositive and negative opinions can both be relevant. Multi-aspect evidence covering only one aspect is capped at 2. Explicit product references take precedence over metadata. Battery health is not endurance, shipping speed is not performance, and bootloader unlocking is not carrier unlocking. Candidate files contain original text and relevance grades without rationale or evidence fields.\n\n## Files\n\n- `queries.jsonl`: the 100 fixed questions, product groups and aspect descriptions.\n- `dataset_labeled.jsonl`: queries with the 20 candidates, original text and labels.\n- `qrels.jsonl`, `qrels.trec`, `qrels.tsv`: equivalent judgment formats; qrels `doc_id` is the chunk ID.\n- `corpus.jsonl`: source metadata for all judged chunks.\n- `scored_candidates.jsonl`: original retrieval order, retrieval features, model-stripped queries and frozen MiniLM logits for training reproduction.\n- `collection/`: per-query collection checkpoints.\n- `direct_labels.json`: 20 grades per query in its original retrieval order.\n- `manifest.json`: collection settings, counts, hashes and separation checks.\n\n## Training and model\n\n```powershell\nuv run python -X utf8 -m scripts.finalize_training_100\nuv run python -X utf8 -m scripts.train_ranker_100\n```\n\nRun from the repository root. MiniLM is frozen; only the Ridge fusion layer is fitted. Product-group cross-validation selects parameters using training100 only, then the frozen model is checked on evaluation40.\n\nThe default artifact is `retrieval/models/review_ranker_v2.json`. It uses seven features, relevance-grade targets and ridge penalty 100. Nested product-group validation gives nDCG@10=0.8635. Training fit is not a test score. Evaluation40 influenced earlier project development and is not an untouched final test set.\n\nSee the [reranker guide](../../retrieval/RERANKER_README.md), [training results](../../results/reranker_training_100/README.md) and [latest ablation report](../../results/ablation_latest/README.md). Delivery archives can be generated locally and are excluded from version control.\n',encoding='utf8')
    with zipfile.ZipFile(OUT.with_suffix('.zip'),'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file():z.write(p,OUT.name+'/'+p.relative_to(OUT).as_posix())
    print(json.dumps(manifest,ensure_ascii=False,indent=2))

if __name__=='__main__':main()

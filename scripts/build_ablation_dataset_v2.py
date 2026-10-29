"""Freeze the original judgments plus the eight-system Top10 supplements as v2."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'evaluation/retrieval_ablation_40'
RUN = ROOT / 'results/ablation_8'
OUT = ROOT / 'evaluation/retrieval_ablation_40_v2'

def read_jsonl(path):
    return [json.loads(s) for s in path.read_text(encoding='utf8').splitlines() if s]

def write_jsonl(path,rows):
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf8')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    if OUT.exists():
        raise FileExistsError(f'Frozen dataset already exists: {OUT}')
    queries=read_jsonl(OLD/'queries.jsonl')
    original=read_jsonl(OLD/'qrels.jsonl')
    labels={q['query_id']:{} for q in queries}
    for r in original:labels[r['query_id']][r['doc_id']]=r['relevance']
    supplements=json.loads((RUN/'supplemental_labels.json').read_text())
    added=0
    for qid,items in supplements.items():
        for cid,grade in items.items():
            assert cid not in labels[qid] and grade in (0,1,2,3)
            labels[qid][cid]=grade;added+=1
    assert added==62 and sum(map(len,labels.values()))==2626
    wanted={cid for items in labels.values() for cid in items}
    corpus={}
    with (ROOT/'data/corpus.jsonl').open(encoding='utf8') as f:
        for line in f:
            r=json.loads(line)
            if r['chunk_id'] in wanted:corpus[r['chunk_id']]=r
    assert set(corpus)==wanted
    old_text={c['chunk_id']:c['text'] for r in read_jsonl(OLD/'dataset_labeled.jsonl') for c in r['candidates']}
    assert all(corpus[cid]['text']==text for cid,text in old_text.items())
    snapshots={q['query_id']:json.loads((RUN/'collection'/f"{q['query_id']}.json").read_text(encoding='utf8')) for q in queries}
    systems=sorted(next(iter(snapshots.values()))['runs'])
    membership={};run_rows=[]
    for qid,d in snapshots.items():
        assert sorted(d['runs'])==systems
        for system,hits in d['runs'].items():
            assert len(hits)==100
            assert all(h['chunk_id'] in labels[qid] for h in hits[:10])
            for rank,h in enumerate(hits,1):
                if h['chunk_id'] in labels[qid]:
                    membership.setdefault((qid,h['chunk_id']),{})[system]=rank
                run_rows.append({'system':system,'query_id':qid,'doc_id':h['chunk_id'],
                                 'rank':rank,'score':101-rank,
                                 'actual_score':h.get('rerank_score',h['score'])})
    OUT.mkdir(parents=True)
    qrels=[{'query_id':qid,'doc_id':cid,'relevance':g} for qid,items in labels.items() for cid,g in sorted(items.items())]
    write_jsonl(OUT/'qrels.jsonl',qrels)
    (OUT/'qrels.trec').write_text(''.join(f"{r['query_id']} 0 {r['doc_id']} {r['relevance']}\n" for r in qrels),encoding='utf8')
    (OUT/'qrels.tsv').write_text('query_id\tchunk_id\trelevance\n'+''.join(f"{r['query_id']}\t{r['doc_id']}\t{r['relevance']}\n" for r in qrels),encoding='utf8')
    write_jsonl(OUT/'corpus.jsonl',[corpus[cid] for cid in sorted(corpus)])
    labeled=[]
    for q in queries:
        candidates=[]
        for cid,grade in sorted(labels[q['query_id']].items()):
            c=corpus[cid]
            candidates.append({'chunk_id':cid,'source_doc_id':c['doc_id'],'brand':c['brand'],
                               'model':c['model'],'text':c['text'],'relevance':grade})
        labeled.append({**q,'candidates':candidates})
    write_jsonl(OUT/'dataset_labeled.jsonl',labeled)
    write_jsonl(OUT/'pool_membership.jsonl',[{'query_id':r['query_id'],'chunk_id':r['doc_id'],
        'system_ranks':membership.get((r['query_id'],r['doc_id']),{}),
        'label_origin':'v2_top10_supplement' if r['doc_id'] in supplements.get(r['query_id'],{}) else 'v1_retained'} for r in qrels])
    for name in ['queries.jsonl','rewrites.jsonl','rewrite_prompt.txt','evaluate.py']:
        (OUT/name).write_bytes((OLD/name).read_bytes())
    (OUT/'runs').mkdir()
    for p in (RUN/'runs').glob('*.trec'):(OUT/'runs'/p.name).write_bytes(p.read_bytes())
    write_jsonl(OUT/'runs.jsonl',run_rows)
    config={'source_experiment':'results/ablation_8','systems':systems,
            'retrieval_depth_per_channel':300,'saved_results':100,'max_chunks_per_review':1,
            'learned_window':20,'legacy_window':100,
            'learned_model_sha256':sha(ROOT/'retrieval/models/review_ranker_v1.json'),
            'recall_qrels':'qrels.trec; unified with ranking metrics in v2',
            'retraining_performed':False,'metrics_recomputed':False}
    (OUT/'collection_config.json').write_text(json.dumps(config,indent=2)+'\n',encoding='utf8')
    distribution=dict(sorted(Counter(r['relevance'] for r in qrels).items()))
    summary={'name':'retrieval_ablation_40_v2','version':'2.0','date':'2026-10-29',
             'queries':40,'judged_pairs':len(qrels),'retained_v1_pairs':len(original),'added_pairs':added,
             'unique_chunks':len(corpus),'unique_source_reviews':len({r['doc_id'] for r in corpus.values()}),
             'label_distribution':distribution,
             'per_query':[{'query_id':qid,'judged':len(items),'binary_relevant':sum(g>=2 for g in items.values())} for qid,items in labels.items()],
             'metrics':{'P':10,'Recall':100,'MRR':10,'nDCG':10,'binary_relevance':'>=2','ndcg_gain':'linear 0,1,2,3','qrels':'one shared qrels for all four metrics','recall_scope':'pooled incomplete judgments'},
             'validation':{'original_labels_unchanged':all(labels[r['query_id']][r['doc_id']]==r['relevance'] for r in original),
                           'all_eight_top10_judged':True,'source_text_verified':True,'eight_saved_top100_runs':True},
             'development_status':'Questions overlap learned-ranker training; this is not an independent held-out test set.',
             'inputs':{'v1_qrels_sha256':sha(OLD/'qrels.jsonl'),'supplement_sha256':sha(RUN/'supplemental_labels.json')},
             'file_sha256':{p.relative_to(OUT).as_posix():sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()}}
    counts=[len(v) for v in labels.values()];positives=[sum(g>=2 for g in v.values()) for v in labels.values()]
    readme='# Retrieval Evaluation40: Version 2\n\nVersion 2 retains all 40 questions and 2,564 original judgments, adding 62 Top-10 supplements from the eight-system experiment. It contains **2,626 query-chunk judgments**, 2,084 unique chunks and 2,000 source reviews. Each query has 49-85 judged candidates and 21-59 binary positives. Grade counts are 0: 813; 1: 352; 2: 665; 3: 796.\n\n## Evaluation protocol\n\nAll four metrics use the same `qrels.trec`: Precision@10, pooled Recall@100, MRR@10 and nDCG@10. Grades >=2 are binary relevant. nDCG uses linear gains 0, 1, 2, 3 and query-specific IDCG from all judgments.\n\nRecall@100 divides known positives found in the first 100 results by all known positives for the query. Ranks 11-100 do not require exhaustive judging and unjudged items earn no known-positive credit. All Top-10 results must be judged.\n\nThe latest HTML report adds six later supplements and evaluates against **2,632 judgments**, using `results/ablation_latest/qrels.trec`. This directory remains the fixed 2,626-pair version. Use the appropriate qrels; do not copy Recall values across versions.\n\n## Relevance rubric\n\n- 0: wrong product or no content answering the question.\n- 1: weak, vague or indirect relevance.\n- 2: a clear partial answer or brief direct assessment.\n- 3: specific direct experience, comparison, context or outcome.\n\nPositive and negative opinions can both be relevant. Explicit product references take precedence over metadata; aspect tags are not relevance labels. Battery health is not endurance, shipping speed is not performance, and bootloader unlocking is not carrier unlocking. For multi-aspect questions, one-aspect support is capped at 2. Comparison evidence about either target product can be relevant.\n\n## Files\n\n- `queries.jsonl`: fixed queries and descriptive product/aspect targets; targets are not injected as extra retrieval inputs.\n- `dataset_labeled.jsonl`: all judged candidates and original text, grouped by question.\n- `qrels.trec`, `qrels.jsonl`, `qrels.tsv`: equivalent labels; qrels `doc_id` is chunk ID.\n- `corpus.jsonl`: full metadata for the judged subset. Actual retrieval searches the full indexes.\n- `runs/`, `runs.jsonl`: eight previously saved Top-100 runs, not newly collected runs.\n- `pool_membership.jsonl`: candidate ranks in the saved systems and label version origins. Original candidates without current ranks remain included.\n- `manifest.json`: counts, file hashes and validation records.\n\n## Run evaluation\n\nFrom the repository root:\n\n```powershell\nuv run --with ir_measures python -X utf8 evaluation/retrieval_ablation_40_v2/evaluate.py\n```\n\nThe script requires 100 saved results per question and complete Top-10 judgments. New unjudged Top-10 results must be supplemented consistently before comparing systems.\n\nThe packaged eighth run is the earlier learned model from the original eight-system collection. The latest training100 model is evaluated in `results/ablation_latest/`; the packaged runs here remain frozen.\n\n## Model and dataset separation\n\nThe current v2 model is trained only on 100 dedicated queries with 20 candidates each, totaling 2,000 training pairs. These 40 queries were excluded from the current fit and selection. Earlier model v1 did use them, and they influenced previous development, so they are not an untouched final test set. See [training100](../reranker_training_100/README.md).\n\nVersion 1 remains in `evaluation/retrieval_ablation_40/`. Local archives are optional delivery files and are not committed.\n'
    (OUT/'README.md').write_text(readme,encoding='utf8')
    summary['file_sha256']['README.md']=sha(OUT/'README.md')
    (OUT/'manifest.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    archive=OUT.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file():z.write(p,OUT.name+'/'+p.relative_to(OUT).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name,h in summary['file_sha256'].items():
            assert hashlib.sha256(z.read(OUT.name+'/'+name)).hexdigest()==h
    print(json.dumps({k:v for k,v in summary.items() if k not in ['per_query','file_sha256']},ensure_ascii=False,indent=2))
    print('Dataset:',OUT,'Archive bytes:',archive.stat().st_size)

if __name__=='__main__':main()

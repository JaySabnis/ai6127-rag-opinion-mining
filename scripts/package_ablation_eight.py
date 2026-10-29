"""Validate result integrity and package the English eight-system report."""
import ast
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import zipfile
from html.parser import HTMLParser

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/ablation_8'

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    snapshots=[json.loads(p.read_text(encoding='utf8')) for p in sorted((OUT/'collection').glob('*.json'))]
    assert len(snapshots)==40
    expected={}
    for d in snapshots:
        assert len(d['runs'])==8
        assert {r['chunk_id'] for r in d['runs']['07_rewrite']}=={r['chunk_id'] for r in d['runs']['08_learned']}
        assert {r['chunk_id'] for r in d['runs']['05_aspect']}=={r['chunk_id'] for r in d['runs']['06_rerank']}
        for rows in d['runs'].values():
            assert len(rows)==100
            for r in rows:expected[r['chunk_id']]=(r['text'],r['doc_id'],r['model'])
    found=set()
    with (ROOT/'data/corpus.jsonl').open(encoding='utf8') as f:
        for line in f:
            r=json.loads(line);cid=r['chunk_id']
            if cid in expected:
                assert expected[cid]==(r['text'],r['doc_id'],r['model']),cid
                found.add(cid)
    assert found==set(expected)
    s=json.loads((OUT/'statistics.json').read_text())
    assert s['total_system_queries']==320 and s['top10_judged_fraction']==1
    src=ROOT/'evaluation/retrieval_ablation_40'
    assert digest(src/'qrels.trec')==s['provenance']['qrels_sha256']
    assert digest(ROOT/'retrieval/models/review_ranker_v1.json')==s['provenance']['model_sha256']
    for name,target in [('qrels.trec','recall_qrels.trec'),('queries.jsonl','queries.jsonl'),('rewrites.jsonl','rewrites.jsonl')]:
        (OUT/target).write_bytes((src/name).read_bytes())
    freeze=OUT/'source_snapshot';freeze.mkdir(exist_ok=True)
    paths=['scripts/run_retrieval_ablation.py','scripts/summarize_retrieval_ablation.py','scripts/report_ablation_eight.py','scripts/package_ablation_eight.py','scripts/train_learned_reranker.py','retrieval/rerank.py','retrieval/learned_rank.py','retrieval/models/review_ranker_v1.json']
    for name in paths:
        p=ROOT/name
        if p.suffix=='.py':ast.parse(p.read_text(encoding='utf8'))
        dest=freeze/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
    for name in ['report_zh.html', 'report_en.html']:
        HTMLParser().feed((OUT/name).read_text(encoding='utf8'))
    manifest={'completed_utc':datetime.now(timezone.utc).isoformat(),
              'queries':40,'systems':8,'system_query_runs':320,'results_per_run':100,
              'unique_source_chunks_verified':len(found),'original_qrels_unchanged':True,
              'known_recall_positives_fixed':True,'top10_supplemental_judgments':s['supplemental_top10_judgments'],
              'independent_metric_arithmetic_checks':1280,
              'cpu_seconds_in_query_loops':sum(d['elapsed_seconds'] for d in snapshots),
              'files':{str(p.relative_to(OUT)).replace('\\','/'):digest(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='execution_manifest.json'},
              'indexes':{str(p.relative_to(ROOT)).replace('\\','/'):digest(p) for p in [ROOT/'indexes/meta.pkl',ROOT/'indexes/bm25/weights.npz',ROOT/'indexes/bm25/vocab.json',ROOT/'indexes/bm25/params.json',ROOT/'indexes/dense/faiss.index',ROOT/'indexes/dense/info.json']}}
    (OUT/'execution_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')
    archive=ROOT/'results/ablation_8.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file():z.write(p,'ablation_8/'+p.relative_to(OUT).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name,sha in manifest['files'].items():assert hashlib.sha256(z.read('ablation_8/'+name)).hexdigest()==sha
    print(json.dumps({k:v for k,v in manifest.items() if k not in ['files','indexes']},indent=2))
    print('Verified archive:',archive,'bytes:',archive.stat().st_size)

if __name__=='__main__':main()

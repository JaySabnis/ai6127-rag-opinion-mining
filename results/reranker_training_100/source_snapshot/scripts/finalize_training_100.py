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
    (OUT/'README.md').write_text('''# 排序模型专用训练集：100题

100个新英文问题，覆盖16款机型；96题为单方面（相机、电池、屏幕、性能、售后、运营商各16题），4题同时询问相机与续航。与原40题的问题、目标机型、chunk和原评论均无交集。

每题从完整共享索引进行产品过滤、去型号、BM25与向量RRF融合并加方面奖励，召回50条、固定保存前20条；每路深度150，每条原评论最多一个chunk。候选在标注前已经固定，没有按标签替换候选或调整正负比例。

共2000对0–3分相关性标注。0=产品错误或无回答内容；1=含糊、间接；2=明确但简短或部分回答；3=具体直接的体验、场景或比较。正负评价均可相关；多方面只覆盖一个方面最高2分。正文明确评价其他型号时优先于元数据；电池健康不等于续航、配送速度不等于性能、bootloader解锁不等于运营商解锁。不附理由或证据字段，没有使用DeepSeek评分。

`dataset_labeled.jsonl`按问题提供候选原文和标签；`qrels.*`为相同标注的标准导出；`corpus.jsonl`为源数据子集。`scored_candidates.jsonl`和`collection/`保留原始顺序、检索特征、去型号查询及冻结MiniLM原始分数，供复现训练。`direct_labels.json`按每题原召回顺序保存20个标签。

这是专用训练集，不能将拟合后的训练分数当作测试成绩。训练脚本按机型分组交叉验证，仅使用这100题选参数和拟合；现有40题仅作最后检查。40题曾参与先前方案开发，因此仍不属于从未接触过的最终盲测集。

复现：先运行 `python -m scripts.finalize_training_100` 导出与检查，再运行 `python -m scripts.train_ranker_100` 训练。MiniLM权重保持冻结，只拟合特征融合层。
''',encoding='utf8')
    with zipfile.ZipFile(OUT.with_suffix('.zip'),'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file():z.write(p,OUT.name+'/'+p.relative_to(OUT).as_posix())
    print(json.dumps(manifest,ensure_ascii=False,indent=2))

if __name__=='__main__':main()

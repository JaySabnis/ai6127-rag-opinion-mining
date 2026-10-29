"""Recompute the eight saved runs under common qrels; emit bilingual HTML."""
import csv
import hashlib
import html
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import ir_measures as ir
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/ablation_latest'
DATA=ROOT/'evaluation/retrieval_ablation_40_v2'
TRAIN=ROOT/'results/reranker_training_100'
IDS=['01_bm25','02_dense','03_rrf','04_strip_model','05_aspect','06_rerank','07_rewrite','08_learned_v2']
METRICS=['P@10','pooled_Recall@100','MRR@10','nDCG@10']
NAMES={
 'zh':['① 仅 BM25','② 仅向量检索','③ 混合检索 · RRF','④ 产品过滤后去掉型号','⑤ 增加召回方面加分','⑥ 增加重排序','⑦ 增加查询改写','⑧ 训练模型排序 · 100题'],
 'en':['1 · BM25 only','2 · Dense only','3 · Hybrid retrieval · RRF','4 · Remove model after filtering','5 · Add retrieval aspect boost','6 · Add reranking','7 · Add query rewriting','8 · Trained ranking model · 100 queries']}
GROUPS={'camera':('相机','Camera'),'battery':('电池','Battery'),'display':('屏幕','Display'),
 'performance':('性能','Performance'),'service':('售后服务','Service'),'carrier':('运营商','Carrier'),
 'multi_aspect':('多方面','Multiple aspects'),'comparison':('产品比较','Product comparison')}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def csv_file(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def calculate():
    OUT.mkdir(exist_ok=True);(OUT/'runs').mkdir(exist_ok=True)
    qrels=list(ir.read_trec_qrels(str(DATA/'qrels.trec')))
    grades=defaultdict(dict)
    for r in qrels:grades[r.query_id][r.doc_id]=r.relevance
    queries=[json.loads(s) for s in (ROOT/'evaluation/retrieval_ablation_40_v2/queries.jsonl').read_text(encoding='utf8').splitlines()]
    qids=[q['query_id'] for q in queries]
    measures=[ir.P(rel=2)@10,ir.Recall(rel=2)@100,ir.RR(rel=2)@10,ir.nDCG(gains={0:0,1:1,2:2,3:3},dcg='log2')@10]
    run_manifest=json.loads((OUT/'run_manifest.json').read_text())
    assert run_manifest['model_sha256']==sha(ROOT/'retrieval/models/review_ranker_v2.json')
    assert run_manifest['inputs_sha256']==sha(OUT/'inputs.jsonl')
    inputs={d['query_id']:d for d in map(json.loads,(OUT/'inputs.jsonl').read_text(encoding='utf8').splitlines())}
    corpus={d['chunk_id']:d for d in map(json.loads,(DATA/'corpus.jsonl').read_text(encoding='utf8').splitlines())}
    model=json.loads((ROOT/'retrieval/models/review_ranker_v2.json').read_text())
    values={};rows=[];summaries=[]
    for idx,system in enumerate(IDS):
        dest=OUT/'runs'/f'{system}.trec'
        run=list(ir.read_trec_run(str(dest)));assert len(run)==4000
        per={q:{} for q in qids}
        assert {r.query_id for r in run}==set(qids)
        for r in ir.iter_calc(measures,qrels,run):per[r.query_id][METRICS[measures.index(r.measure)]]=r.value
        for qid in qids:
            ranked=sorted((r for r in run if r.query_id==qid),key=lambda r:r.score,reverse=True)
            assert len(ranked)==len({r.doc_id for r in ranked})==100
            assert all(r.doc_id in grades[qid] for r in ranked[:10])
            if idx>=5:
                d=inputs[qid];pool=d['retrieval_runs']['05_aspect'] if idx==5 else d['rewritten_candidates']
                cache=d['semantic_cache']['06_rerank' if idx==5 else '07_rewrite']
                assert [r.doc_id for r in ranked[20:]]==[r['chunk_id'] for r in pool[20:]]
                assert {r.doc_id for r in ranked}=={r['chunk_id'] for r in pool}
                assert cache['chunk_ids']==[r['chunk_id'] for r in pool[:20]]
                if idx<7:
                    semantic={i:r for r,i in enumerate(sorted(range(20),key=lambda i:cache['logits'][i],reverse=True),1)}
                    scores=[(1-.8)/(10+i+1)+.8/(10+semantic[i]) for i in range(20)]
                else:
                    from retrieval.learned_rank import predict
                    head=[{**corpus[r['chunk_id']],**r} for r in pool[:20]]
                    scores=predict(cache['query'],head,cache['logits'],model)
                assert [r.doc_id for r in ranked[:20]]==[pool[i]['chunk_id'] for i in sorted(range(20),key=lambda i:scores[i],reverse=True)]
            g=[grades[qid][r.doc_id] for r in ranked[:10]]
            dcg=lambda a:sum(v/math.log2(i+2) for i,v in enumerate(a))
            expected=[sum(v>=2 for v in g)/10,
              sum(grades[qid].get(r.doc_id,0)>=2 for r in ranked)/sum(v>=2 for v in grades[qid].values()),
              next((1/(i+1) for i,v in enumerate(g) if v>=2),0),
              dcg(g)/dcg(sorted(grades[qid].values(),reverse=True)[:10])]
            assert all(abs(per[qid][m]-v)<1e-9 for m,v in zip(METRICS,expected))
            rows.append({'system':system,'query_id':qid,**per[qid]})
        values[system]=np.array([[per[q][m] for m in METRICS] for q in qids])
        summaries.append({'system':system,'configuration':NAMES['en'][idx],**dict(zip(METRICS,values[system].mean(axis=0).tolist()))})
    rng=np.random.default_rng(42);draws=rng.integers(0,40,(10000,40));paired=[]
    pairs=[(2,0),(2,1),(3,2),(4,3),(5,4),(6,5),(7,6),(7,4)]
    for new,base in pairs:
        d=values[IDS[new]]-values[IDS[base]];low,high=np.quantile(d[draws].mean(axis=1),[.025,.975],axis=0)
        for i,m in enumerate(METRICS):paired.append({'system':IDS[new],'baseline':IDS[base],'metric':m,
            'delta':float(d[:,i].mean()),'ci95_low':float(low[i]),'ci95_high':float(high[i]),
            'wins':int((d[:,i]>1e-10).sum()),'ties':int((abs(d[:,i])<=1e-10).sum()),'losses':int((d[:,i]<-1e-10).sum())})
    groups=defaultdict(list)
    for i,q in enumerate(queries):groups[q['aspects'][0] if q['query_type']=='single_aspect' else q['query_type']].append(i)
    grouped=[{'system':system,'group':group,'queries':len(indices),**dict(zip(METRICS,values[system][indices].mean(axis=0).tolist()))}
             for group,indices in groups.items() for system in IDS]
    stats={'date':'2026-10-29','queries':40,'systems':8,'judged_pairs':len(qrels),
      'metrics':METRICS,'summary':summaries,'paired_deltas':paired,'by_query_group':grouped,
      'training':json.loads((TRAIN/'training_statistics.json').read_text()),
      'provenance':{'qrels_sha256':sha(DATA/'qrels.trec'),'inputs_sha256':sha(OUT/'inputs.jsonl'),
                    'model_sha256':sha(ROOT/'retrieval/models/review_ranker_v2.json'),
                    'ir_measures_version':ir.__version__,'metric_checks':1280,
                    'retrieval_rerun':run_manifest['retrieval_rerun'],'minilm_inference_rerun':run_manifest['minilm_inference_rerun'],
                    'evaluation':'Eight configurations share one canonical 2640-pair qrels. Candidate text is stored once; retrieval inputs and frozen MiniLM logits reproduce all ranks. Configurations 6/7 use rank fusion and 8 uses the trained model, all with a 20-candidate window.'}}
    for name,data in [('summary.csv',summaries),('per_query.csv',rows),('paired_deltas.csv',paired),('by_query_group.csv',grouped)]:csv_file(OUT/name,data)
    (OUT/'statistics.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    return stats

STYLE='''
:root{--ink:#1c2d3a;--muted:#667480;--line:#dce4e9;--accent:#176482;--soft:#f2f6f8;--pale:#edf5f8}*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#fff;color:var(--ink);font:15px/1.75 Inter,"Segoe UI","Microsoft YaHei",sans-serif}main{max-width:1320px;margin:auto;padding:36px 40px 52px}header{border-bottom:1px solid var(--line);padding-bottom:24px}nav{display:flex;align-items:center;justify-content:space-between;color:var(--muted);font-size:13px;margin-bottom:24px}a{color:var(--accent);text-underline-offset:4px}h1{font-size:29px;letter-spacing:-.7px;line-height:1.35;margin:0 0 10px}h2{font-size:20px;line-height:1.5;margin:0}h3{font-size:15px;margin:14px 0 5px}p{margin:9px 0}.meta,.note,footer{font-size:13px;color:var(--muted)}.eyebrow{font-size:12px;letter-spacing:1.5px;color:var(--accent);margin-bottom:10px}.kpis{display:grid;grid-template-columns:repeat(4,1fr);padding-top:22px;gap:20px}.kpis>div{border-right:1px solid var(--line);padding-right:16px}.kpis>div:last-child{border:0}.kpis strong{display:block;font-size:28px;font-weight:650;font-variant-numeric:tabular-nums;line-height:1.3}.kpis span{font-size:12px;color:var(--muted)}.lead{background:var(--soft);border-left:3px solid var(--accent);padding:15px 21px;margin-top:25px}.section{margin-top:38px}.section-head{display:flex;align-items:baseline;gap:12px;margin:0 0 16px}.num{color:var(--accent);font-size:12px;letter-spacing:1px}.scroll{width:100%;overflow-x:auto;border-top:1px solid var(--line)}table{width:100%;border-collapse:collapse;table-layout:fixed;font-variant-numeric:tabular-nums}th,td{padding:15px 16px;border-bottom:1px solid var(--line);text-align:right;vertical-align:middle;overflow-wrap:anywhere}thead th{background:var(--soft);color:var(--muted);font-size:12px;font-weight:600;line-height:1.6}tbody th{font-weight:500}th:first-child,td:first-child{text-align:left}tbody tr:nth-child(even){background:#fafcfd}tbody tr.latest{background:var(--pale)}tbody tr:hover{background:#f0f5f7}.metrics th:first-child{width:38%}.metrics td{font-size:16px}.best{color:var(--accent);font-weight:750;text-decoration:underline;text-underline-offset:5px}.two-col th:first-child{width:32%}.two-col td{text-align:left;line-height:1.8}.pairs th:first-child{width:32%}.pairs th:nth-child(3){width:28%}.groups th:first-child{width:25%}.controls{display:flex;gap:24px;align-items:center;flex-wrap:wrap;margin:16px 0}select{font:inherit;font-size:14px;border:1px solid var(--line);padding:7px 12px;background:#fff;color:var(--ink);margin-left:9px;border-radius:3px}.bar-row{display:grid;grid-template-columns:280px 1fr 70px;gap:16px;align-items:center;margin:12px 0;font-size:13px}.track{height:23px;background:var(--soft)}.fill{height:100%;background:#a9bbc5}.fill.new{background:var(--accent)}.bar-value{text-align:right;font-variant-numeric:tabular-nums}.axis{display:flex;justify-content:space-between;color:var(--muted);font-size:11px;margin:4px 86px 0 296px}.protocol{display:grid;grid-template-columns:1fr 1fr;gap:30px}.protocol>div{border-top:2px solid var(--line);padding-top:12px}.formula{font-family:Consolas,monospace;background:var(--soft);padding:12px 16px;overflow:auto;font-size:13px}.pill{font-size:11px;background:white;border:1px solid var(--line);padding:2px 7px;white-space:nowrap;margin-left:8px}ul{padding-left:20px}li{margin:7px 0}.downloads{display:flex;gap:12px;flex-wrap:wrap;margin-top:18px}.downloads a{border:1px solid var(--line);padding:7px 14px;text-decoration:none;font-size:13px}footer{border-top:1px solid var(--line);margin-top:38px;padding-top:18px}@media(max-width:800px){.kpis{grid-template-columns:repeat(2,minmax(0,1fr))}.kpis>div:nth-child(2){border:0}main{padding:22px 18px}.kpis{gap:12px}.kpis strong{font-size:22px}.protocol{grid-template-columns:1fr;gap:15px}h1{font-size:24px}.bar-row{grid-template-columns:150px 1fr 55px;gap:10px}.axis{margin-left:160px;margin-right:65px}table{min-width:660px}th,td{padding:12px 10px}.section-head{align-items:center}}@media print{main{max-width:none;padding:12px;font-size:11px}nav,.controls,.downloads{display:none}.scroll{overflow:visible}table{min-width:0}th,td{padding:8px}h2{break-after:avoid}tr,.lead{break-inside:avoid}a{color:inherit}.section{margin-top:22px}.kpis strong{font-size:22px}}
'''

def render(s,lang):
    zh=lang=='zh';names=NAMES[lang];local=lambda a,b:a if zh else b
    best={m:max(r[m] for r in s['summary']) for m in METRICS}
    latest=s['summary'][-1];base=s['summary'][-2]
    metric_labels=['P@10','pooled Recall@100','MRR@10','nDCG@10']
    rows=''.join('<tr'+(' class="latest"' if i==7 else '')+'><th scope="row">'+names[i]+'</th>'+''.join(
      f'<td class="{"best" if abs(r[m]-best[m])<1e-10 else ""}">{r[m]:.4f}</td>' for m in METRICS)+'</tr>' for i,r in enumerate(s['summary']))
    descriptions=local([
      '按产品过滤；使用原查询，只进行关键词检索。','同样的产品过滤与原查询，使用 BGE 向量检索。',
      '融合 BM25 与向量排名；查询保留型号。','先确定目标产品并过滤，再从检索文本删除型号。',
      '在④基础上，对命中目标方面的候选增加召回奖励。','在⑤基础上，用召回排名与MiniLM语义排名融合重排前20条，后80条保留召回顺序。',
      '增加缓存的 DeepSeek 查询改写、产品与方面解析；使用与⑥相同的排名融合算法。',
      '复用⑦的Top100；由100题训练的模型评分并重排前20条。'],[
      'Product filtering; lexical retrieval with the original query.','Same filters and original query; BGE dense retrieval.',
      'Fuse BM25 and dense ranks; retain the model name in the query.','Resolve and filter the product, then remove its name from search text.',
      'Add an aspect reward to configuration 4 during retrieval.','Fuse retrieval rank and MiniLM semantic rank within the first 20; preserve the remaining 80 positions.',
      'Add cached DeepSeek rewriting and product/aspect extraction; use the same rank fusion algorithm as configuration 6.',
      'Reuse configuration 7’s Top100; score and reorder the first 20 with the model trained on 100 separate queries.'])
    config_rows=''.join(f'<tr><th scope="row">{names[i]}</th><td>{text}</td></tr>' for i,text in enumerate(descriptions))
    comparisons=[]
    for r in s['paired_deltas']:
        if r['metric']!='nDCG@10':continue
        comparisons.append(f'<tr><th>{IDS.index(r["system"])+1} {local("对","vs")} {IDS.index(r["baseline"])+1}</th><td>{r["delta"]:+.4f}</td><td>[{r["ci95_low"]:+.4f}, {r["ci95_high"]:+.4f}]</td><td>{r["wins"]} / {r["ties"]} / {r["losses"]}</td></tr>')
    group_rows=''
    by={(r['system'],r['group']):r for r in s['by_query_group']}
    for group,labels in GROUPS.items():
        a,b=by[IDS[6],group],by[IDS[7],group]
        group_rows+=f'<tr><th>{labels[0 if zh else 1]}</th><td>{b["queries"]}</td><td>{a["P@10"]:.4f}</td><td>{b["P@10"]:.4f}</td><td>{a["nDCG@10"]:.4f}</td><td>{b["nDCG@10"]:.4f}</td></tr>'
    pair=next(r for r in s['paired_deltas'] if r['system']==IDS[7] and r['baseline']==IDS[6] and r['metric']=='nDCG@10')
    sections=[]
    def section(n,title,body):sections.append(f'<section class="section"><div class="section-head"><span class="num">{n:02d}</span><h2>{title}</h2></div>{body}</section>')
    def tab(headers,body,cls):return '<div class="scroll"><table class="'+cls+'"><thead><tr>'+''.join('<th scope="col">'+h+'</th>' for h in headers)+'</tr></thead><tbody>'+body+'</tbody></table></div>'
    section(1,local('八组配置 · 总体结果','Eight configurations · Overall results'),
      '<p class="note">'+local('统一使用40题与同一份标注，数值为宏平均，越高越好。下划线标出每列最高值；浅蓝行为训练模型配置。','All values are macro-averages over the same 40 queries and common qrels. Higher is better. Underlined values are column bests; the blue row uses the trained model.')+'</p>'+tab([local('实验配置','Configuration'),*metric_labels],rows,'metrics'))
    options=''.join(f'<option value="{g}">{v[0 if zh else 1]}</option>' for g,v in GROUPS.items())
    section(2,local('按指标与问题类别查看','Explore metrics and question groups'),
      f'<div class="controls"><label>{local("指标","Metric")}<select id="metric">'+''.join(f'<option value="{m}">{m.replace("_"," ")}</option>' for m in ['nDCG@10','P@10','pooled_Recall@100','MRR@10'])+f'</select></label><label>{local("问题类别","Question group")}<select id="group"><option value="all">{local("全部40题","All 40 queries")}</option>{options}</select></label></div><div id="chart"></div><div class="axis"><span>0</span><span>0.25</span><span>0.50</span><span>0.75</span><span>1.00</span></div><p class="note">'+local('横轴：指标数值（0–1）；纵轴：实验配置。灰色为对照，蓝色为训练模型配置。切换分类仅改变图表，不改变上方总体表。','X axis: metric value (0–1). Y axis: configuration. Gray denotes controls; blue denotes the trained model. Group selection changes the chart only, not the overall table above.')+'</p>')
    section(3,local('每一步改动的作用','What each change contributes'),tab(
      [local('配置比较','Comparison'),'Δ nDCG@10',local('配对95%区间','Paired 95% interval'),local('胜 / 平 / 负','Wins / ties / losses')],''.join(comparisons),'pairs')+
      '<p class="note">'+local('按问题进行10,000次配对bootstrap，种子42；未做多重比较校正。区间反映这40题的变化，不代表已经证明对新问题的泛化优势。','10,000 paired query bootstrap resamples, seed 42; no multiple-comparison adjustment. Intervals describe these 40 queries and do not establish generalization to unseen queries.')+'</p>')
    fusion=s['summary'][5];retrieval=s['summary'][4]
    findings=local([
      '产品过滤后去掉型号是主要的检索改进；方面奖励在此基础上继续提高靠前排序。',
      f'⑥对⑤增加排名融合重排序，P@10变化{fusion["P@10"]-retrieval["P@10"]:+.4f}，nDCG@10变化{fusion["nDCG@10"]-retrieval["nDCG@10"]:+.4f}。⑦再加入查询改写和解析，nDCG@10变化{base["nDCG@10"]-fusion["nDCG@10"]:+.4f}。',
      f'⑧使用训练模型，相较⑦的排名融合算法，P@10变化{latest["P@10"]-base["P@10"]:+.4f}，nDCG@10变化{latest["nDCG@10"]-base["nDCG@10"]:+.4f}；逐题nDCG为{pair["wins"]}胜、{pair["ties"]}平、{pair["losses"]}负。',
      '⑤⑥、⑦⑧分别使用相同Top100集合，因此对应的Recall@100不变。⑥⑦⑧均只重排前20条，语义查询均在产品过滤后去掉型号。'],[
      'Removing model names after product filtering is the main retrieval improvement; aspect boosting adds further early-ranking gains.',
      f'Rank fusion reranking in 6 versus 5 changes P@10 by {fusion["P@10"]-retrieval["P@10"]:+.4f} and nDCG@10 by {fusion["nDCG@10"]-retrieval["nDCG@10"]:+.4f}. Rewriting and parsing in 7 versus 6 change nDCG@10 by {base["nDCG@10"]-fusion["nDCG@10"]:+.4f}.',
      f'The trained model in configuration 8 versus rank fusion in 7 changes P@10 by {latest["P@10"]-base["P@10"]:+.4f} and nDCG@10 by {latest["nDCG@10"]-base["nDCG@10"]:+.4f}; per-query nDCG records {pair["wins"]} wins, {pair["ties"]} ties and {pair["losses"]} losses.',
      'Configurations 5/6 and 7/8 respectively share Top100 membership, so Recall@100 is unchanged. Configurations 6–8 all rerank the first 20 and remove model names from semantic queries after filtering.'])
    section(4,local('结果解读','Interpretation'),'<ul>'+''.join('<li>'+f+'</li>' for f in findings)+'</ul>')
    section(5,local('各类问题 · ⑦与⑧对比','Question groups · Configuration 7 versus 8'),tab(
      [local('类别','Group'),local('题数','Queries'),'⑦ P@10','⑧ P@10','⑦ nDCG@10','⑧ nDCG@10'],group_rows,'groups')+
      '<p class="note">'+local('六个单方面组各6题，多方面和产品比较各2题；小样本分类结果仅作描述。','Six single-aspect groups contain six queries each; multi-aspect and comparison groups contain two each. Subgroup results are descriptive.')+'</p>')
    section(6,local('配置定义与比较关系','Configuration definitions'),tab([local('配置','Configuration'),local('设置与改动','Settings and changes')],config_rows,'two-col'))
    cv=s['training']['nested_product_group_cv']
    section(7,local('第⑧组模型训练','Configuration 8 · Model training'),local(
      f'<p><strong>100个新问题 × 20个固定召回候选 = 2,000对标注。</strong>覆盖16款机型，与原40题的问题、目标机型、chunk、评论及完全相同正文无交集。原40题不参与本次训练或选参。</p><p>MiniLM保持冻结，只训练带ridge正则的线性融合层。按机型分组进行五折验证，从32种配置中选参；同一机型不会跨训练折和验证折。最终使用7项标准化特征：MiniLM原始分数、语义倒数排名、召回倒数排名、方面覆盖率、关键词对数数量、关键词密度、文本对数长度。正则系数为100，训练目标为问题内中心化后的0–3分标签。</p><div class="formula">score = Σ wᵢ × zᵢ &nbsp; · &nbsp; zᵢ = (xᵢ − mean(xᵢ)) / max(std(xᵢ), 10⁻⁶)</div><p>训练100题按机型分组的嵌套验证：P@10={cv[0]:.4f}，MRR@10={cv[1]:.4f}，nDCG@10={cv[2]:.4f}。这是训练数据内部交叉验证，不是本报告40题结果。</p>',
      f'<p><strong>100 new queries × 20 fixed retrieval candidates = 2,000 judgments.</strong> The 16 training products, query texts, chunks, source reviews and exact text are disjoint from evaluation40. The 40 evaluation queries were not used for this fit or parameter selection.</p><p>MiniLM remains frozen; only a ridge-regularized linear fusion layer is fitted. Five-fold product-group validation selects among 32 settings, keeping each product entirely within one fold. The selected seven standardized features are raw MiniLM score, semantic reciprocal rank, retrieval reciprocal rank, aspect coverage, log keyword count, keyword density and log text length. The ridge penalty is 100; targets are query-centered grades 0–3.</p><div class="formula">score = Σ wᵢ × zᵢ &nbsp; · &nbsp; zᵢ = (xᵢ − mean(xᵢ)) / max(std(xᵢ), 10⁻⁶)</div><p>Nested product-group validation within training100 gives P@10={cv[0]:.4f}, MRR@10={cv[1]:.4f} and nDCG@10={cv[2]:.4f}. These are internal training-set validation values, separate from the 40-query report.</p>'))
    section(8,local('评测口径与数据范围','Evaluation protocol and scope'),local(
      f'<div class="protocol"><div><h3>完整索引与候选</h3><p>459,418条chunk的共享索引；每路有效检索深度300，每产品保留100条，同一原评论最多1条。⑦⑧合并多产品候选并保留全局Top100。⑥⑦⑧均重排前20条并保留21–100名的召回顺序，语义查询均在产品过滤后删除型号。⑥⑦使用 score = 0.2/(10+召回名次) + 0.8/(10+语义名次)；⑧使用训练模型。</p><h3>统一标注</h3><p>统一评测集包含{s["judged_pairs"]:,}对。八组四项指标统一使用同一份qrels。Top10全部标注；11–100名不强制补标，Recall只计命中的已知正相关项。补标不参与模型训练或选参。</p></div><div><h3>四项指标</h3><p>P@10：前10条相关数量 / 10。Recall@100：前100条命中的已知正相关数量 / 该题全部已知正相关数量。MRR@10：前10条首个相关结果名次的倒数。P、Recall、MRR以标签≥2为相关。</p><p>nDCG@10使用0、1、2、3线性增益，DCG = Σ grade / log₂(rank+1)，IDCG来自该题全标注池的最佳10条。每题计算后取40题宏平均。</p><h3>结果来源</h3><p>使用集中存储的候选正文、检索快照和冻结MiniLM分数复现八组排序；⑥⑦为排名融合，⑧为训练模型。八组统一重算指标，未重跑检索或调用改写API。1,280个逐题指标均通过独立算式核对。</p></div></div><p class="note">这是累计式消融，不是全因子实验；①②为并列基线。第⑦组将改写与多产品解析一起引入。Recall基于不完整pooling标注，不是穷尽全库的召回率；本次统一标注后不能直接沿用旧报告的Recall数值。40题曾参与早期方案开发，仍需新的盲测题检验泛化。指标衡量证据检索，不衡量生成答案质量。</p>',
      f'<div class="protocol"><div><h3>Full index and candidates</h3><p>Shared index of 459,418 chunks; effective depth 300 per channel; retain 100 per product and at most one chunk per review. Configurations 7/8 merge products into a global Top100. Configurations 6–8 all rerank the first 20 and preserve ranks 21–100; all use model-stripped semantic queries. Configurations 6/7 use score = 0.2/(10+retrieval rank) + 0.8/(10+semantic rank); configuration 8 uses the trained model.</p><h3>Common judgments</h3><p>The canonical evaluation set contains {s["judged_pairs"]:,} judgments. All eight systems and all four metrics share these qrels. Every Top10 result is judged; ranks 11–100 do not require exhaustive labeling. Recall credits known positive hits only. Supplemental judgments were excluded from model fitting and selection.</p></div><div><h3>Four metrics</h3><p>P@10: relevant results in the first 10 / 10. Recall@100: known positives found in the first 100 / all known positives for that query. MRR@10: reciprocal rank of the first relevant result in the first 10. Grades ≥2 are binary relevant.</p><p>nDCG@10 uses linear gains 0, 1, 2, 3: DCG = Σ grade / log₂(rank+1). IDCG uses the ten best judgments for that query. Metrics are computed per query then macro-averaged over 40 queries.</p><h3>Result provenance</h3><p>Shared candidate text, retrieval snapshots and frozen MiniLM logits reproduce all eight rankings. Configurations 6/7 use rank fusion; 8 uses the trained model. All metric sets are recomputed without retrieval or rewrite API calls. All 1,280 per-query metric values pass independent arithmetic checks.</p></div></div><p class="note">This is cumulative ablation, not a factorial experiment; configurations 1/2 are parallel baselines. Stage 7 bundles rewriting and multi-product parsing. Recall uses incomplete pooled judgments, not exhaustive corpus relevance. The common qrels differ from historical reports, so their Recall values cannot be carried over. Evaluation40 influenced earlier development and is not a historically untouched blind test. Metrics measure evidence retrieval, not generated answer quality.</p>'))
    payload={'summary':s['summary'],'groups':s['by_query_group'],'names':dict(zip(IDS,names))}
    script='''const D=__DATA__;function draw(){const m=document.getElementById('metric').value,g=document.getElementById('group').value;const rows=g==='all'?D.summary:D.groups.filter(r=>r.group===g);document.getElementById('chart').innerHTML=rows.map((r,i)=>`<div class="bar-row"><span>${D.names[r.system]}</span><div class="track"><div class="fill ${i===7?'new':''}" style="width:${r[m]*100}%"></div></div><span class="bar-value">${r[m].toFixed(4)}</span></div>`).join('')}document.getElementById('metric').onchange=draw;document.getElementById('group').onchange=draw;draw();'''.replace('__DATA__',json.dumps(payload,ensure_ascii=False))
    title=local('全库检索消融实验报告','Full-Corpus Retrieval Ablation Report')
    nav=local('<strong>实验结果</strong><span>中文 · <a href="report_en.html">English</a></span>','<strong>EXPERIMENT RESULTS</strong><span><a href="report_zh.html">中文</a> · English</span>')
    hero=local('100题专用训练 · 40题统一评测 · 八组配置','100 dedicated training queries · 40 evaluation queries · Eight configurations')
    lead=local(f'<strong>八组配置比较检索改进、重排序算法和训练模型的作用。</strong><p>P@10={latest["P@10"]:.4f}，nDCG@10={latest["nDCG@10"]:.4f}。训练与本次评测问题分开；八组使用统一标注。</p>',f'<strong>Eight configurations compare retrieval improvements, reranking and the trained model.</strong><p>P@10={latest["P@10"]:.4f}; nDCG@10={latest["nDCG@10"]:.4f}. The model is fitted on separate training queries; all eight configurations use common judgments.</p>')
    kpis=''.join(f'<div><strong>{latest[m]:.4f}</strong><span>{label}</span></div>' for m,label in zip(METRICS,metric_labels))
    links=local(['总体指标 CSV','逐题结果 CSV','配对差值 CSV','完整统计 JSON'],['Overall metrics CSV','Per-query CSV','Paired differences CSV','Full statistics JSON'])
    downloads='<div class="downloads">'+''.join(f'<a href="{path}">{label}</a>' for path,label in zip(['summary.csv','per_query.csv','paired_deltas.csv','statistics.json'],links))+'</div>'
    footer=local(f'来源：固定40题、共享全库索引、训练100题模型。统计工具 ir_measures {ir.__version__}。数据与模型哈希见 execution_manifest.json。',f'Sources: fixed evaluation40, shared full-corpus indexes and the training100 model. Metrics: ir_measures {ir.__version__}. Data and model hashes: execution_manifest.json.')
    page=f'<!doctype html><html lang="{"zh-CN" if zh else "en"}"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title><style>{STYLE}</style></head><body><main><nav>{nav}</nav><header><div class="eyebrow">RETRIEVAL ABLATION · 2026.10.29</div><h1>{title}</h1><p class="meta">{hero}</p><div class="kpis">{kpis}</div></header><div class="lead">{lead}</div>'+''.join(sections)+downloads+f'<footer>{footer}</footer></main><script>{script}</script></body></html>'
    (OUT/f'report_{lang}.html').write_text(page,encoding='utf8')

def package(s):
    sources=['scripts/run_retrieval_ablation.py','scripts/report_latest_ablation.py','scripts/export_evaluation_dataset.py',
             'scripts/train_ranker_100.py','retrieval/rerank.py','retrieval/learned_rank.py','pyproject.toml','uv.lock']
    manifest={'generated_utc':datetime.now(timezone.utc).isoformat(),'queries':40,'systems':8,'judged_pairs':s['judged_pairs'],
      'metric_checks':1280,'model_sha256':sha(ROOT/'retrieval/models/review_ranker_v2.json'),
      'dataset_manifest_sha256':sha(DATA/'manifest.json'),
      'source_sha256':{name:sha(ROOT/name) for name in sources},
      'files':{p.relative_to(OUT).as_posix():sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='execution_manifest.json'}}
    (OUT/'execution_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8',newline='\n')


def main():
    s=calculate()
    for lang in ['zh','en']:render(s,lang)
    package(s)
    print(json.dumps({'summary':s['summary'],'report':str(OUT),'qrels':s['judged_pairs']},ensure_ascii=False,indent=2))

if __name__=='__main__':main()

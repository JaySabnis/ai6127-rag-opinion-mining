"""Recompute the eight saved runs under common qrels; emit bilingual HTML."""
import csv
import hashlib
import html
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import zipfile
import ir_measures as ir
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/ablation_latest'
OLD=ROOT/'results/ablation_8'
TRAIN=ROOT/'results/reranker_training_100'
IDS=['01_bm25','02_dense','03_rrf','04_strip_model','05_aspect','06_rerank','07_rewrite','08_learned_v2']
METRICS=['P@10','pooled_Recall@100','MRR@10','nDCG@10']
NAMES={
 'zh':['① 仅 BM25','② 仅向量检索','③ 混合检索 · RRF','④ 产品过滤后去掉型号','⑤ 增加召回方面加分','⑥ 增加旧版重排序','⑦ 增加查询改写','⑧ 学习型重排序 · 100题训练'],
 'en':['1 · BM25 only','2 · Dense only','3 · Hybrid retrieval · RRF','4 · Remove model after filtering','5 · Add retrieval aspect boost','6 · Add legacy reranking','7 · Add query rewriting','8 · Learned reranker · 100 training queries']}
GROUPS={'camera':('相机','Camera'),'battery':('电池','Battery'),'display':('屏幕','Display'),
 'performance':('性能','Performance'),'service':('售后服务','Service'),'carrier':('运营商','Carrier'),
 'multi_aspect':('多方面','Multiple aspects'),'comparison':('产品比较','Product comparison')}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def csv_file(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def calculate():
    OUT.mkdir(exist_ok=True);(OUT/'runs').mkdir(exist_ok=True)
    qrels=list(ir.read_trec_qrels(str(TRAIN/'evaluation_qrels.trec')))
    grades=defaultdict(dict)
    for r in qrels:grades[r.query_id][r.doc_id]=r.relevance
    queries=[json.loads(s) for s in (ROOT/'evaluation/retrieval_ablation_40_v2/queries.jsonl').read_text(encoding='utf8').splitlines()]
    qids=[q['query_id'] for q in queries]
    measures=[ir.P(rel=2)@10,ir.Recall(rel=2)@100,ir.RR(rel=2)@10,ir.nDCG(gains={0:0,1:1,2:2,3:3},dcg='log2')@10]
    values={};rows=[];summaries=[]
    for idx,system in enumerate(IDS):
        src=(TRAIN/'runs/learned_v2.trec') if idx==7 else OLD/'runs'/f'{system}.trec'
        dest=OUT/'runs'/f'{system}.trec';dest.write_bytes(src.read_bytes())
        run=list(ir.read_trec_run(str(dest)));assert len(run)==4000
        per={q:{} for q in qids}
        assert {r.query_id for r in run}==set(qids)
        for r in ir.iter_calc(measures,qrels,run):per[r.query_id][METRICS[measures.index(r.measure)]]=r.value
        for qid in qids:
            ranked=sorted((r for r in run if r.query_id==qid),key=lambda r:r.score,reverse=True)
            assert len(ranked)==len({r.doc_id for r in ranked})==100
            assert all(r.doc_id in grades[qid] for r in ranked[:10])
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
      'previous_model_comparison':json.loads((TRAIN/'evaluation40.json').read_text()),
      'provenance':{'qrels_sha256':sha(TRAIN/'evaluation_qrels.trec'),
                    'model_sha256':sha(ROOT/'retrieval/models/review_ranker_v2.json'),
                    'ir_measures_version':ir.__version__,'metric_checks':1280,
                    'retrieval_rerun':False,'evaluation':'Saved full-index Top100 runs reused; new100-trained reranker replaces stage8. All four metrics recomputed against one shared qrels, v2+6 supplemental judgments.'}}
    for name,data in [('summary.csv',summaries),('per_query.csv',rows),('paired_deltas.csv',paired),('by_query_group.csv',grouped)]:csv_file(OUT/name,data)
    (OUT/'statistics.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    (OUT/'qrels.trec').write_bytes((TRAIN/'evaluation_qrels.trec').read_bytes())
    (OUT/'queries.jsonl').write_bytes((ROOT/'evaluation/retrieval_ablation_40_v2/queries.jsonl').read_bytes())
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
      '在④基础上，对命中目标方面的候选增加召回奖励。','在⑤基础上，用旧版 MiniLM / RRF / 方面公式重排100条。',
      '增加缓存的 DeepSeek 查询改写、产品与方面解析，仍用旧版重排。',
      '复用⑦的Top100；改用100题专用训练的融合模型，只重排前20条。'],[
      'Product filtering; lexical retrieval with the original query.','Same filters and original query; BGE dense retrieval.',
      'Fuse BM25 and dense ranks; retain the model name in the query.','Resolve and filter the product, then remove its name from search text.',
      'Add an aspect reward to configuration 4 during retrieval.','Add the legacy MiniLM / RRF / aspect formula; rerank all 100 candidates.',
      'Add cached DeepSeek rewriting, product and aspect extraction; retain legacy reranking.',
      'Reuse configuration 7’s Top100; use the fusion model trained on 100 separate queries, reranking only the first 20.'])
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
      '<p class="note">'+local('统一使用40题与同一份标注，数值为宏平均，越高越好。下划线标出每列最高值；浅蓝行为最新版。','All values are macro-averages over the same 40 queries and common qrels. Higher is better. Underlined values are column bests; the blue row is the latest model.')+'</p>'+tab([local('实验配置','Configuration'),*metric_labels],rows,'metrics'))
    options=''.join(f'<option value="{g}">{v[0 if zh else 1]}</option>' for g,v in GROUPS.items())
    section(2,local('按指标与问题类别查看','Explore metrics and question groups'),
      f'<div class="controls"><label>{local("指标","Metric")}<select id="metric">'+''.join(f'<option value="{m}">{m.replace("_"," ")}</option>' for m in ['nDCG@10','P@10','pooled_Recall@100','MRR@10'])+f'</select></label><label>{local("问题类别","Question group")}<select id="group"><option value="all">{local("全部40题","All 40 queries")}</option>{options}</select></label></div><div id="chart"></div><div class="axis"><span>0</span><span>0.25</span><span>0.50</span><span>0.75</span><span>1.00</span></div><p class="note">'+local('横轴：指标数值（0–1）；纵轴：实验配置。灰色为对照，蓝色为最新版。切换分类仅改变图表，不改变上方总体表。','X axis: metric value (0–1). Y axis: configuration. Gray denotes controls; blue denotes the latest model. Group selection changes the chart only, not the overall table above.')+'</p>')
    section(3,local('每一步改动的作用','What each change contributes'),tab(
      [local('配置比较','Comparison'),'Δ nDCG@10',local('配对95%区间','Paired 95% interval'),local('胜 / 平 / 负','Wins / ties / losses')],''.join(comparisons),'pairs')+
      '<p class="note">'+local('按问题进行10,000次配对bootstrap，种子42；未做多重比较校正。区间反映这40题的变化，不代表已经证明对新问题的泛化优势。','10,000 paired query bootstrap resamples, seed 42; no multiple-comparison adjustment. Intervals describe these 40 queries and do not establish generalization to unseen queries.')+'</p>')
    findings=local([
      '产品过滤后去掉型号是主要的检索改进；方面奖励在此基础上继续提高靠前排序。',
      f'最新版相较⑦组，P@10提高{latest["P@10"]-base["P@10"]:.4f}，nDCG@10提高{latest["nDCG@10"]-base["nDCG@10"]:.4f}；逐题nDCG为{pair["wins"]}胜、{pair["ties"]}平、{pair["losses"]}负。',
      '⑤⑥、⑦⑧分别使用相同的Top100集合，重排只改变顺序，因此对应的Recall@100不变。',
      '旧版重排并未带来稳定收益；RRF在保留型号时也没有优于纯向量的靠前排序，不能声称融合始终更好。'],[
      'Removing the model name after product filtering is the main retrieval improvement; aspect boosting adds further early-ranking gains.',
      f'The latest model improves P@10 by {latest["P@10"]-base["P@10"]:.4f} and nDCG@10 by {latest["nDCG@10"]-base["nDCG@10"]:.4f} over configuration 7; per-query nDCG records {pair["wins"]} wins, {pair["ties"]} ties and {pair["losses"]} losses.',
      'Configurations 5/6 and 7/8 respectively share the same Top100 membership. Reordering alone therefore leaves Recall@100 unchanged.',
      'Legacy reranking does not show a stable gain. With model names retained, RRF also trails dense-only early ranking; fusion is not universally better.'])
    section(4,local('结果解读','Interpretation'),'<ul>'+''.join('<li>'+f+'</li>' for f in findings)+'</ul>')
    section(5,local('各类问题 · ⑦与⑧对比','Question groups · Configuration 7 versus 8'),tab(
      [local('类别','Group'),local('题数','Queries'),'⑦ P@10','⑧ P@10','⑦ nDCG@10','⑧ nDCG@10'],group_rows,'groups')+
      '<p class="note">'+local('六个单方面组各6题，多方面和产品比较各2题；小样本分类结果仅作描述。','Six single-aspect groups contain six queries each; multi-aspect and comparison groups contain two each. Subgroup results are descriptive.')+'</p>')
    section(6,local('配置定义与比较关系','Configuration definitions'),tab([local('配置','Configuration'),local('设置与改动','Settings and changes')],config_rows,'two-col'))
    cv=s['training']['nested_product_group_cv'];old=s['previous_model_comparison']['summary']['learned_v1'];new=s['previous_model_comparison']['summary']['learned_v2']
    section(7,local('新版模型如何训练','How the latest model was trained'),local(
      f'<p><strong>100个新问题 × 20个固定召回候选 = 2,000对标注。</strong>覆盖16款机型，与原40题的问题、目标机型、chunk、评论及完全相同正文无交集。原40题不参与本次训练或选参。</p><p>MiniLM保持冻结，只训练带ridge正则的线性融合层。按机型分组进行五折验证，从32种配置中选参；同一机型不会跨训练折和验证折。最终使用7项标准化特征：MiniLM原始分数、语义倒数排名、召回倒数排名、方面覆盖率、关键词对数数量、关键词密度、文本对数长度。正则系数为100，训练目标为问题内中心化后的0–3分标签。</p><div class="formula">score = Σ wᵢ × zᵢ &nbsp; · &nbsp; zᵢ = (xᵢ − mean(xᵢ)) / max(std(xᵢ), 10⁻⁶)</div><p>训练100题按机型分组的嵌套验证：P@10={cv[0]:.4f}，MRR@10={cv[1]:.4f}，nDCG@10={cv[2]:.4f}。这是训练数据内部交叉验证，不是本报告40题结果。</p><p class="note">旧的40题训练版模型在这40题上nDCG为{old[3]:.4f}；新100题训练版为{new[3]:.4f}，略低，但P@10从{old[0]:.4f}升至{new[0]:.4f}。旧版用过这40题训练，不能据此比较独立泛化能力。旧版仅作说明，不作为第九组配置。</p>',
      f'<p><strong>100 new queries × 20 fixed retrieval candidates = 2,000 judgments.</strong> The 16 training products, query texts, chunks, source reviews and exact text are disjoint from evaluation40. The 40 evaluation queries were not used for this fit or parameter selection.</p><p>MiniLM remains frozen; only a ridge-regularized linear fusion layer is fitted. Five-fold product-group validation selects among 32 settings, keeping each product entirely within one fold. The selected seven standardized features are raw MiniLM score, semantic reciprocal rank, retrieval reciprocal rank, aspect coverage, log keyword count, keyword density and log text length. The ridge penalty is 100; targets are query-centered grades 0–3.</p><div class="formula">score = Σ wᵢ × zᵢ &nbsp; · &nbsp; zᵢ = (xᵢ − mean(xᵢ)) / max(std(xᵢ), 10⁻⁶)</div><p>Nested product-group validation within training100 gives P@10={cv[0]:.4f}, MRR@10={cv[1]:.4f} and nDCG@10={cv[2]:.4f}. These are internal training-set validation values, separate from the 40-query report.</p><p class="note">The previous model trained on evaluation40 scored nDCG={old[3]:.4f} on those queries; the new training100 model scores {new[3]:.4f}, slightly lower, while P@10 increases from {old[0]:.4f} to {new[0]:.4f}. The earlier model used these queries in fitting, so this is not a comparison of independent generalization. It is context, not a ninth configuration.</p>'))
    section(8,local('评测口径与数据范围','Evaluation protocol and scope'),local(
      f'<div class="protocol"><div><h3>完整索引与候选</h3><p>459,418条chunk的共享索引；每路有效检索深度300，每产品保留100条，同一原评论最多1条。⑦⑧合并多产品候选并保留全局Top100。⑥⑦旧版重排100条，⑧只重排前20条；⑧对⑦同时改变语义查询、融合方法和窗口，不能将全部收益只归因于学习权重。</p><h3>统一标注</h3><p>原2,564对，加前次62对及本次检查阶段6对补标，共{s["judged_pairs"]:,}对。八组四项指标统一使用同一份qrels。Top10全部标注；11–100名不强制补标，Recall只计命中的已知正相关项。补标没有用于训练新模型。</p></div><div><h3>四项指标</h3><p>P@10：前10条相关数量 / 10。Recall@100：前100条命中的已知正相关数量 / 该题全部已知正相关数量。MRR@10：前10条首个相关结果名次的倒数。P、Recall、MRR以标签≥2为相关。</p><p>nDCG@10使用0、1、2、3线性增益，DCG = Σ grade / log₂(rank+1)，IDCG来自该题全标注池的最佳10条。每题计算后取40题宏平均。</p><h3>结果来源</h3><p>复用已保存的全库Top100与冻结MiniLM分数，替换最新版排序后，重新统一统计八组指标；本次制作报告未重跑检索、未调用改写API。1,280个逐题指标均通过独立算式核对。</p></div></div><p class="note">这是累计式消融，不是全因子实验；①②为并列基线。第⑦组将改写与多产品解析一起引入。Recall基于不完整pooling标注，不是穷尽全库的召回率；本次统一标注后不能直接沿用旧报告的Recall数值。40题曾参与早期方案开发，仍需新的盲测题检验泛化。指标衡量证据检索，不衡量生成答案质量。</p>',
      f'<div class="protocol"><div><h3>Full index and candidates</h3><p>Shared index of 459,418 chunks; effective depth 300 per channel; retain 100 per product and at most one chunk per review. Configurations 7/8 merge products into a global Top100. Legacy configurations 6/7 rerank 100 candidates; configuration 8 reranks 20. The replacement changes semantic query handling, fusion and window together, so its entire gain cannot be attributed to learned coefficients alone.</p><h3>Common judgments</h3><p>Original 2,564 pairs plus 62 earlier and six later Top10 supplements: {s["judged_pairs"]:,} judgments. All eight systems and all four metrics share these qrels. Every Top10 result is judged; ranks 11–100 do not require exhaustive labeling. Recall credits known positive hits only. Supplements were not used to train the new model.</p></div><div><h3>Four metrics</h3><p>P@10: relevant results in the first 10 / 10. Recall@100: known positives found in the first 100 / all known positives for that query. MRR@10: reciprocal rank of the first relevant result in the first 10. Grades ≥2 are binary relevant.</p><p>nDCG@10 uses linear gains 0, 1, 2, 3: DCG = Σ grade / log₂(rank+1). IDCG uses the ten best judgments for that query. Metrics are computed per query then macro-averaged over 40 queries.</p><h3>Result provenance</h3><p>Saved full-index Top100 runs and frozen MiniLM logits are reused. The latest reranking replaces configuration 8 and all eight metric sets are recomputed. Report preparation makes no retrieval or rewrite API calls. All 1,280 per-query metric values pass independent arithmetic checks.</p></div></div><p class="note">This is cumulative ablation, not a factorial experiment; configurations 1/2 are parallel baselines. Stage 7 bundles rewriting and multi-product parsing. Recall uses incomplete pooled judgments, not exhaustive corpus relevance. The common qrels differ from historical reports, so their Recall values cannot be carried over. Evaluation40 influenced earlier development and is not a historically untouched blind test. Metrics measure evidence retrieval, not generated answer quality.</p>'))
    payload={'summary':s['summary'],'groups':s['by_query_group'],'names':dict(zip(IDS,names))}
    script='''const D=__DATA__;function draw(){const m=document.getElementById('metric').value,g=document.getElementById('group').value;const rows=g==='all'?D.summary:D.groups.filter(r=>r.group===g);document.getElementById('chart').innerHTML=rows.map((r,i)=>`<div class="bar-row"><span>${D.names[r.system]}</span><div class="track"><div class="fill ${i===7?'new':''}" style="width:${r[m]*100}%"></div></div><span class="bar-value">${r[m].toFixed(4)}</span></div>`).join('')}document.getElementById('metric').onchange=draw;document.getElementById('group').onchange=draw;draw();'''.replace('__DATA__',json.dumps(payload,ensure_ascii=False))
    title=local('全库检索消融实验报告','Full-Corpus Retrieval Ablation Report')
    nav=local('<strong>实验结果 / 最新版</strong><span>中文 · <a href="report_en.html">English</a></span>','<strong>EXPERIMENT RESULTS / LATEST</strong><span><a href="report_zh.html">中文</a> · English</span>')
    hero=local('100题专用训练 · 40题统一评测 · 八组配置','100 dedicated training queries · 40 evaluation queries · Eight configurations')
    lead=local(f'<strong>新版学习型重排序取得八组中最高的靠前排序指标。</strong><p>P@10={latest["P@10"]:.4f}，nDCG@10={latest["nDCG@10"]:.4f}。训练与本次评测问题分开；Recall@100使用统一的新标注口径。</p>',f'<strong>The latest learned reranker leads the eight configurations on early-ranking metrics.</strong><p>P@10={latest["P@10"]:.4f}; nDCG@10={latest["nDCG@10"]:.4f}. This model is fitted on separate training queries; Recall@100 uses common, updated judgments.</p>')
    kpis=''.join(f'<div><strong>{latest[m]:.4f}</strong><span>{label}</span></div>' for m,label in zip(METRICS,metric_labels))
    links=local(['总体指标 CSV','逐题结果 CSV','配对差值 CSV','完整统计 JSON'],['Overall metrics CSV','Per-query CSV','Paired differences CSV','Full statistics JSON'])
    downloads='<div class="downloads">'+''.join(f'<a href="{path}">{label}</a>' for path,label in zip(['summary.csv','per_query.csv','paired_deltas.csv','statistics.json'],links))+'</div>'
    footer=local(f'来源：固定40题、共享全库索引、训练100题模型v2。统计工具 ir_measures {ir.__version__}。数据与模型哈希见 execution_manifest.json。',f'Sources: fixed evaluation40, shared full-corpus indexes and training100 model v2. Metrics: ir_measures {ir.__version__}. Data and model hashes: execution_manifest.json.')
    page=f'<!doctype html><html lang="{"zh-CN" if zh else "en"}"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title><style>{STYLE}</style></head><body><main><nav>{nav}</nav><header><div class="eyebrow">RETRIEVAL ABLATION · 2026.10.29</div><h1>{title}</h1><p class="meta">{hero}</p><div class="kpis">{kpis}</div></header><div class="lead">{lead}</div>'+''.join(sections)+downloads+f'<footer>{footer}</footer></main><script>{script}</script></body></html>'
    (OUT/f'report_{lang}.html').write_text(page,encoding='utf8')

def package(s):
    src=OUT/'source_snapshot';src.mkdir(exist_ok=True)
    (src/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    (OUT/'model.json').write_bytes((ROOT/'retrieval/models/review_ranker_v2.json').read_bytes())
    manifest={'generated_utc':datetime.now(timezone.utc).isoformat(),'queries':40,'systems':8,
      'judged_pairs':s['judged_pairs'],'model':'review_ranker_v2; trained only on 100 dedicated queries',
      'metric_checks':1280,'retrieval_rerun':False,
      'files':{p.relative_to(OUT).as_posix():sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='execution_manifest.json'}}
    (OUT/'execution_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')
    with zipfile.ZipFile(OUT.with_suffix('.zip'),'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file():z.write(p,OUT.name+'/'+p.relative_to(OUT).as_posix())
    with zipfile.ZipFile(OUT.with_suffix('.zip')) as z:
        assert z.testzip() is None
        for name,h in manifest['files'].items():assert hashlib.sha256(z.read(OUT.name+'/'+name)).hexdigest()==h

def main():
    s=calculate()
    for lang in ['zh','en']:render(s,lang)
    package(s)
    print(json.dumps({'summary':s['summary'],'report':str(OUT),'qrels':s['judged_pairs']},ensure_ascii=False,indent=2))

if __name__=='__main__':main()

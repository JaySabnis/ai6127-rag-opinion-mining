"""Build the English report from measured results, without rerunning retrieval."""
import json
import html
from pathlib import Path
import markdown

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/ablation_8'

def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,r))+' |' for r in rows])

ZH_NAMES = {
    '01_bm25':'① 仅 BM25','02_dense':'② 仅向量检索','03_rrf':'③ 混合检索 RRF',
    '04_strip_model':'④ 过滤产品后去掉型号','05_aspect':'⑤ 增加召回方面加分',
    '06_rerank':'⑥ 增加旧版重排序','07_rewrite':'⑦ 增加查询改写',
    '08_learned':'⑧ 替换为学习型重排序'}
ZH_GROUPS={'camera':'相机','battery':'电池','display':'屏幕','performance':'性能',
           'service':'售后服务','carrier':'运营商','multi_aspect':'多方面','comparison':'产品比较'}

def chinese_report(s,cv):
    metrics=['P@10','pooled_Recall@100','MRR@10','nDCG@10']
    results=table(['配置','P@10','pooled Recall@100','MRR@10','nDCG@10'],
        [[ZH_NAMES[r['system']],*[f'{r[m]:.4f}' for m in metrics]] for r in s['summary']])
    paired=table(['比较','nDCG@10 平均差值','配对 95% 区间','胜 / 平 / 负'],[
        [r['system'][:2]+' 对 '+r['baseline'][:2],f"{r['delta']:+.4f}",
         f"[{r['ci95_low']:+.4f}, {r['ci95_high']:+.4f}]",f"{r['wins']} / {r['ties']} / {r['losses']}"]
        for r in s['paired_deltas'] if r['metric']=='nDCG@10'])
    groups={r['group']:r for r in s['by_query_group'] if r['system']=='07_rewrite'}
    grouped=table(['问题类别','题数','⑦ P@10','⑧ P@10','⑦ nDCG@10','⑧ nDCG@10'],[
        [ZH_GROUPS[r['group']],r['queries'],f"{groups[r['group']]['P@10']:.4f}",f"{r['P@10']:.4f}",
         f"{groups[r['group']]['nDCG@10']:.4f}",f"{r['nDCG@10']:.4f}"]
        for r in s['by_query_group'] if r['system']=='08_learned'])
    return f'''# 检索消融实验报告：八组配置

实验日期：2026 年 10 月 29 日。40 个固定问题重新执行全部 8 组配置，共 320 次系统—问题运行，每次保存前 100 条结果。

## 主要结果

最新版配置取得 **P@10 = 0.8875、pooled Recall@100 = 0.7488、MRR@10 = 1.0000、nDCG@10 = 0.8628**。相较第⑦组，P@10 提高 0.0825，nDCG@10 提高 0.0987；40 题中 nDCG 提高 33 题、下降 7 题。400 个 Top-10 位置中的相关结果由 322 个增加至 355 个。MRR@10 为 1.0 表示每题的第一条结果都被标为相关，不代表全部结果都相关。

这些属于**开发集结果**。部署的学习型排序模型使用这 40 个问题训练，因此八组对比不能当成独立测试中的泛化能力证明。后文单独列出先前嵌套交叉验证的结果，其评测口径与本次运行不同。

## 八组配置

1. **仅 BM25：**按产品过滤，查询保留型号，不加方面奖励。
2. **仅向量检索：**保持同样的过滤和查询策略，使用 BGE 向量和共享 FAISS 索引。
3. **混合检索 RRF：**融合两路检索排名，查询仍保留型号。
4. **去掉型号：**先解析并过滤产品，再从检索文本中删除型号。
5. **召回方面加分：**在第④组基础上使用已有方面奖励。
6. **旧版重排序：**对第⑤组的 100 条候选使用旧版 MiniLM、RRF 和方面加分公式重新排序。
7. **查询改写：**检索前使用已缓存的 DeepSeek 改写，再使用旧版重排序。改写可能同时改变检索文本、产品解析和方面提取。
8. **最新版学习型重排序：**复用第⑦组同一批改写后的 Top-100 候选，将旧排序替换为学习型排序。

第①②组是并列基线，第③至⑦组按步骤累加模块，第⑧组替换第⑦组内部的重排序实现。⑧对⑦同时包含去型号后的语义查询、学习型特征和 20 条重排窗口等变化，不能将全部收益只归因于学习到的权重。

## 数据与检索设置

检索使用原有 459,418 条 chunk 的完整共享索引，没有重建索引。40 个固定问题包括 36 个单方面问题，相机、电池、屏幕、性能、售后服务、运营商各 6 题，另有 2 个多方面问题和 2 个产品比较问题。

原始 qrels 含 2,564 个问题—chunk 标注对，相关性为 0–3 分，其中 2、3 分视为二元相关。标注池并不穷尽全库相关内容；正面和负面评价都可以相关。

每次检索保留 100 个候选，同一原评论最多保留一个 chunk。去重模式下，召回深度为 max(candidate_k, 3 × k)，因此即使调用传入 candidate_k=150，**实际每路检索深度也是 300**。RRF 常数为 60，匹配任一目标方面的检索奖励为 0.5/(60+1)。

第①至⑥组使用规则 parser 解析的单个产品。第⑦⑧组使用缓存改写中解析的产品，每个产品召回最多 100 条，按 chunk ID 合并、按召回分数排序后保留全局前 100 条。因此比较题不强制两种产品等量出现；第⑦组同时引入了改写和多产品解析的变化。

学习型模型只重排前 20 条，后 80 条保持召回顺序；旧版模型对全部 100 条排序。窗口大小的变化是新实现的一部分。第⑤⑥组的 Top-100 集合相同，第⑦⑧组的 Top-100 集合也相同。

本次运行在 Top 10 中发现 **{s['supplemental_top10_judgments']} 个原来未标注的问题—chunk 对**。依据原有 0–3 分标准逐条阅读并补标，没有修改原标签，也没有用补标结果重新训练模型。八组的 P@10、MRR@10 和 nDCG@10 统一使用扩展后的 {s['judged_pairs']+s['supplemental_top10_judgments']:,} 条排序标注，所有 Top-10 结果都有标注。这些是开发用标签，尚未经过独立复核。

**Recall@100 仍只使用原始固定 qrels。** 新补标条目既不进入它的分子，也不进入分母，保持“找回原有正相关答案”的口径。不要求对第 11–100 名全部补标；未有原始标注的候选不会获得命中原正相关答案的计分，也不等于断言它实际不相关。扩展排序标注对所有 Top-100 返回位置的覆盖率为 {s['top100_judged_fraction']:.2%}。

## 指标计算

对于问题 q，Rq 为原始标签 ≥2 的 chunk 集合，Tq 为检索前 100 条的集合。

- **Precision@10：**前 10 条中标签 ≥2 的数量除以 10，使用扩展排序标注。
- **pooled Recall@100：**|Tq ∩ Rq| / |Rq|，只计算原始固定正相关集合。
- **MRR@10：**前 10 条中第一条标签 ≥2 的结果名次的倒数；没有则为 0。
- **nDCG@10：**DCG@10 / IDCG@10。DCG 为各名次 grade / log2(rank+1) 之和，增益直接使用 0、1、2、3。每题的 IDCG 从扩展标注中的最佳 10 个分数构造，八组使用同一份。

使用 `ir_measures` {s['provenance']['ir_measures_version']}，所有指标按 40 个问题取宏平均。TREC 导出的分数仅编码实际保存的顺序，避免后端对相同分数任意重新排序。1,280 个逐题指标值均与独立算式核对一致。

## 总体结果

{results}

重排序只改变同一批 100 个候选的顺序时，Recall@100 不会变化。因此⑤⑥之间以及⑦⑧之间的 Recall 相等是预期行为；最新版主要改善候选的靠前排序，没有在同一集合中找回额外的原正相关答案。

## 配对比较

{paired}

区间来自按问题配对进行的 10,000 次 bootstrap 重采样，随机种子为 42，采用百分位法构造 95% 区间。结果属于探索性分析，没有做多重比较校正。学习型模型的区间仅描述这批开发问题上的变化，不能消除训练与评测问题重叠的影响。

过滤产品后去掉型号是检索阶段收益最大的改进：④对③的 nDCG@10 增加 0.3164；方面奖励进一步增加 0.0326。保留型号时，RRF 的靠前排序好于 BM25、差于纯向量，因此不能笼统声称融合对所有指标都更好。

旧版重排序相较⑤组降低了平均 nDCG，但区间跨 0。查询改写将 pooled Recall@100 从 0.7275 提高到 0.7488；其相较⑥组的 nDCG 差值区间也跨 0。最新版学习型替换方案取得本次最好的开发集排序结果。

## 按问题类别分析

{grouped}

最新版在八类问题的平均 nDCG 上都有提高，其中电池、售后和相机相较⑦组增益较大。不过性能类和多方面类的 P@10 略有下降，提升并非覆盖每项指标。单方面类别每类仅 6 题，另两类每类仅 2 题，分类结果应作为描述性观察。

## 学习型排序与训练重叠

模型使用带 ridge 正则的线性打分，输入 12 项特征：MiniLM 原始 logit、语义倒数排名、召回倒数排名、向量分数、对数变换后的 BM25 分数、方面覆盖率、对数关键词数量、关键词密度、对数文本长度、语义 logit 的平方、向量倒数排名和 BM25 倒数排名。每项特征在当前候选窗口内中心化、标准化，标准差下限为 1e-6。最终分数为标准化特征与保存权重的点积，分数越高排名越靠前。推理不使用相关性标签或问题、chunk 的标识符作为特征。

此前在 32 种特征子集、正则强度和训练目标组合中选择模型，最后用原有 40 个开发问题的固定 Top-20 候选训练部署参数。部署版使用全部 12 项特征、0–3 等级训练目标和正则系数 100，没有用本次 62 条新增标注重新训练。

此前的嵌套五折验证将每个外层验证问题同时排除在训练和参数选择之外。在**此前固定、未改写的 Top-20 协议**下，P@10={cv['nested_cv'][0]:.4f}、MRR@10={cv['nested_cv'][1]:.4f}、nDCG@10={cv['nested_cv'][2]:.4f}。它与本次改写后 Top-100 开发集运行的协议不同，不能直接用作本次第⑧组的独立测试分数。本实验没有额外未接触过的测试集。

## 复现与结果文件

在项目根目录执行：

```powershell
.venv\\Scripts\\python.exe -X utf8 -m scripts.run_retrieval_ablation
uv run --with ir_measures python -X utf8 -m scripts.summarize_retrieval_ablation
uv run --with markdown python -X utf8 -m scripts.report_ablation_eight
```

采集脚本会复用已完成的检查点；修改代码或参数后如需完整重跑，应先归档输出目录或更换输出路径。本报告使用的 40 个检查点均为本次八组实验重新生成。

结果目录包含总体和逐题 CSV、配对区间、分类结果、八份 TREC 运行文件、完整 Top-100 快照、补充标注、排序 qrels、统计 JSON 和执行清单。清单记录代码、模型和数据哈希。DeepSeek 改写复用既有缓存，没有新增改写 API 调用；原测试集和历史实验数据保留。

本实验是累计式开发消融，不是完整的全因子实验；衡量的是证据检索与排序，不是生成答案质量。原标注池来自相关系统的候选合并，可能对这些系统存在偏向。后续应冻结模型，用新标注的问题验证泛化，并单独消融去型号、重排窗口和学习型特征。
'''

def main():
    s=json.loads((OUT/'statistics.json').read_text())
    cv=json.loads((ROOT/'results/reranker_learned/statistics.json').read_text())
    scores={r['system']:r for r in s['summary']}
    last=scores['08_learned'];prev=scores['07_rewrite']
    results=table(['Configuration','P@10','Pooled Recall@100','MRR@10','nDCG@10'],
        [[r['configuration'],*[f'{r[m]:.4f}' for m in ['P@10','pooled_Recall@100','MRR@10','nDCG@10']]] for r in s['summary']])
    comparisons=table(['Comparison','Mean nDCG@10 change','Paired 95% interval','Wins / ties / losses'],[
        [r['system'][:2]+' vs '+r['baseline'][:2],f"{r['delta']:+.4f}",f"[{r['ci95_low']:+.4f}, {r['ci95_high']:+.4f}]",f"{r['wins']} / {r['ties']} / {r['losses']}"]
        for r in s['paired_deltas'] if r['metric']=='nDCG@10'])
    groups={r['group']:r for r in s['by_query_group'] if r['system']=='07_rewrite'}
    grouped=table(['Question group','Questions','Stage 7 P@10','Stage 8 P@10','Stage 7 nDCG@10','Stage 8 nDCG@10'],[
        [r['group'],r['queries'],f"{groups[r['group']]['P@10']:.4f}",f"{r['P@10']:.4f}",f"{groups[r['group']]['nDCG@10']:.4f}",f"{r['nDCG@10']:.4f}"]
        for r in s['by_query_group'] if r['system']=='08_learned'])
    text=f'''# Retrieval Ablation Study: Eight Configurations

Experiment date: 29 October 2026. All 40 questions were rerun across all eight configurations: 320 system–question runs, each saving 100 results.

## Findings

The latest configuration achieved **P@10 = {last['P@10']:.4f}, pooled Recall@100 = {last['pooled_Recall@100']:.4f}, MRR@10 = {last['MRR@10']:.4f}, and nDCG@10 = {last['nDCG@10']:.4f}**. Relative to stage 7, Precision@10 increased by {last['P@10']-prev['P@10']:.4f} and nDCG@10 by {last['nDCG@10']-prev['nDCG@10']:.4f}; nDCG improved on 33 questions and declined on seven. The number of relevant Top-10 positions increased from 322 to 355 out of 400. MRR@10 of 1.0 means that the first result was graded relevant for every question; it does not mean every returned result was relevant.

These are **development results**. The deployed learned model was trained on these same 40 questions. The eight-way comparison is not an independent test of generalization. A separate nested cross-validation result from earlier development is reported below with its different protocol.

## Experimental configurations

1. **BM25 only:** product filtering, original query including its model name, no aspect reward.
2. **Dense only:** the same filtering and query policy, using BGE embeddings and the shared FAISS index.
3. **Hybrid RRF:** combine the two retrieval rankings; retain the model name.
4. **Model-name removal:** after resolving and filtering the product, remove its name from the search text before retrieval.
5. **Retrieval aspect boost:** add the existing aspect reward to configuration 4.
6. **Legacy reranker:** rerank configuration 5's 100 candidates using the previous additive MiniLM/RRF/aspect formula.
7. **Query rewriting:** apply the cached DeepSeek rewrite before retrieval and legacy reranking. This can change search text, product resolution and aspect extraction.
8. **Latest learned reranker:** reuse the same rewritten Top-100 candidate set as configuration 7 and replace its reranking strategy with the latest trained review ranker.

Configurations 1 and 2 are parallel baselines. Configurations 3–7 form a cumulative sequence; configuration 8 replaces the reranking implementation within configuration 7. The 8-versus-7 comparison measures the entire replacement: model-stripped semantic query, learned features and a 20-candidate reranking window, rather than learned weights alone.

## Data and retrieval protocol

The original index contains 459,418 chunks. Retrieval used the shared existing indexes; no index was rebuilt. The 40 fixed questions comprise 36 single-aspect questions (six each for camera, battery, display, performance, service and carrier), two multi-aspect questions and two product-comparison questions.

The original qrels contain 2,564 question–chunk judgments, with relevance grades 0–3. Grades 2 and 3 count as binary relevant. The pool is incomplete: it is not an exhaustive enumeration of all relevant chunks in the database. Positive and negative opinions can both be relevant.

Each retrieval request keeps 100 candidates and at most one chunk per source review. Because the retriever uses depth = max(candidate_k, 3 × k) with review deduplication enabled, the effective depth is **300 per retrieval channel**, despite candidate_k=150 in the call. Hybrid RRF uses k=60. The retrieval aspect reward is 0.5/(60+1) for any matching query aspect.

Configurations 1–6 use the rule parser's single resolved product. Configurations 7–8 use the cached rewrite's products, retrieve up to 100 per product, merge by chunk ID, sort by retrieval score and retain a global Top 100. Comparison questions therefore do not force equal representation of both products. Rewriting and improved multi-product parsing are bundled in stage 7.

The learned model reorders only the first 20 of those 100 candidates; positions 21–100 retain their retrieval order. The legacy model reorders all 100. This window difference is intentional and must be considered when interpreting their comparison. Configurations 5 and 6 have identical Top-100 membership; configurations 7 and 8 also have identical Top-100 membership.

The run exposed **{s['supplemental_top10_judgments']} previously unjudged question–chunk pairs in Top-10 output**. These were read and graded directly using the existing 0–3 rubric, without changing original labels or retraining the model. All eight configurations use the same expanded ranking qrels of {s['judged_pairs']+s['supplemental_top10_judgments']:,} judgments for P@10, MRR@10 and nDCG@10. All Top-10 results are now judged. These judgments are development labels, not independently reviewed gold labels.

**Recall@100 uses the unchanged original qrels only.** Supplemental labels are excluded from both its numerator and denominator, preserving the requested fixed-positive evaluation. Items without an original judgment earn no credit for finding an original positive; this does not assert that they are truly irrelevant. No exhaustive Top-100 labeling is required. The observed ranking-qrels coverage of returned Top-100 positions is {s['top100_judged_fraction']:.2%}.

## Metric definitions

For each question q, let Rq be the set of original chunks labeled at least 2 and Tq be its returned first 100 chunks.

- **Precision@10:** number of Top-10 chunks with grade ≥2 / 10, using the expanded ranking qrels.
- **Pooled Recall@100:** |Tq ∩ Rq| / |Rq|, using only the original fixed positives.
- **MRR@10:** reciprocal of the rank of the first grade-≥2 result in the Top 10, or zero if none exists.
- **nDCG@10:** DCG@10 / IDCG@10. DCG = sum of grade(rank) / log2(rank+1). Gains are linear grades 0, 1, 2, 3. IDCG uses the best ten grades in the same expanded question-specific ranking qrels for every system.

All metrics are macro-averaged across the 40 questions with `ir_measures` {s['provenance']['ir_measures_version']}. Exported TREC scores encode the saved order, preventing arbitrary tie reordering. Each of the 1,280 per-question metric values was checked against an independent arithmetic implementation.

## Overall results

{results}

Recall@100 stays unchanged when a reranker merely permutes the same 100 candidates. The equality between stages 5 and 6, and between 7 and 8, is therefore expected. The latest ranking improvement comes from ordering the retrieved candidates more effectively, not finding additional original positives.

## Paired comparisons

{comparisons}

Intervals use 10,000 paired question bootstrap resamples, seed 42, and percentile 95% bounds. They are exploratory and not adjusted for multiple comparisons. For the learned model, these intervals describe variation across development questions; they do not remove training overlap or establish held-out superiority.

Removing the model name after product filtering produced the largest retrieval improvement: stage 4 improved nDCG@10 over stage 3 by 0.3164. Aspect boosting added another 0.0326. RRF with an unmodified query outperformed BM25 but underperformed dense-only retrieval in early-ranking quality, so this experiment does not support an unconditional claim that fusion improves every metric.

The legacy reranker reduced mean nDCG relative to stage 5; its difference interval crosses zero. Query rewriting increased pooled Recall@100 from 0.7275 to 0.7488, but its nDCG difference against stage 6 also has an interval crossing zero. The latest learned replacement has the strongest development ranking results.

## Results by question group

{grouped}

The learned replacement improved mean nDCG in all eight groups. The largest gains relative to stage 7 were in battery, service and camera. Precision@10 declined slightly for performance and multi-aspect questions, so the improvement is not uniform across all measures. Each single-aspect group contains only six questions; each remaining group contains two. Subgroup results are descriptive.

## Learned reranker and training overlap

The model is a ridge-regularized linear scorer with 12 query-normalized features: raw MiniLM logit; semantic reciprocal rank; retrieval reciprocal rank; dense score; log-transformed BM25 score; aspect coverage; log keyword count; keyword density; log text length; squared semantic logit; dense reciprocal rank; and BM25 reciprocal rank. Each feature is centered and scaled within the candidate window. A standard-deviation floor of 1e-6 handles constant features. The score is the dot product with the saved coefficients; higher is better. Neither a relevance label nor a query/chunk identifier is used as an inference feature.

The deployed artifact was selected from 32 combinations of feature subsets, regularization strengths and grade/binary training targets, then fit on all 40 original development questions using the fixed original Top-20 candidates. It uses all 12 features, grade targets and ridge penalty 100. It was not retrained on the 62 supplemental judgments.

Earlier nested five-fold query validation excluded each outer validation question from both fitting and parameter selection. Under that **earlier fixed, non-rewritten Top-20 protocol**, it achieved P@10={cv['nested_cv'][0]:.4f}, MRR@10={cv['nested_cv'][1]:.4f} and nDCG@10={cv['nested_cv'][2]:.4f}. This is distinct from the present rewritten Top-100 development run and cannot be substituted for stage 8's held-out score. No untouched test set is available in this experiment.

## Reproduction and artifacts

Run from the repository root:

```powershell
.venv\\Scripts\\python.exe -X utf8 -m scripts.run_retrieval_ablation
uv run --with ir_measures python -X utf8 -m scripts.summarize_retrieval_ablation
uv run --with markdown python -X utf8 -m scripts.report_ablation_eight
```

Completed collection checkpoints are reused by the runner; archive the output directory or change the output path for a fresh execution after modifying the code or parameters. All 40 checkpoints in this report were freshly collected for the eight-system run.

Files in this report directory include summary.csv, per_query.csv, paired_deltas.csv, by_query_group.csv, eight TREC runs, full Top-100 result snapshots, supplemental_labels.json, ranking_qrels.trec, statistics.json and execution_manifest.json. The manifest records code, model and data hashes. Existing DeepSeek rewrites were reused; no new rewriting API calls were made. Original dataset files and historical experiment results were preserved.

These eight configurations form a cumulative development ablation, not a full factorial experiment. Results measure evidence retrieval and ranking, not generated answer quality. The original qrels were pooled using earlier related systems, which may favor those systems. The most useful next validation is a frozen learned model on newly labeled questions, followed by component ablations that separately vary query stripping, reranking window and learned features.
'''
    body=markdown.markdown(text,extensions=['tables','fenced_code'])
    chart='''<section><h2>Compare metrics</h2><label>Metric <select id="metric"><option>nDCG@10</option><option>P@10</option><option>pooled_Recall@100</option><option>MRR@10</option></select></label><div id="chart"></div><p class="note">X axis: metric value (0–1). Y axis: experimental configuration. Source: 40 questions, 29 October 2026. Values are question macro-averages.</p></section>'''
    script='''const data=DATA;function draw(){const m=document.getElementById('metric').value;document.getElementById('chart').innerHTML=data.map(r=>`<div class="bar"><span>${r.configuration}</span><div class="track"><div style="width:${100*r[m]}%"></div></div><b>${r[m].toFixed(4)}</b></div>`).join('')}document.getElementById('metric').onchange=draw;draw();'''.replace('DATA',json.dumps(s['summary']))
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Eight-Configuration Retrieval Ablation Report</title><style>
body{font:16px/1.65 system-ui,sans-serif;color:#18232d;background:white;margin:0}main{max-width:1120px;margin:auto;padding:40px 28px}h1{font-size:28px}h2{font-size:21px;margin-top:36px;border-bottom:1px solid #d9e0e5;padding-bottom:8px}table{border-collapse:collapse;width:100%;font-size:14px;margin:20px 0;display:block;overflow:auto}td,th{border-bottom:1px solid #d9e0e5;padding:10px 14px;text-align:right;white-space:nowrap}td:first-child,th:first-child{text-align:left}th{background:#f2f5f7}code,pre{background:#f2f5f7}pre{padding:15px;overflow:auto}.note{font-size:13px;color:#596774}.bar{display:grid;grid-template-columns:300px 1fr 65px;gap:12px;margin:12px 0;align-items:center;font-size:13px}.track{background:#edf1f3;height:18px}.track div{height:100%;background:#246b8b}select{font:inherit;padding:5px}a{color:#246b8b}@media(max-width:700px){main{padding:18px}.bar{grid-template-columns:160px 1fr 50px}}@media print{select{display:none}h2{break-after:avoid}tr{break-inside:avoid}}</style><main>'''+chart+body+'<p><a href="summary.csv">Download metrics CSV</a> · <a href="per_query.csv">Per-question metrics</a></p></main><script>'+script+'</script></html>'
    english=page.replace('<main>','<main><nav><a href="report_zh.html">中文版</a> · English</nav>',1)
    (OUT/'report_en.html').write_text(english,encoding='utf8')
    zh_body=markdown.markdown(chinese_report(s,cv),extensions=['tables','fenced_code'])
    zh_chart=chart.replace('Compare metrics','指标对比').replace('Metric <select','指标 <select').replace('X axis: metric value (0–1). Y axis: experimental configuration. Source: 40 questions, 29 October 2026. Values are question macro-averages.','横轴：指标数值（0–1）；纵轴：实验配置。来源：2026 年 10 月 29 日的 40 题实验；数值为逐题宏平均。')
    zh_data=[{**r,'configuration':ZH_NAMES[r['system']]} for r in s['summary']]
    zh_script=script.replace(json.dumps(s['summary']),json.dumps(zh_data,ensure_ascii=False))
    chinese=page.replace('lang="en"','lang="zh-CN"').replace('<title>Eight-Configuration Retrieval Ablation Report</title>','<title>八组检索消融实验报告</title>').replace(chart,zh_chart).replace(body,zh_body).replace(script,zh_script).replace('Download metrics CSV','下载总体指标 CSV').replace('Per-question metrics','逐题指标').replace('<main>','<main><nav>中文版 · <a href="report_en.html">English</a></nav>',1)
    (OUT/'report_zh.html').write_text(chinese,encoding='utf8')
    print('HTML reports written: report_zh.html and report_en.html')

if __name__=='__main__':main()

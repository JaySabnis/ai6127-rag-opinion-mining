import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/ablation_100"


def main():
    data = json.loads((OUT / "statistics.json").read_text(encoding="utf-8"))
    metrics = ["P@10", "pooled_Recall@100", "MRR@10", "nDCG@10"]
    best = {m: max(r[m] for r in data["summary"]) for m in metrics}
    rows = "".join("<tr><th>" + html.escape(r["configuration"]) + "</th>" + "".join(
        f'<td class="{"best" if r[m] == best[m] else ""}">{r[m]:.4f}</td>' for m in metrics) + "</tr>" for r in data["summary"])
    page = '''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>全库检索消融实验 · 40题</title>
<style>
:root{color-scheme:light;--ink:#17212b;--muted:#5d6975;--line:#dce2e7;--accent:#175f86;--soft:#f2f6f8}
*{box-sizing:border-box}body{margin:0;background:white;color:var(--ink);font:15px/1.7 system-ui,"Microsoft YaHei",sans-serif}
main{max-width:1160px;margin:auto;padding:36px 28px}h1{font-size:24px;margin:0 0 8px}h2{font-size:19px;margin:30px 0 12px}p{margin:8px 0}a{color:var(--accent)}
.meta,.note{color:var(--muted);font-size:13px}.lead{border-left:4px solid var(--accent);padding:10px 18px;background:var(--soft);margin:24px 0}
.kpis{display:flex;gap:36px;flex-wrap:wrap;border-bottom:1px solid var(--line);padding:12px 0 20px}.kpis strong{display:block;font-size:22px}.kpis span{font-size:13px;color:var(--muted)}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}td,th{padding:10px 12px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}thead th{color:var(--muted);font-size:13px}td:first-child,th:first-child{text-align:left}tbody th{font-weight:500}.best{font-weight:750;color:var(--accent);text-decoration:underline;text-underline-offset:4px}
.scroll{overflow-x:auto}.controls{display:flex;gap:18px;align-items:center;flex-wrap:wrap}select{font:inherit;padding:5px 10px;border:1px solid var(--line);background:white;color:var(--ink)}
.axes{font-size:13px;color:var(--muted);margin:14px 0 10px}.bar-row{display:grid;grid-template-columns:170px 1fr 62px;gap:12px;align-items:center;margin:9px 0}.track{height:22px;background:var(--soft)}.bar{height:100%;background:#a7b5bf}.bar.top{background:var(--accent)}.value{text-align:right;font-variant-numeric:tabular-nums}.axis{display:flex;justify-content:space-between;margin:2px 74px 0 182px;font-size:12px;color:var(--muted)}
li{margin:7px 0}.protocol{display:grid;grid-template-columns:1fr 1fr;gap:24px}.protocol p{font-size:14px}footer{border-top:1px solid var(--line);margin-top:30px;padding-top:15px;font-size:12px;color:var(--muted)}
@media(max-width:700px){main{padding:22px 16px}.protocol{grid-template-columns:1fr}.bar-row{grid-template-columns:120px 1fr 54px;font-size:12px}.axis{margin-left:132px;margin-right:66px}.kpis{gap:20px}td,th{padding:8px}}
</style>
<main>
<h1>全库检索消融实验</h1><p class="meta">2026-10-29 · 40个问题 · 7组累计配置 · 固定0–3分标注 · ir_measures 0.4.3</p>
<div class="kpis"><div><strong>280 / 280</strong><span>重新执行的系统×问题，系统运行结果</span></div><div><strong>100%</strong><span>7组Top10已有标注</span></div><div><strong>2,564</strong><span>原始问题—chunk标注对，补标后以统计JSON为准</span></div></div>
<div class="lead"><strong>召回方面加分后，P@10最高；完整流程的nDCG仅小幅领先。</strong><p>⑤的P@10为0.8325、MRR@10为0.9313。⑦的nDCG@10为0.7716，较⑤仅高0.0016。当前reranker的平均增益为负；本次不调整权重或标注。</p></div>
<h2>全部配置的平均指标</h2><p class="note">每列均为40题宏平均；越高越好。加粗下划线为该列最高值。Recall基于同一标注池。</p>
<div class="scroll"><table><thead><tr><th>累计配置</th><th>P@10</th><th>pooled Recall@100</th><th>MRR@10</th><th>nDCG@10</th></tr></thead><tbody>__ROWS__</tbody></table></div>
<h2>按指标和问题类型查看</h2><div class="controls"><label>指标 <select id="metric"><option>nDCG@10</option><option>P@10</option><option>MRR@10</option><option>pooled_Recall@100</option></select></label><label>问题类型 <select id="group"><option value="all">全部40题</option></select></label></div>
<p class="axes" id="chart-title"></p><div id="chart"></div><div class="axis"><span>0</span><span>0.25</span><span>0.50</span><span>0.75</span><span>1.00</span></div><p class="note">横轴：所选指标的平均分（0–1，无单位）。纵轴：实验配置。来源：本次重新检索/重排结果与固定qrels。</p>
<h2>每一步增加模块的变化</h2><p class="note">下表跟随上方所选指标。差值=新配置−对照配置，正值表示提高。胜/平/负按逐题指标比较。</p>
<div class="scroll"><table><thead><tr><th>比较</th><th>平均差值</th><th>配对bootstrap 95%区间</th><th>胜 / 平 / 负</th></tr></thead><tbody id="deltas"></tbody></table></div>
<p class="note">10,000次按问题配对重采样，种子42；区间未作多重比较校正，作为探索性结果。区间跨0时，不据此认定存在稳定增益或退化。</p>
<h2>实验口径与解释边界</h2><div class="protocol"><div>
<p><strong>检索与排序。</strong>搜索原459,418条chunk的全库索引，按各系统解析的产品过滤；每路召回至少300，候选池100，每评论最多1个chunk。①BM25、②向量、③RRF均保留查询型号；④去型号；⑤加召回方面奖励0.5；⑥加学习型重排序；⑦加查询改写模块。</p>
<p><strong>评分。</strong>P、Recall、MRR以标注≥2为相关；nDCG使用线性增益0、1、2、3和全标注池IDCG。Recall截断在100；P、MRR、nDCG截断在10。不删除不相关候选，不将未标注项默认为0。仅Top10要求完整标注；Recall@100统计命中的已有正相关chunk。</p>
</div><div>
<p><strong>改写与比较题。</strong>第7组复用保存的实际DeepSeek改写，无新API调用。该模块会改变文本、产品提取和aspect；其增益不能仅归于措辞。①至⑥规则parser只识别一个产品，⑦支持多产品，各召回100后全局重排取100。</p>
<p><strong>数据范围。</strong>40题共用不完全的pooling标注，不能把pooled Recall称为穷尽全库召回率。Recall@100使用共同标注池，11–100名不强制补标。当前40题是既有测试题，没有额外盲测或调参开发集；每个单方面组仅6题，另外两类各2题。</p>
</div></div>
<h2>可复查数据</h2><p><a href="summary.csv">全部配置指标 CSV</a> · <a href="paired_deltas.csv">配对差值与区间 CSV</a> · <a href="per_query.csv">逐题指标 CSV</a> · <a href="by_query_group.csv">分组指标 CSV</a> · <a href="statistics.json">完整统计 JSON</a> · <a href="replication.csv">复现检查 CSV</a> · <a href="top10_nonrelevant.jsonl">Top10非相关候选明细</a></p>
<footer>来源：本项目 retrieval_ablation_40 固定问题与qrels；results/ablation_100 本次运行。索引和数据文件SHA256已核验；7×40×4=1,120个逐题指标均与独立算式核对一致。结果不涉及答案生成质量。</footer>
</main><script>
const DATA=__DATA__;
const names=Object.fromEntries(DATA.summary.map(r=>[r.system,r.configuration]));
const fmt=v=>v.toFixed(4);const signed=v=>(v>=0?'+':'')+fmt(v);
const group=document.getElementById('group'), metric=document.getElementById('metric');
const groupNames={camera:'相机',battery:'续航',display:'屏幕',performance:'性能',service:'售后服务',carrier:'运营商',multi_aspect:'多方面',comparison:'产品对比'};
[...new Set(DATA.by_query_group.map(r=>r.group))].forEach(g=>{const o=document.createElement('option');o.value=g;o.textContent=groupNames[g]+'（'+DATA.by_query_group.find(r=>r.group===g).queries+'题）';group.append(o)});
function render(){const m=metric.value,g=group.value;const rows=g==='all'?DATA.summary:DATA.by_query_group.filter(r=>r.group===g);const max=Math.max(...rows.map(r=>r[m]));document.getElementById('chart-title').textContent=(g==='all'?'全部40题':group.options[group.selectedIndex].text)+' · '+m+' 宏平均分';document.getElementById('chart').innerHTML=rows.map(r=>`<div class="bar-row"><span>${names[r.system]}</span><div class="track"><div class="bar ${r[m]===max?'top':''}" style="width:${r[m]*100}%"></div></div><span class="value">${fmt(r[m])}</span></div>`).join('');document.getElementById('deltas').innerHTML=DATA.paired_deltas.filter(r=>r.metric===m).map(r=>`<tr><th>${names[r.system]} − ${names[r.baseline]}</th><td>${signed(r.delta)}</td><td>[${signed(r.ci95_low)}, ${signed(r.ci95_high)}]</td><td>${r.wins} / ${r.ties} / ${r.losses}</td></tr>`).join('')}
group.onchange=metric.onchange=render;render();
</script></html>'''
    page = page.replace("__ROWS__", rows).replace("__DATA__", json.dumps(data, ensure_ascii=False).replace("</", "<\\/"))
    (OUT / "report.html").write_text(page, encoding="utf-8", newline="\n")
    print(OUT / "report.html")


if __name__ == "__main__":
    main()

# 本地修复版运行说明

这份修复基于仓库 main 的 `d5b1f5b`，开发与验证分支为 `local-audit-fixes`。人工标注、分歧表、`eval.xls` 和抽样文件均未修改。

本次修复涉及机型别名合并、检查器错误证据处理和旧缓存隔离、弃答导出、实验对比方式、耗时统计及评估文件的版本核对。最终结果见 [测试报告](results/local_validation_20261105/VALIDATION_REPORT.md)。

## 1. 安装与离线测试

使用 Python 3.13 或以上版本，在项目根目录运行：

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest tests -q
```

已安装 uv 时，也可以使用 `uv sync --all-groups` 和 `uv run --all-groups python -m pytest tests -q`。依赖清单由现有 `uv.lock` 导出；本次测试使用 Python 3.13.13。

Windows 激活环境使用 `.venv\Scripts\activate`。运行 API 测试前，将 `.env.example` 复制为 `.env` 并填写自己的 DeepSeek 配置。压缩包不包含密钥、缓存、Python 环境和完整检索索引。

## 2. 不调用 API 的结果复核

以下命令只读取已有预测和标签，把新结果写到独立目录：

```bash
python -m evaluation.report.recompute_saved_statistics --output-dir results/recheck/statistics
python evaluation/eval_classifier.py evaluation/predictions/deepseek_overall.csv --output-dir results/recheck/classification
python evaluation/eval_classifier.py evaluation/predictions/vader_baseline.csv --output-dir results/recheck/classification
python -m evaluation.summarize_latency evaluation/timings/B40_learned.jsonl --output results/recheck/latency.json
```

不要为了复核代码而重新运行 `merge_labels.py` 或重新生成 `eval.xls`。本次评分始终使用原有 1,194 条有效金标。

## 3. 检查器控制对比（会调用 API）

```bash
python -m evaluation.report.run_paired_checker --env-file .env --output-dir results/recheck/paired --max-api-calls 80
```

输入是保存的 40 份相同草稿、评论和分类结果。两个实验组分别保留原草稿、仅应用新版检查器；不会重新分类或生成草稿。中断后可加 `--resume`，但输入、源码或模型改变时会拒绝续用旧结果。重新实验应使用新的输出目录。

运行四项 RAGAS 评分前，设置：

```bash
export EMBEDDING_PROVIDER=huggingface
export EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
export RAGAS_DO_NOT_TRACK=true
python evaluation/eval_rag.py results/recheck/paired/rag_outputs/checker_off.jsonl --env-file .env --output-dir results/recheck/ragas
python evaluation/eval_rag.py results/recheck/paired/rag_outputs/checker_new.jsonl --env-file .env --output-dir results/recheck/ragas
```

这些是模型辅助评分，RAGAS 会多次调用 API。`answer_relevancy` 使用本地文本向量模型，生成和检查仍使用 DeepSeek。不要把旧 B40 checker 结果当作固定同一草稿的对照实验，也不要把标出的句子直接视为人工确认的幻觉。

## 4. 五条完整流程测试（会调用 API）

按主 README 的下载链接恢复完整 `indexes/` 后运行：

```bash
python -m scripts.benchmark_pipeline --queries results/local_validation_20261105/five_queries.json --env-file .env --index-dir indexes --expected-chunks 459418 --top-k 10 --output-dir results/recheck/end_to_end
```

此命令实际执行改写、检索、重排、分类、总结和检查；不会以样例候选替代全库检索。首次运行可能下载 BGE 和 MiniLM 检索模型。`--check-only` 可先检查输入、依赖和索引文件，但它本身不证明完整流程成功。

默认禁用应用响应缓存；索引加载和本地预热单列，查询耗时包含各业务步骤。串行吞吐估计不能当作并发压测结论。

## 5. 结果使用

- `results/local_validation_20261105/` 是本次实测证据；历史结果保留在原目录。
- `statistics/` 保存重算后的分组比例，不替换任何人工标签或模型分类预测。
- `paired_checker/` 与 `paired_ragas/` 是最终检查器对比及四项评分。
- `end_to_end/` 是最终五条全库实测。
- 早期 `initial`、`before_cache_isolation` 批次仅保留在开发机器，未纳入公开提交；最终汇报使用测试报告指明的结果。
- 现有实验没有证明检查器提升回答质量；基线与检查器的结果和限制应一起呈现。

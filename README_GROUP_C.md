# C组情感分类 总结与证据检查

本地开发版本使用DeepSeek官方API完成三项任务：读取评论判断情感，按独立评论统计好评比例，生成有引用的总结并逐句检查依据。C组代码只依赖Python标准库，不下载或运行本地大语言模型。它目前是提示词分类器，没有进行模型微调。

本地代码基于仓库retrieve分支commit `998a188621c06e88472f48d5309218f05f163b83`，并取回evaluation分支commit `9229a4eee2476684ed4615d6e02e739055d8206c`中的评估文件。尚未推送或合并到远端。已有检索、向量编码、重排序模块仍属于B组，单独运行C组不需要加载它们。

## 快速运行

在本文件所在目录运行。`python`代表你使用的Python 3解释器；现有仓库声明Python 3.13+，C组独立命令也已在Python 3.12验证。

```bash
python -m scripts.group_c classify --text "The camera is excellent, but the battery is terrible." --aspect battery --output results/group_c/single.csv

python -m scripts.group_c run --input tests/fixtures/rerank_candidates.json --question "Galaxy S10的续航怎么样？" --aspects battery --no-check --output results/group_c/baseline.json

python -m scripts.group_c check --input tests/fixtures/rerank_candidates.json --summary-file examples/group_c/checker_challenge.json --output results/group_c/checker_challenge.json

python -m unittest discover -s tests/group_c -v
```

分享代码包不含密钥。接收者需复制`.env.example`为`.env`，填入自己的`DEEPSEEK_API_KEY`。默认模型`deepseek-flash`、地址`https://api.deepseek.com`，可用`--model`和`--env-file`覆盖。环境变量优先于文件；不打印密钥。

当前采用纯API基线：分类、代码统计、总结，运行时显式加`--no-check`关闭依据检查。检查器保留用于后续对照；省略`--no-check`即可开启。

没有调用检索/搜索工具：API只接收本次指定的评论片段和问题。评论会发送至DeepSeek进行推理。运行环境、数据文件、请求缓存、结果和代码保存在本机。

## 文件与职责

| 文件 | 职责 |
|---|---|
| `generation/client.py` | API认证、JSON输出、有限重试、缓存、调用预算和token记录 |
| `generation/sentiment.py` | 主观性与极性判断、方面情感、独立评论计数 |
| `generation/summary.py` | 只用输入证据生成带chunk_id引用的结构化句子 |
| `generation/verifier.py` | 引用合法性、原文引句检查及语义支持判断 |
| `generation/pipeline.py` | 串联流程及B组接口适配 |
| `generation/io.py` | 读JSON/JSONL/CSV，输出预测、回答、时延与可读报告 |
| `scripts/group_c.py` | classify、run、check三个命令 |

## 情感标签与统计口径

先判断目标方面是否出现，再判断主观性，最后对有观点内容判断极性。`positive`为夸，`negative`为批评，`mixed`为同一目标既夸又批评，`neutral`为没有主观情感。`aspect_present=false`表示该方面未出现，单独排除，不与相关的客观描述混淆。调用不成功或输出不合法时报告错误，不能用neutral代替失败。

不指定`--aspect`时判断整段评论；指定battery时，只判断该手机续航，不能让相机评价影响续航标签。输入API时不提供星级或人工标签。`rating`仅保留在输入元数据中。

同一doc_id的多个chunk只计一条评论；出现正负冲突归为mixed，相关中立片段不覆盖有观点片段。仅分析检索出来的片段，未见到的其余原始评论内容可能改变完整评论判断，报告应说明这个限制。不同品牌/机型分别统计。

默认好评比例：

```text
positive_ratio = 正面独立评论数 / 该方面全部相关独立评论数
分母包括positive、negative、mixed、neutral，排除方面未出现的评论。
```

同时输出`polarized_positive_ratio = positive / (positive + negative)`供需要二分类口径时使用；中立/混合的数量始终保留。分母为零时返回null，不能写成0%或100%。所有比例均为retrieved_sample，不代表总体消费者满意度。统计由代码计算，LLM不负责计算或编写比例。

## B组接口

接收B组`Hit.to_dict()`的扁平字段，也接受含metadata的候选；必须有`chunk_id`、`doc_id`、`text`。重复chunk_id自动去重，冲突数据直接报错。

```python
from generation.client import DeepSeekClient
from generation.pipeline import enrich_reply

# reply来自app.pipeline.search_full_corpus，或现有reranker流程。
# 自动读取reply['chunks']和reply['rewrite']['aspects']。
reply = enrich_reply(reply, question,
                     client=DeepSeekClient(cache_dir='.cache/group_c'),
                     verify=False)  # 当前基线；对照实验设为True
print(reply['answer'])
print(reply['statistics'])
print(reply['flagged_sentences'])
```

Streamlit已有本地接入开关。按B组说明安装其依赖后运行：

```bash
streamlit run app/streamlit_app.py
```

勾选侧栏`Answer & sentiment (DeepSeek)`。当前基线需关闭`Check summary evidence`；核查对照实验时开启。回答下显示比例，开启核查后还显示逐句检查。默认不开启生成以保留B组原有检索操作；C组命令行可独立运行，不需要Streamlit或检索索引。本次尚未构建全库索引、启动完整Streamlit系统或下载完整语料。

## 总结和检查器

总结最多六句，每句附本次输入中存在的chunk_id。输入评论只能作为资料，不作为系统指令；提示词要求不使用外部常识补全、不过度推广个人体验，并保留相反观点。

检查器分两层。程序先检查引用ID和引句是否确实存在；另外一次API调用再判断整句是否被引用证据支持。一个句子中只支持一部分内容，必须标为partially_supported。不存在的引用直接标unsupported，不调用模型。

检查结果包括supported、partially_supported、unsupported、contradicted、unverifiable；除supported外都标记。结果保留原始草稿、每句判断、理由和原文引句，默认最终回答只保留supported句。检查服务失败时保留草稿供检查，同时全部标记未验证，不将失败当作通过。

这是模型辅助核查，不能保证没有幻觉。同一服务同时生成和核查有相关性偏差，真实质量应由独立人工标注与盲评验证，不把模型自评分当作准确率。

可用`run --no-check`做关闭核查器的消融实验；该模式保留未验证草稿。使用同样问题、检索证据、模型及参数进行对比。

## 评估数据输出

`run`生成：完整JSON、可读的同名`.md`报告、`.rag.jsonl`及`.timings.jsonl`。后两者匹配evaluation分支所需字段。时延文件不写入total_s，避免现有脚本把总耗时再次加总；尚无检索时延，可在完整系统中补入retrieval_s。

人工eval_sample.csv的现有标签是整段情感，因此正式分类预测应不带`--aspect`，不得把方面情感预测与整段人工标签直接比较：

```bash
# 首先只试12条，确认输出；这不是正式准确率评估。
python -m scripts.group_c classify --input evaluation/eval_sample.csv --limit 12 --output evaluation/predictions/deepseek_overall_smoke.csv

# 1200条每12条一批，至少100次调用；明确增大预算。
python -m scripts.group_c classify --input evaluation/eval_sample.csv --max-api-calls 150 --output evaluation/predictions/deepseek_overall.csv

# 需要人工标注完成后的eval.xls，以及evaluation分支的依赖。
python evaluation/eval_classifier.py evaluation/predictions/deepseek_overall.csv
```

预测CSV严格是`chunk_id,label`，额外详情存同名`.details.json`。不把模型预测冒充人工gold或写入人工标签文件。现有evaluation分支允许单标注记录进入eval.xls；正式提交前应由评估负责人核查至少1000条有效记录的多人独立标注和仲裁前一致性是否满足课程要求。

这里不自动调用RAGAS，也不自动给全部1200条付费推理。正式评价需要人工gold、真实查询和已冻结的检索结果。B组的六条fixture和故意制造的checker_challenge只是开发样例，不构成测试集或课程实验结论。

## 调用与复现

默认每个进程最多32次API请求，含重试。相同端点、模型、提示和输入命中本地缓存，不重复付费。用`--no-cache`实测未缓存时延；缓存命中时记录为cache_hits，不重复计算token消费。真实费用需按实际模型及官方价格计算，代码不猜测美元金额。

每批分类最多12条，run最多64条片段、60000字符；超出即报错，不静默截断。大批分类有失败时可重新运行，已成功批次会复用缓存，预测文件只在全部成功后替换。不同提示词或输入自然产生不同缓存key。

官方API接口说明：https://api-docs.deepseek.com/ 。模型名称和价格可能变化；运行配置和结果中的model、usage用于复现。不要分享`.env`、`.cache/`或含凭据的本机配置。

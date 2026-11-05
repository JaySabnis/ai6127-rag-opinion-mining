# Group C: Sentiment Classification, Summarization and Evidence Checking

The Group C pipeline uses the official DeepSeek API for sentiment classification, cited summarization and sentence-level evidence checking. Python computes positive ratios over unique reviews. Group C code depends only on the Python standard library and does not download or run a local LLM. It is currently a prompted classifier; no model has been fine-tuned. Output (answers, check reasons, reports) is in English by default; pass `language="zh"` (or `run --language zh`) for Chinese. The Streamlit app answers in Chinese when the question contains Chinese characters.

This local revision starts from main commit `d5b1f5b` and adds the November 5 audit fixes. These additional fixes have not been pushed. The existing retrieval, embedding and reranking modules still belong to Group B; running Group C on its own does not load them.

## Quick start

Run from the directory containing this file. `python` stands for your Python 3 interpreter; the repository declares Python 3.13+, and the standalone Group C commands have also been verified on Python 3.12.

```bash
python -m scripts.group_c classify --text "The camera is excellent, but the battery is terrible." --aspect battery --output results/group_c/single.csv

python -m scripts.group_c run --input tests/fixtures/rerank_candidates.json --question "How is the Galaxy S10 battery life?" --aspects battery --no-check --output results/group_c/baseline.json

python -m scripts.group_c check --input tests/fixtures/rerank_candidates.json --summary-file examples/group_c/checker_challenge.json --output results/group_c/checker_challenge.json

python -m unittest discover -s tests/group_c -v
```

The shared code package contains no keys. Recipients must copy `.env.example` to `.env` and fill in their own `DEEPSEEK_API_KEY`. The default model is `deepseek-flash` and the default endpoint is `https://api.deepseek.com`; override them with `--model` and `--env-file`. Environment variables take precedence over the file; keys are never printed.

The current baseline is API-only: classification, statistics computed in code, and summarization, with the evidence check explicitly turned off by `--no-check`. The checker is kept for later comparison; omit `--no-check` to enable it.

No retrieval or search tools are called: the API receives only the review chunks and question given for that run. Reviews are sent to DeepSeek for inference. The runtime environment, data files, request cache, results and code stay on the local machine.

## Files and responsibilities

| File | Responsibility |
|---|---|
| `generation/client.py` | API authentication, JSON output, bounded retries, caching, call budget and token accounting |
| `generation/sentiment.py` | Subjectivity and polarity judgments, aspect sentiment, unique-review counts |
| `generation/summary.py` | Structured sentences with chunk_id citations, generated only from the input evidence |
| `generation/verifier.py` | Citation validity, verbatim quote checks and semantic support judgments |
| `generation/pipeline.py` | Chains the stages and adapts the Group B interface |
| `generation/io.py` | Reads JSON/JSONL/CSV; writes predictions, answers, timings and readable reports |
| `scripts/group_c.py` | The three commands: classify, run, check |

## Sentiment labels and statistics

The classifier first decides whether the target aspect appears, then whether the text is subjective, and finally the polarity of opinionated content. `positive` means praise, `negative` means criticism, `mixed` means both praise and criticism of the same target, and `neutral` means no subjective sentiment. `aspect_present=false` means the aspect does not appear; such reviews are excluded separately and not confused with relevant objective descriptions. A failed call or invalid output is reported as an error; neutral is never used as a stand-in for failure.

Without `--aspect`, the whole review is judged; with `--aspect battery`, only that phone's battery life is judged, so a camera opinion cannot affect the battery label. Star ratings and human labels are not sent to the API. `rating` is kept only in the input metadata.

Multiple chunks with the same doc_id count as one review; conflicting positive and negative chunks become mixed, and relevant neutral chunks do not override opinionated ones. Only retrieved chunks are analyzed, so unseen parts of the original review could change the whole-review judgment; reports should state this limitation. Products are grouped by stable `product_id`. For old saved files, the shared catalogue spelling rules merge aliases such as Note9 / Note 9 and S20FE / S20 FE. Plus, Ultra, FE and 5G variants remain distinct. Unknown models without a stable ID are isolated by source review; the original model text is preserved.

Default positive ratio:

```text
positive_ratio = positive unique reviews / all relevant unique reviews for the aspect
The denominator includes positive, negative, mixed and neutral, and excludes reviews where the aspect does not appear.
```

`polarized_positive_ratio = positive / (positive + negative)` is also output for when a binary measure is needed; neutral and mixed counts are always kept. A zero denominator returns null, never 0% or 100%. All ratios are over the retrieved_sample and do not represent overall consumer satisfaction. Statistics are computed by code; the LLM does not compute or write ratios.

## Group B interface

Accepts the flat fields of Group B's `Hit.to_dict()`, as well as candidates with nested metadata; `chunk_id`, `doc_id` and `text` are required. Duplicate chunk_ids are deduplicated automatically; conflicting data raises an error.

```python
from generation.client import DeepSeekClient
from generation.pipeline import enrich_reply

# reply comes from app.pipeline.search_full_corpus, or the existing reranker flow.
# reply['chunks'] and reply['rewrite']['aspects'] are read automatically.
reply = enrich_reply(reply, question,
                     client=DeepSeekClient(cache_dir='.cache/group_c'),
                     verify=False)  # current baseline; set True for the comparison run
print(reply['answer'])
print(reply['statistics'])
print(reply['flagged_sentences'])
```

Streamlit already has a local integration switch. After installing its dependencies per Group B's instructions, run:

```bash
streamlit run app/streamlit_app.py
```

Tick `Answer & sentiment (DeepSeek)` in the sidebar. Turn `Check summary evidence` off for the current baseline and on for the checker comparison. Ratios are shown under the answer, and the per-sentence check is also shown when checking is on. Generation is off by default to keep Group B's original retrieval workflow; the Group C CLI runs standalone without Streamlit or a retrieval index. The November 5 validation loaded the complete 459,418-chunk index and ran five live search-to-answer queries. UI behavior is also covered by automated Streamlit tests. See `results/local_validation_20261105/VALIDATION_REPORT.md` for the tested settings and limits.

## Summarizer and checker

The summary has at most six sentences, each citing chunk_ids present in this run's input. Input reviews are treated only as data, never as system instructions; the prompt forbids filling gaps with outside knowledge and over-generalizing personal experience, and requires keeping opposing opinions.

The checker has two layers. Code rejects nonexistent citations before inference. A separate API call judges whether the whole sentence is supported; code then validates every returned quote against the cited source. If any returned evidence is invalid, that sentence becomes unverifiable even when other quotes are valid. Independent sentences keep their own judgments. The checker uses a versioned cache namespace so verdicts sanitized by older validation code cannot bypass the repair. This changes cache identity without changing the model request. A sentence that is only partly supported must be labeled partially_supported. Citations that do not exist are labeled unsupported directly, without calling the model.

Check statuses are supported, partially_supported, unsupported, contradicted and unverifiable; everything except supported is flagged. Results keep the original draft, each sentence's judgment, the reason and the verbatim quotes; by default the final answer keeps only supported sentences. If the checker service fails, the draft is kept for inspection and every sentence is marked unverified; a failure is never treated as a pass.

This is model-assisted checking and cannot guarantee the absence of hallucinations. Using the same service to generate and to check introduces correlated bias; real quality should be validated by independent human annotation and blind evaluation, and model self-scores should not be reported as accuracy.

Use `run --no-check` to keep an unverified draft. For a controlled checker ablation, freeze the same draft and evidence and run `evaluation.report.run_paired_checker`; running the whole pipeline twice also changes the summary and does not isolate the checker. See [paired checker instructions](evaluation/report/PAIRED_CHECKER_README.md). Historical checker runs remain available but are not a controlled off/on comparison.

## Evaluation outputs

`run` produces the full JSON, a readable `.md` report with the same name, a `.rag.jsonl` and a `.timings.jsonl`. The last two match the fields required by the evaluation branch. The timing export includes the measured `total_s` and `scope=downstream`. The latency evaluator uses that total directly rather than adding it to stages. The full search-to-answer benchmark is `scripts.benchmark_pipeline`; its outer timer includes rewriting, retrieval, reranking, classification, summary and checking. Startup/prewarm and per-query time are reported separately. Serial throughput estimates are not concurrent load measurements.

The existing labels in the human eval_sample.csv are whole-review sentiment, so official classification predictions must be run without `--aspect`; aspect-level predictions must not be compared directly with whole-review human labels:

```bash
# Try only 12 rows first to confirm the output; this is not an official accuracy evaluation.
python -m scripts.group_c classify --input evaluation/eval_sample.csv --limit 12 --output evaluation/predictions/deepseek_overall_smoke.csv

# 1200 rows in batches of 12 need at least 100 calls; raise the budget explicitly.
python -m scripts.group_c classify --input evaluation/eval_sample.csv --max-api-calls 150 --output evaluation/predictions/deepseek_overall.csv

# Requires the completed human-annotated eval.xls and the evaluation branch's dependencies.
python evaluation/eval_classifier.py evaluation/predictions/deepseek_overall.csv
```

The prediction CSV is strictly `chunk_id,label`; extra details go to a `.details.json` with the same name. Model predictions are never passed off as human gold or written into human label files. The current sample contains 1,200 records with two annotations each; the unchanged `eval.xls` contains 1,194 scored records after six exclusions. Recomputed classification metrics use these existing labels. Human annotations, exclusions and adjudication decisions were not changed by this code audit; outstanding annotation-quality limitations still apply.

RAGAS is not called automatically here, and paid inference is not run automatically over all 1200 rows. Formal evaluation needs human gold labels, real queries and frozen retrieval results. Group B's six fixtures and the deliberately constructed checker_challenge are development examples only; they are not a test set and do not support course conclusions.

## API calls and reproducibility

By default each process makes at most 32 API requests, including retries. Requests with the same endpoint, model, prompt and input hit the local cache and are not paid for again. Use `--no-cache` to measure uncached latency; cache hits are recorded as cache_hits and their tokens are not counted again. Real cost must be computed from the actual model and official pricing; the code does not guess dollar amounts.

Each classification batch holds at most 12 rows, and run accepts at most 64 chunks and 60000 characters; anything larger raises an error rather than being silently truncated. If a large classification job fails partway, rerun it: successful batches reuse the cache, and the prediction file is replaced only after everything succeeds. Different prompts or inputs naturally produce different cache keys.

Official API documentation: https://api-docs.deepseek.com/ . Model names and prices may change; the model and usage recorded in run configs and results are for reproducibility. Do not share `.env`, `.cache/` or any local configuration containing credentials.

## Local validation and rerunning

Install Python 3.13+ dependencies with `uv sync --all-groups`, or `python -m pip install -r requirements.txt` in an isolated environment. The requirements file is now exported from `uv.lock` with all groups, including FAISS and evaluation dependencies. Run `python -m pytest tests -q` to include both retrieval/UI and nested Group C tests.

The complete local results and reproducible commands are in [the validation report](results/local_validation_20261105/VALIDATION_REPORT.md). Old result files are kept as historical records; corrected aggregates are written to a separate folder. Do not infer a new classification score from a product-grouping or checker change.

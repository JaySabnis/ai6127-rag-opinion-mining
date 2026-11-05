# Local code fixes and validation — 5 November 2026

This revision starts from main commit `d5b1f5bac905e382630b746fa25116c8e5081907`. Validation was completed locally on `local-audit-fixes` before publication. The recorded source manifest preserves that validation-time status. Human labels were not corrected or regenerated. Eleven protected annotation/sample/gold/relevance files match HEAD byte for byte; hashes are in `protected_data_verification.json`.

## Changes

1. **Product ratios:** retrieval now carries a stable product ID and canonical display metadata. Generation groups aliases using the same rules, preserves the original model names, and keeps Plus/5G/FE/Ultra variants separate. Unknown products without IDs remain separated by review. Conflicting metadata fails explicitly.
2. **Evidence checking:** one valid quote can no longer rescue a sentence whose other evidence is invalid. The sentence is downgraded, diagnostics remain visible, and independent sentences are unaffected. The checker has a versioned cache namespace so older sanitized `supported` records cannot bypass the fix; request bodies are unchanged.
3. **Abstentions and failures:** ordinary pipeline exports now distinguish empty claim sets from display placeholders. Checker failure is explicit; it cannot be counted as a successful answer by RAG evaluation.
4. **Controlled comparison:** the paired runner freezes questions, chunks, draft sentences and predictions, then applies only the checker. It records source/input hashes, retries, token usage, coverage and failures. Resume refuses stale inputs or code. The old runner no longer presents two executions of the same checker as different versions.
5. **Evaluation and timing:** duplicate label IDs fail rather than being overwritten. RAG score merges require matching input, row order, configuration, source version and score-file hashes. Explicit total time is used once. First query is not called cold start; serial estimates are not called concurrent throughput.
6. **Reproduction:** requirements were regenerated from `uv.lock` with all groups, stale UI assertions were repaired, and regression tests were added. Documentation now describes the actual evaluation scope.

## Automated verification

- **147 tests passed, zero failures/errors/skips**, including retrieval, UI behavior, Group C integration, legacy-cache isolation, aliases, evidence rejection, unchanged statistics sources and evaluation integrity. Pytest additionally reports 59 subtests; the JUnit XML counts 206 cases in total. See `pytest.xml`.
- Python 3.13.13, dependencies synchronized to the existing lock; `pip check` and `git diff --check` passed. See `environment.json` and `requirements-tested.txt`.
- The complete 459,418-chunk catalogue contains 4,531 unique brand/model pairs. Old and shared parser identity rules produced the same keys on all of them (`catalogue_compatibility.json`). No retrieval scoring model or relevance labels were changed.

## Statistics and classification

All 200 saved B40 runs were reaggregated without model calls. Eight saved runs had numerical grouping corrections; historical predictions and result files remain untouched. In the learned run, Q012 is **3/10 = 30%** and Q030 is **3/7 = 42.86%**. These are retrieved-sample ratios over relevant unique reviews, not population satisfaction rates. See `statistics/changes.csv` and the per-question before/after files.

Using the same 1,194 valid gold records and existing predictions:

| Classifier | Accuracy | Macro-F1 |
|---|---:|---:|
| DeepSeek | 0.9389 | 0.8646 |
| VADER | 0.6248 | 0.3895 |

The DeepSeek scores did not change. Subjectivity macro-F1 is 0.8297; polarity macro-F1 is 0.9374. Neutral recall remains 17/31 = 0.5484. VADER cannot output mixed, which limits its four-class score. `classification/` contains exact values, input hashes and per-class results. These figures do not resolve the previously identified annotation-quality questions.

## Paired checker experiment

Both arms use the same 40 questions, contexts and order, with the same 240 frozen draft sentences. The final checker run completed 40/40, retained **215/240 sentences (89.58%)**, and retained at least one answer for every query. There were 23 unverifiable sentences and two partially supported sentences. These are model-assisted flags, not human-confirmed hallucinations. See `flagged_summary_sentences.csv`.

RAGAS 0.4.3 used DeepSeek-flash as judge, temperature 0, thinking disabled, and BGE-small embeddings for answer relevancy. All four metrics scored all 40 rows in each arm with no judge failures or skipped abstentions.

| Metric | Checker off | Checker on | Mean paired change | Valid pairs |
|---|---:|---:|---:|---:|
| faithfulness | 0.9731 | 0.9661 | -0.0070 | 40 |
| answer_relevancy | 0.8435 | 0.8373 | -0.0061 | 40 |
| context_precision | 0.8305 | 0.8472 | +0.0167 | 40 |
| context_relevance | 0.9875 | 0.9875 | +0.0000 | 40 |

**This experiment does not show a quality improvement from enabling the checker.** Faithfulness and answer relevancy are slightly lower. Small differences may reflect evaluator variability; this is not a significance test. Context precision uses the generated answer as its reference, so its increase does not demonstrate better retrieval. Contexts are identical. Generation and evaluation use the same provider, so scores are not independent human accuracy. Detailed scores and provenance are in `paired_ragas/`; comparison alignment is in `paired_comparison.json`.

## Five live full-pipeline queries

The final run loaded the complete 459,418-chunk index, used top-10 per product and pool-50, the learned reranker on MPS, DeepSeek-flash rewriting/classification/summarization/checking, and no application response cache. No simultaneous evaluation job was running for this final batch. All five returned answers without degraded-stage errors.

| Query | Question | Returned chunks | Total seconds |
|---|---|---:|---:|
| Q001 | How do reviewers rate the camera quality on the Samsung Galaxy S10? | 10 | 8.592 |
| Q007 | How long does the Samsung Galaxy S10 last between charges in everyday use? | 10 | 8.495 |
| Q014 | What screen or touch-response issues do iPhone X reviewers describe? | 10 | 8.932 |
| Q021 | What do reviewers say about the Google Pixel 4a speed when running apps? | 10 | 8.322 |
| Q032 | What experiences do iPhone 11 buyers report when using SIM cards with different carriers? | 10 | 10.070 |

Mean **8.882 s**, median **8.592 s**, descriptive p95 **9.842 s**. The timer includes rewriting, retrieval, reranking, classification, summary and checking. Index/encoder loading (4.652 s), local search/reranker warmup (0.264 s) and output serialization are outside individual query totals. Model weights were already cached; these startup values are not clean-machine cold-start measurements.

The serial estimate is 6.75 queries/minute. Five requests do not establish concurrency, scalability or production SLA. The original B40 **6.005 s** remains a valid downstream-only observation; it excluded retrieval and cannot be compared directly to this full-pipeline run. See `latency/` and `end_to_end/`.

## Usage and cost boundary

The final checker experiment recorded 41 API requests, including a structured-output retry. Recorded final-batch estimates are **$0.01612** for the checker and **$0.00581** for the five-query downstream classification/summary/check stages. These use the [official pricing page](https://api-docs.deepseek.com/quick_start/pricing/) checked on 5 November 2026 and the off-peak rate applicable to these runs: $0.15/M uncached input, $0.003/M cached input and $0.60/M output tokens.

These estimates are not account billing or the total expense of this audit. They exclude earlier validation batches, query rewriting, RAGAS judging and local compute. Token counts, cache splits, formula and source hashes are in `usage_estimates.json`; the unverified historical RMB cost calculation is not reused.

## Reproduce and remaining limits

See [the run guide](../../LOCAL_RUN_GUIDE.md) for offline tests, read-only rescoring, the paired checker experiment and the full-index benchmark. API tests incur usage; the package contains no credentials or full index.

Human labels and adjudication decisions remain unchanged as requested. The 40-question retrieval evaluation informed earlier development and is not a historically untouched blind test. No new human evaluation, concurrency/load experiment or final course submission PDF is created by this code repair. The checker still needs independent quality evaluation before claiming an improvement. Earlier batches ending in `initial` or `before_cache_isolation` remain only on the development machine and are excluded from the published commit; the figures above use the final directories explicitly named here.

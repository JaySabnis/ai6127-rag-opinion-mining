# Frozen-draft checker comparison

This experiment compares the stored `B40_learned` draft with the current evidence
checker applied to that same draft. Both arms use the same saved review chunks,
sentences, classification predictions and statistics. It does not retrieve
reviews again or call the classifier or summary generator. It does not reproduce
the historical checker or measure end-to-end system throughput.

Run these commands from the repository root. The environment file must contain
`DEEPSEEK_API_KEY`; it may also contain `DEEPSEEK_MODEL` and `DEEPSEEK_BASE_URL`.
The key is read locally and is not saved in experiment artifacts.

```sh
python -m evaluation.report.run_paired_checker \
  --env-file /absolute/path/to/local.env \
  --output-dir results/paired_checker \
  --limit 2 --max-api-calls 4

python -m evaluation.report.run_paired_checker \
  --env-file /absolute/path/to/local.env \
  --output-dir results/paired_checker \
  --resume --max-api-calls 80
```

`--limit` selects the first N query files. All input files are validated and
frozen before the first paid call, so a small trial can be extended with
`--resume`. `--max-api-calls` limits actual requests in that invocation, including
retries. It is not a lifetime budget across separate invocations. Successful
pairs are skipped on resume; failed or interrupted pairs are retried, retaining
their earlier attempt records. A new nonempty output directory is rejected.
Resume is refused if input bytes, relevant source code, model, endpoint or
pricing differ. Use a new output directory for a changed experiment.

The default input directory is `results/group_c/b40/B40_learned`. Override it
with `--input-dir` only when the replacement contains the same complete frozen
run schema. Missing drafts are never generated automatically.

## Saved evidence

- `manifest.json`: experiment identity, source hashes, code hashes, Git revision,
  dirty state, model, endpoint and completion status.
- `frozen_inputs/Q*.json`: byte-for-byte copies of the source files.
- `pairs/Q*.json`: shared frozen input, original answer, checked sentences,
  retained answer, failures, attempt history, elapsed checker time and API usage.
- `rag_outputs/checker_off.jsonl` and `checker_new.jsonl`: the same successfully
  completed pairs in each arm, suitable for subsequent evaluation.
- `summary.json`: completed/pending/failed queries, token totals, recorded costs,
  answer coverage and sentence retention.

When no sentence passes the checker, the checked arm is a completed abstention:
`answer` is empty and `abstained` is true. A separate `display_answer` is provided
for a user interface. It is not inserted into evaluation answers. An API failure
has `status: failed` and `answer: null`; it is excluded from both paired exports.
Always report completion coverage and abstention coverage alongside any answer
quality score. Removing claims can improve a faithfulness score while reducing
useful answer coverage.

The checker conservatively downgrades a whole sentence to `unverifiable` if any
supplied evidence quote is malformed, uncited or absent from its cited review.
It retains valid evidence and a diagnostic in `reason`, and independently valid
sentences remain available. Cached validation cannot restore an approved status
after that downgrade. The model's checker judgments remain automated judgments,
not independent human ground truth.

## Cost accounting

Raw provider usage and request counts are always saved. A request without usage
is explicitly counted as unknown. Costs remain `null` unless `--pricing-file`
points to a JSON object with these fields:

```json
{
  "input_per_million": 0,
  "cached_input_per_million": 0,
  "output_per_million": 0,
  "currency": "USD",
  "source_url": "https://official-provider-pricing-page.example",
  "as_of": "YYYY-MM-DD"
}
```

The zero values above are schema examples, not provider prices. Replace them
with verified rates before running. Estimated costs cover recorded usage only;
requests without usage may still have unknown cost. Checker time covers only
this new intervention. The frozen off arm incurs no new inference time, so it
cannot serve as a measured zero-latency end-to-end baseline.

## Offline regression checks

```sh
python -m unittest tests.group_c.test_verifier_regression tests.group_c.test_paired_checker -v
```

These tests use fake transports and temporary files. They make no network calls
and do not modify source datasets or human labels.

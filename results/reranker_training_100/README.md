# Reranker Training Validation

The fusion layer is fitted on 100 dedicated questions with 20 fixed candidates each. MiniLM remains frozen. Product-group cross-validation selects from 32 settings; nested validation estimates internal performance. No evaluation40 labels participate in fitting or selection.

- Nested validation: P@10=0.8450, MRR@10=0.9783, nDCG@10=0.8635.
- Training fit: P@10=0.8520, MRR@10=0.9783, nDCG@10=0.8684.

The selected model uses seven window-standardized features, ridge penalty 100 and query-centered relevance grades 0–3. See the [reranker guide](../../retrieval/RERANKER_README.md) for the objective and inference details.

## Files

- [training_statistics.json](training_statistics.json): folds, settings, selected model configuration and validation scores.
- [training_oof.csv](training_oof.csv): per-question nested validation results.
- [execution_manifest.json](execution_manifest.json): artifact hashes and training protocol.
- [Model parameters](../../retrieval/models/review_ranker_v2.json): the single production parameter file.

```powershell
uv run python -X utf8 -m scripts.train_ranker_100
```

Use the [ablation workflow](../ablation_latest/README.md) to evaluate the frozen model. Training fit, internal validation and evaluation40 are separate protocols; evaluation40 influenced earlier development and is not an untouched blind test.

# Query Rewrite and Reranker

`query_rewrite.py` calls DeepSeek V4 Pro and returns an English query, products and aspect labels.
`rerank.py` ranks existing candidates with MiniLM; it does not retrieve or filter products.

## Usage

Run `uv sync` at the repository root to install dependencies (Python >=3.13).
Set `DEEPSEEK_API_KEY` in the root `.env` file; see `.env.example` for the template:

```dotenv
DEEPSEEK_API_KEY=your-api-key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-pro
```

Keep `.env` local and do not commit API keys. Process environment variables override the file. MiniLM alone does not require a DeepSeek API key.

```python
from retrieval.query_rewrite import rewrite_query
from retrieval.rerank import CrossEncoderReranker, RerankerConfig

rewritten = rewrite_query("How is the battery life on the Galaxy S10?")
# Demo input; replace with candidates returned by the retriever.
candidates = [{
    "chunk_id": "demo_0", "doc_id": "demo", "text": "The battery lasts all day.",
    "brand": "Samsung", "model": "Galaxy S10", "aspect_tags": ["battery"],
}]
# Reuse the same reranker instance across queries.
reranker = CrossEncoderReranker(RerankerConfig(aspect_weight=0.5))
ranked = reranker.rerank(
    rewritten["query"], candidates,
    query_aspects=rewritten["aspects"], top_k=5,
)
```

Query rewrite returns:

```json
{
  "query": "How is the battery life on the Samsung Galaxy S10?",
  "product": [{"brand": "Samsung", "model": "Galaxy S10"}],
  "aspects": ["battery"]
}
```

`product` supports multiple products and is `[]` when none are specified. An unknown brand or model is `null`.
`aspects` allows only camera, battery, display, performance, service and carrier; it is `[]` when none apply. Unknown labels are rejected.
Query rewrite does not use conversation history. Extracted products are not guaranteed to exist in the corpus and do not automatically filter candidates.

## Ranking Rules

Each candidate must contain `chunk_id`, `doc_id` and `text`. The brand, model and aspect_tags fields may be at the top level or inside metadata; top-level fields take precedence.

1. Prefix model input with product context, such as `This is a review of Samsung Galaxy S10.`, without changing the stored text.
2. MiniLM produces `cross_encoder_score`. The aspect reward is `aspect_weight * distinct_match_count * gate`, where `gate = sigmoid((cross_encoder_score - aspect_threshold) / aspect_temperature)`.
3. Set `rerank_score = cross_encoder_score + aspect_bonus`, sort descending, take Top-K, and assign `final_rank`.

Returned dictionaries preserve existing fields and add aspect_match_count, aspect_gate, aspect_bonus and the scores above. Only the top-level dictionary is copied; nested metadata is shared with the input and is not modified by this pipeline.

Defaults are aspect_weight=0.5, aspect_threshold=-5 and aspect_temperature=1; these have not been tuned. Lower model scores receive smaller rewards. Half the reward is retained at the threshold. Temperature must be positive; smaller values produce a sharper transition. The gate is not a relevance probability.

Omitting query_aspects disables bonuses; aspect_weight=0 also disables them. CLI flags `--aspect-threshold` and `--aspect-temperature` adjust the gate. The UI uses the defaults.
`top_k=None` returns all candidates; `top_k=0` returns an empty list. `rerank_with_details()` also returns timings and configuration.

The default model is `cross-encoder/ms-marco-MiniLM-L6-v2`, running on CPU with batch_size=16 and max_length=512. The model is downloaded on first use and cached afterward.
The 512-token limit includes the query, product context, chunk and special tokens. The tokenizer truncates longer input.
Raw MiniLM scores may be positive or negative; higher is more relevant. Final scores include aspect bonuses. Neither score represents sentiment.

`rerank_with_details()` returns candidates, input_count, timing_ms and config. Timings are in milliseconds: model_load covers initial loading during this call, rerank covers scoring and sorting, and total covers the full reranking call. These exclude the external query rewrite API call.

## Standalone Commands

```powershell
uv run python -m retrieval.query_rewrite "How is the battery life on the Galaxy S10?"
uv run python -m scripts.run_reranking --input tests/fixtures/rerank_candidates.json --query "Galaxy S10 battery life" --aspects battery --top-k 3 --output results/reranking/demo.json
```

The reranking CLI accepts a JSON candidate array, an object containing a candidates array, or JSONL. It does not call query rewrite.
It reads up to 1,000 candidates by default. Once the model is cached, use `--local-files-only` to run MiniLM offline.
The six real sample chunks are development data, not a formal evaluation set. Run core checks with `uv run python -m unittest discover -s tests -v`.

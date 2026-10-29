# Streamlit UI

Set `DEEPSEEK_API_KEY` in `.env` at the repository root (see [API configuration](../retrieval/README.md)), then start the app:

```powershell
uv sync
uv run streamlit run app/streamlit_app.py
```

Open `http://localhost:8501`. Press `Ctrl+C` to stop the server.

The UI supports chat history, evidence inspection and an optional reranking switch. Each question is processed independently. Answer generation is not connected yet.

## Candidate sources (sidebar)

| source | what happens to a question |
|---|---|
| **Sample data** (default) | DeepSeek rewrite, then the six Galaxy S10 development chunks are reranked |
| **Upload candidates** | same, on an uploaded JSON/JSONL pool (up to 1,000 records and 5 MB; see [candidate formats](../retrieval/README.md)) |
| **Full corpus** | DeepSeek rewrite → product names checked against the corpus → hybrid retrieval over all 459,418 chunks (50 candidates per product) → MiniLM rerank → top K |

**Full corpus** needs the shared retrieval indexes in `indexes/` (see the
[download and setup instructions](../README.md#hybrid-retrieval)). They are
loaded once, on the first question (about 5–7 s). Extra controls and behaviour
in this mode:

- **Query rewrite (DeepSeek)** switch. Off, or when the call fails, the
  original question is searched and the retriever's rule-based parser picks
  the model filter (the evidence panel says so). The same pipeline measured
  with `scripts/test_pipeline.py` on a MacBook (MPS): ~1.6 s per question
  with the rewrite, ~0.2 s without.
- Comparison questions ("S10e or Pixel 3a?") are retrieved and reranked per
  product, so each product gets its own top K.
- Retrieval aspect boost is enabled with weight 0.5 during candidate selection.
  The UI has no separate aspect-boost switch. The default reranker uses the
  v2 fusion model trained on 100 separate questions, reorders the first 20
  candidates, and preserves the remaining retrieval order. MiniLM stays frozen;
  the default does not add the legacy gated aspect reward.
- The evidence panel shows, per product, the filter used (model and all its
  corpus spellings, or brand), how many reviews were searched, and the text
  actually searched.

The full-corpus flow lives in `app/pipeline.py` (`search_full_corpus()`), which
has no Streamlit dependency; `streamlit_app.py` only calls it. The reranker
device is chosen automatically (CUDA > MPS > CPU): on macOS a CPU reranker in
the same process as FAISS crashed with a segmentation fault in testing.

## Integration

Add generated text as an `answer` field in the reply returned by `search()` /
`search_full_corpus()` to display it in the conversation.

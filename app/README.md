# Streamlit UI

Set `DEEPSEEK_API_KEY` in `.env` at the repository root (see [API configuration](../retrieval/README.md)), then start the app:

```powershell
uv sync
uv run streamlit run app/streamlit_app.py
```

Open `http://localhost:8501` in your browser. Press `Ctrl+C` in the terminal to stop the server.

The wide layout separates conversation on the left from evidence and candidates on the right. The evidence selector lets you inspect past successful questions without rerunning them. Evidence appears only after a successful question; loading a candidate file does not display results. Chat text and the multiline input are enlarged.

Each question runs query rewrite, then MiniLM reranking with aspect bonuses. The page shows the rewritten query, extracted products/aspects, chunks, sources and timings. Questions can be in English or other languages; the rewritten retrieval query is always English.

Turning reranking off still runs query rewrite but preserves the original candidate order. History is stored only in the current session and is not passed to query rewrite.

Settings changes affect the next submission, without recomputing past replies. Submitting the same question again makes a new API call; query rewrite is not cached or deduplicated. History is not guaranteed to survive a refresh or a closed page.

The app currently uses six Galaxy S10 sample chunks or uploaded JSON/JSONL candidates (up to 1,000 records and 5 MB). Full-corpus retrieval, product filtering and answer generation are not connected yet.

Candidate files may be a JSON array, an object containing `{"candidates": [...]}`, or JSONL with one record per line. See the [candidate example](../retrieval/README.md). The question does not change the loaded candidate pool.

The dark interface displays ratings as stars and aspects as rounded tags. Chunk and review IDs are hidden, and scoring is summarized as the final score and aspect bonus. The page shows total processing time; detailed timings remain in the saved result. Missing API keys and API failures are displayed as errors; reranking does not continue after a rewrite failure.

To connect retrieval, update `search()` in `streamlit_app.py`: retrieve candidates using `rewritten["query"]` and pass the returned pool to the reranker.
To connect answer generation, add the generated text as an `answer` field in the saved reply returned by `search()`. It will appear in the conversation while chunks remain in the evidence panel.

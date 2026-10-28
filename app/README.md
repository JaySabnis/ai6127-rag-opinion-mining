# Streamlit UI

Set `DEEPSEEK_API_KEY` in `.env` at the repository root (see [API configuration](../retrieval/README.md)), then start the app:

```powershell
uv sync
uv run streamlit run app/streamlit_app.py
```

Open `http://localhost:8501`. Press `Ctrl+C` to stop the server.

Questions are rewritten into English, then candidates are reranked with MiniLM and aspect bonuses. The UI supports chat history, evidence inspection and an optional reranking switch. Each question is processed independently.

Use the six Galaxy S10 sample chunks or upload JSON/JSONL candidates (up to 1,000 records and 5 MB). See [candidate formats](../retrieval/README.md). Full-corpus retrieval, product filtering and answer generation are not connected yet.

## Integration

In `streamlit_app.py`, update `search()` to retrieve candidates using `rewritten["query"]`. Add generated text as an `answer` field in the returned reply to display it in the conversation.

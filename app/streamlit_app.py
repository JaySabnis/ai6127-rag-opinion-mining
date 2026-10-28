"""Chat interface for query rewrite, reranking, and review-chunk inspection.

Input : user questions and sample/uploaded candidate chunks
Output: rewritten query, extracted products/aspects, and expandable evidence

Each new question runs rewrite -> rerank once. Saved replies are displayed from
session history; full-corpus retrieval and answer generation are not connected.
"""

import json
from html import escape
from pathlib import Path
import sys
from time import perf_counter
from urllib.parse import urlparse

import streamlit as st

# Streamlit runs this file as an entry point; make repository imports available.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retrieval.rerank import CrossEncoderReranker, RerankerConfig
from retrieval.query_rewrite import rewrite_query
from app.pipeline import search_full_corpus

INDEX_DIR = ROOT / "indexes"
FULL_CORPUS = "Full corpus"


# ---- candidate input
def parse_candidates(data: bytes, filename: str) -> list[dict]:
    """Read a bounded candidate pool, preserving all original fields."""
    if len(data) > 5 * 1024 * 1024:
        raise ValueError("Candidate files must not exceed 5 MB. Do not upload the full corpus.")
    text = data.decode("utf-8-sig")
    if filename.lower().endswith(".jsonl"):
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        payload = json.loads(text)
        rows = payload.get("candidates") if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or len(rows) > 1000:
        raise ValueError("The file must contain a list of up to 1,000 candidates.")
    for row in rows:
        if not isinstance(row, dict) or any(
            not isinstance(row.get(key), str) or not row[key].strip()
            for key in ("chunk_id", "doc_id", "text")
        ):
            raise ValueError("Each candidate must contain nonempty chunk_id, doc_id and text fields.")
        metadata = row.get("metadata")
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError("metadata must be a JSON object.")
        aspects = row.get("aspect_tags", (metadata or {}).get("aspect_tags"))
        if aspects is not None and (
            not isinstance(aspects, list) or any(not isinstance(a, str) for a in aspects)
        ):
            raise ValueError("aspect_tags must be a list of strings.")
    return rows


@st.cache_resource
def get_reranker() -> CrossEncoderReranker:
    """Reuse one lazy-loading model instance instead of loading on every UI rerun.

    Device cuda > mps > cpu: on macOS a CPU reranker in the same process as
    FAISS crashed with a segmentation fault in testing.
    """
    from retrieval.dense import pick_device

    return CrossEncoderReranker(RerankerConfig(device=pick_device()))


@st.cache_resource(show_spinner="Loading the retrieval indexes (first time only)...")
def get_retriever():
    """BM25 + FAISS + BGE over the full corpus, loaded once per server process."""
    from retrieval import Retriever

    return Retriever.load(INDEX_DIR)


# ---- query processing (future retriever integration point)
def search(query: str, candidates: list[dict], top_k: int, rerank: bool) -> dict:
    """Rewrite once, optionally rerank, and save the result for history display."""
    started = perf_counter()
    rewritten = rewrite_query(query)
    rewrite_ms = (perf_counter() - started) * 1000
    # Use rewritten["query"] for retrieval; reserve product for product filtering.
    if rerank:
        result = get_reranker().rerank_with_details(
            rewritten["query"], candidates, top_k=top_k,
            query_aspects=rewritten["aspects"],
        )
        chunks, timing = result.candidates, result.timing_ms
    else:
        # Disabling reranking keeps candidate order; rewrite still runs.
        chunks = candidates[:top_k]
        timing = {"model_load": 0.0, "rerank": 0.0}
    return {
        "chunks": chunks, "reranked": rerank, "candidate_count": len(candidates),
        "rewrite": rewritten,
        "timing_ms": {**timing, "rewrite": rewrite_ms,
                      "total": (perf_counter() - started) * 1000},
    }


# ---- conversation and evidence display

def render_reply(reply: dict) -> None:
    """Keep the conversation focused on answers; evidence lives in the right panel."""
    if "error" in reply:
        st.error(reply["error"])
    elif reply.get("answer"):
        # The generation module can provide this field without changing the layout.
        st.markdown(reply["answer"])
    elif reply["chunks"]:
        count = len(reply["chunks"])
        reviews = len({chunk["doc_id"] for chunk in reply["chunks"]})
        st.write(f"Found {count} chunks from {reviews} reviews.")
        st.caption("Answer generation is not connected yet.")
    else:
        st.info("No evidence found in the current candidate pool.")


def render_aspects(aspects: list[str]) -> None:
    """Display metadata labels as pills, escaping uploaded text before using HTML."""
    if aspects:
        tags = "".join(f'<span class="aspect-tag">{escape(tag)}</span>' for tag in aspects)
        st.markdown(f'<div class="aspect-tags">{tags}</div>', unsafe_allow_html=True)


def render_evidence(reply: dict) -> None:
    """Inspect one saved query result without running rewrite or reranking again."""
    rewritten = reply.get("rewrite")
    if rewritten:
        st.text(rewritten["query"])
        products = [" ".join(p[k] for k in ("brand", "model") if p.get(k))
                    for p in rewritten["product"]]
        if products:
            st.caption(", ".join(products))
        render_aspects(rewritten["aspects"])
    if reply.get("rewrite_error"):
        st.caption(f"Query rewrite failed, searched the original question ({reply['rewrite_error']})")
    for group in reply.get("retrieval") or []:
        f = group["filters"]
        scope = (f"Model: {', '.join(group['model_variants'] or [f['model']])}" if f.get("model")
                 else f"Brand: {f['brand']}" if f.get("brand") else "No product filter")
        size = f"{group['n_candidates']:,}" if group["n_candidates"] is not None else "all"
        st.caption(f"{scope} · {size} reviews searched · {group['hits']} candidates · "
                   f"searched for \"{group['search_text']}\"")
    chunks = reply["chunks"]
    timing = reply.get("timing_ms")
    if timing:
        ordering = "Reranked" if reply["reranked"] else "Original order"
        st.caption(f"{ordering} · {timing['total'] / 1000:.2f} s")
    for rank, chunk in enumerate(chunks, 1):
        metadata = {**(chunk.get("metadata") or {}), **chunk}
        product = " ".join(str(metadata.get(k) or "") for k in ("brand", "model")).strip()
        with st.expander(f"{rank:02d} · {product or 'Unknown product'}", expanded=rank == 1):
            st.text(chunk["text"])  # review content is displayed literally
            with st.container(horizontal=True, vertical_alignment="center"):
                rating = metadata.get("rating")
                if isinstance(rating, (int, float)) and 1 <= rating <= 5:
                    st.markdown(f'<span class="review-stars" aria-label="{rating} out of 5 stars">'
                                f'{"★" * int(rating)}</span>', unsafe_allow_html=True)
                if metadata.get("timestamp"):
                    st.caption(str(metadata["timestamp"]))
            render_aspects(metadata.get("aspect_tags") or [])
            if reply["reranked"]:
                st.caption(f"Score {chunk['rerank_score']:.2f}"
                           f" · Aspect +{chunk.get('aspect_bonus', 0):.2f}")
            url = metadata.get("source_url")
            try:
                valid_url = isinstance(url, str) and urlparse(url).scheme in ("http", "https")
            except ValueError:
                valid_url = False
            if valid_url:
                st.link_button("Product source", url)


# ---- page settings and chat history
def main() -> None:
    st.set_page_config(page_title="Review Evidence Assistant", layout="wide")
    st.markdown("""
    <style>
    .stMainBlockContainer { max-width: 1800px; padding-top: 1.5rem; }
    .stMain p, .stMain [data-testid="stText"] { font-size: 1.1rem; line-height: 1.75; }
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] [data-testid="stText"] { font-size: 1.2rem; line-height: 1.8; }
    [data-testid="stChatInput"] textarea { font-size: 1.2rem; }
    [data-testid="stCaptionContainer"] p { font-size: 0.85rem; }
    [data-testid="stText"] { white-space: pre-wrap; overflow-wrap: anywhere; }
    .st-key-chat_history, .st-key-evidence_panel { background: #1B2330; }
    .aspect-tags { display: flex; flex-wrap: wrap; gap: 0.4rem; margin: 0.3rem 0 0.7rem; }
    .aspect-tag { border: 1px solid #445674; border-radius: 999px; padding: 0.15rem 0.7rem;
                  color: #CBDCFA; background: #27344A; font-size: 0.85rem; }
    .review-stars { color: #F0BF64; font-size: 1.25rem; letter-spacing: 0.12rem; }
    </style>
    """, unsafe_allow_html=True)
    st.title("Review Evidence Assistant")
    st.session_state.setdefault("messages", [])
    candidates, source_label, ready = [], "", False
    with st.sidebar:
        st.subheader("Query settings")
        source = st.radio("Candidate source", ["Sample data", "Upload candidates", FULL_CORPUS],
                          key="source")
        use_rewrite = True
        try:
            if source == FULL_CORPUS:
                if not (INDEX_DIR / "meta.pkl").exists():
                    raise ValueError(f"no indexes in {INDEX_DIR}; run scripts.build_retrieval_indexes first")
                source_label, ready = "Full corpus (hybrid retrieval)", True
                use_rewrite = st.toggle("Query rewrite (DeepSeek)", value=True, key="use_rewrite")
            elif source == "Sample data":
                fixture = ROOT / "tests/fixtures/rerank_candidates.json"
                candidates = parse_candidates(fixture.read_bytes(), fixture.name)
                source_label, ready = "Galaxy S10 development sample", True
            else:
                uploaded = st.file_uploader("Candidate file", type=["json", "jsonl"], max_upload_size=5)
                if uploaded is not None:
                    candidates = parse_candidates(uploaded.getvalue(), uploaded.name)
                    source_label, ready = uploaded.name, True
        except (OSError, ValueError, TypeError) as exc:
            st.error(f"Could not load candidates: {exc}")
        rerank = st.toggle("Rerank", value=True, key="rerank")
        top_k = st.select_slider("Number of chunks", options=[3, 5, 10], value=5, key="top_k")
        st.caption("Searching all reviews (50 candidates per product)." if source == FULL_CORPUS
                   else f"{len(candidates)} candidates loaded.")
        if st.button("Clear chat", key="clear_chat", width="stretch"):
            st.session_state.messages = []
            st.session_state.pop("evidence_turn", None)

    chat_column, evidence_column = st.columns([1.5, 1], gap="large")
    with chat_column:
        st.subheader("Conversation")
        conversation = st.container(height=580, border=True, autoscroll=True, key="chat_history")
        with conversation:
            if not st.session_state.messages:
                st.markdown("### What would you like to know?")
            # Settings and evidence selection do not recompute saved replies.
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    if message["role"] == "user":
                        st.text(message["content"])
                    else:
                        render_reply(message["content"])
        prompt = st.chat_input("Ask about a phone...",
                               disabled=not ready, max_chars=2000, height=110)
        # Only new submissions call the API and reranker.
        if prompt and prompt.strip():
            st.session_state.messages.append({"role": "user", "content": prompt})
            with conversation:
                with st.chat_message("user"):
                    st.text(prompt)
                with st.chat_message("assistant"):
                    try:
                        with st.spinner("Finding review evidence..."):
                            if source == FULL_CORPUS:
                                result = search_full_corpus(
                                    prompt, get_retriever(), top_k=top_k, rerank=rerank,
                                    reranker=get_reranker() if rerank else None,
                                    use_rewrite=use_rewrite, rewrite_fn=rewrite_query)
                            else:
                                result = search(prompt, candidates, top_k, rerank)
                            reply = {**result, "source": source_label}
                    except (RuntimeError, ValueError, OSError) as exc:
                        reply = {"error": f"Request failed: {exc}"}
                    render_reply(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})
            if "error" not in reply:
                st.session_state.evidence_turn = len(st.session_state.messages) - 1
            st.rerun()

    with evidence_column:
        st.subheader("Evidence & candidates")
        with st.container(height=760, border=True, key="evidence_panel"):
            # Select a saved successful turn; never infer again when switching evidence.
            messages = st.session_state.messages
            turns = [i for i, message in enumerate(messages)
                     if message["role"] == "assistant" and "error" not in message["content"]]
            if turns:
                if st.session_state.get("evidence_turn") not in turns:
                    st.session_state.evidence_turn = turns[-1]
                selected = st.selectbox("Evidence for", options=turns, index=None,
                    format_func=lambda i: f"{i // 2 + 1}. {messages[i - 1]['content']}",
                    key="evidence_turn")
                render_evidence(messages[selected]["content"])
            else:
                st.caption("No evidence yet.")


if __name__ == "__main__":
    main()

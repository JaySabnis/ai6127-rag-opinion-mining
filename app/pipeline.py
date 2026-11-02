"""Full-corpus search for the UI: rewrite -> resolve -> retrieve -> rerank.

Kept free of Streamlit so it can be tested on its own. Returns the same reply
fields ``streamlit_app.render_reply`` / ``render_evidence`` already use
(chunks, reranked, candidate_count, rewrite, timing_ms) plus ``retrieval``
(one entry per product searched) and ``rewrite_error`` when the rewrite failed.

Design (see retrieval/README.md): the DeepSeek rewrite identifies products and
aspects, ``resolve_model()`` maps each product onto a corpus model (else its
brand, else no filter), and the retriever's own parser is the fallback when the
rewrite fails or is switched off.
"""

from __future__ import annotations

from time import perf_counter
from typing import Callable


def _ms(t: float) -> float:
    return (perf_counter() - t) * 1000


def product_filter(retriever, product: dict) -> dict | None:
    """LLM product -> {"model": corpus model} | {"brand": corpus brand} | None."""
    model = retriever.parser.resolve_model(product.get("brand"), product.get("model"))
    if model:
        return {"model": model}
    brand = retriever.parser.resolve_brand(product.get("brand"))
    return {"brand": brand} if brand else None


def search_full_corpus(
    question: str,
    retriever,
    *,
    top_k: int,
    rerank: bool,
    reranker=None,
    use_rewrite: bool = True,
    rewrite_fn: Callable[[str], dict] | None = None,
    pool_k: int = 50,
    retrieval_aspect_weight: float = 0.5,
) -> dict:
    started, timing = perf_counter(), {}

    # 1. rewrite (optional); any failure falls back to the original question
    rewritten, rewrite_error = None, None
    if use_rewrite and rewrite_fn is not None:
        t = perf_counter()
        try:
            rewritten = rewrite_fn(question)
        except (RuntimeError, ValueError) as exc:
            rewrite_error = str(exc)
        timing["rewrite"] = _ms(t)

    # 2. retrieve: one pool per product (comparison questions search each product)
    t = perf_counter()
    if rewritten:
        query = rewritten["query"]
        filters = [product_filter(retriever, p) for p in rewritten["product"]] or [None]
        results = [retriever.retrieve(query, k=pool_k, auto_filter=False,
                                      aspects=rewritten["aspects"], filters=f,
                                      aspect_weight=retrieval_aspect_weight) for f in filters]
    else:
        query = question
        results = [retriever.retrieve(query, k=pool_k, aspect_weight=retrieval_aspect_weight)]
    timing["retrieve"] = _ms(t)
    aspects = results[0].parsed.aspects

    # 3. rerank each pool and keep top_k per product
    t = perf_counter()
    chunks = []
    for res in results:
        candidates = [h.to_dict() for h in res.hits]
        if rerank and reranker is not None and candidates:
            chunks += reranker.rerank(query, candidates, query_aspects=aspects, top_k=top_k,
                                     scoring_query=res.search_text)
        else:
            chunks += candidates[:top_k]
    if rerank and reranker is not None:
        timing["rerank"] = _ms(t)
    timing["total"] = _ms(started)

    return {
        "chunks": chunks,
        "reranked": bool(rerank and reranker is not None),
        "candidate_count": sum(len(res.hits) for res in results),
        "rewrite": rewritten or {"query": question, "product": [], "aspects": aspects},
        "rewrite_error": rewrite_error,
        "retrieval": [
            {
                "filters": res.filters,
                "model_variants": res.model_variants,
                "n_candidates": res.n_candidates,
                "search_text": res.search_text,
                "hits": len(res.hits),
                "aspect_weight": retrieval_aspect_weight,
            }
            for res in results
        ],
        "timing_ms": timing,
    }

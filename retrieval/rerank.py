"""Re-rank an existing pool of review chunks with MiniLM.

Input : query, candidate dictionaries, and query aspects from query rewrite
Output: ranked candidates with model score, aspect bonus, and final rank

Default: score model-stripped query/raw-text pairs and apply the trained review
ranker to the first 20 candidates. Rank fusion is available for the algorithm ablation.
This module does not retrieve or filter products.
"""

from dataclasses import asdict, dataclass
import math
from threading import Lock
from time import perf_counter

from chunking.metadata import ASPECT_KEYWORDS


DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
DEFAULT_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"


# ---- model and aspect-reward settings
@dataclass
class RerankerConfig:
    strategy: str = "learned"
    learned_model_path: str | None = None
    rerank_window: int = 20
    semantic_rank_weight: float = 0.8
    rank_offset: float = 10.0
    model_name: str = DEFAULT_MODEL
    revision: str | None = None
    device: str = "cpu"
    batch_size: int = 16
    max_length: int = 512
    local_files_only: bool = False



@dataclass
class RerankResult:
    """Ranked chunks plus timings in milliseconds and settings for this call."""
    candidates: list[dict]
    input_count: int
    timing_ms: dict[str, float]
    config: dict


def _metadata(chunk: dict) -> dict:
    """Read either candidate format; top-level fields override nested metadata."""
    return {**(chunk.get("metadata") or {}), **chunk}


# ---- reusable model and ranking pipeline
class CrossEncoderReranker:
    """Load the model on first use and reuse it across queries."""
    def __init__(self, config: RerankerConfig | None = None):
        self.config = config or RerankerConfig()
        self._model = None
        # Streamlit shares this instance across sessions.
        self._lock = Lock()

    def _load_model(self):
        """Use a fixed revision for the default MiniLM model for reproducibility."""
        from sentence_transformers import CrossEncoder
        from torch import nn

        revision = self.config.revision
        if revision is None and self.config.model_name == DEFAULT_MODEL:
            revision = DEFAULT_REVISION
        return CrossEncoder(
            self.config.model_name,
            revision=revision,
            device=self.config.device,
            max_length=self.config.max_length,
            local_files_only=self.config.local_files_only,
            trust_remote_code=False,
            activation_fn=nn.Identity(),  # Keep the raw relevance logit.
        )

    def rerank(self, query: str, candidates: list[dict], *, top_k: int | None = None,
               query_aspects: list[str] | None = None, scoring_query: str | None = None) -> list[dict]:
        """Return ranked candidates only; top_k=None keeps all, 0 skips scoring."""
        return self.rerank_with_details(
            query, candidates, top_k=top_k, query_aspects=query_aspects, scoring_query=scoring_query,
        ).candidates

    def rerank_with_details(self, query: str, candidates: list[dict], *,
                            top_k: int | None = None,
                            query_aspects: list[str] | None = None,
                            scoring_query: str | None = None) -> RerankResult:
        """Also return load/inference timings and the aspect settings used."""
        started = perf_counter()
        if not query.strip():
            raise ValueError("query must not be empty")
        if self.config.strategy not in {"rank_fusion", "learned"}:
            raise ValueError("Unknown reranking strategy")
        if self.config.rerank_window < 1 or not isinstance(self.config.rerank_window, int):
            raise ValueError("rerank_window must be a positive integer")
        if not 0 <= self.config.semantic_rank_weight <= 1:
            raise ValueError("semantic_rank_weight must be in [0, 1]")
        if not math.isfinite(self.config.rank_offset) or self.config.rank_offset < 0:
            raise ValueError("rank_offset must be finite and non-negative")
        if top_k is not None and top_k < 0:
            raise ValueError("top_k must be non-negative")
        aspects = set(query_aspects or [])
        if not aspects.issubset(ASPECT_KEYWORDS):
            raise ValueError("Unknown query aspect")

        # Shallow copies are enough: only top-level score fields are written.
        # Nested metadata is shared with the input and must not be modified here.
        records = [dict(chunk) for chunk in candidates]
        input_count = len(records)
        load_ms = rerank_ms = 0.0
        if records and top_k != 0 and self.config.strategy in {"rank_fusion", "learned"}:
            # Preserve retrieval's aspect-aware order. Only reorder its judged head;
            # deeper candidates keep their original positions and are not scored.
            head, tail = records[:self.config.rerank_window], records[self.config.rerank_window:]
            if scoring_query is None:
                import pandas as pd
                from retrieval.query_parser import QueryParser
                meta = pd.DataFrame([{"model": _metadata(r).get("model") or "",
                                      "brand": _metadata(r).get("brand") or ""} for r in records])
                parser = QueryParser(meta)
                model_name = parser.detect_model(query)
                scoring_query = parser.remove_model(query, model_name) if model_name else query
            scoring_query = scoring_query.strip() or query.strip()
            with self._lock:
                if self._model is None:
                    t = perf_counter()
                    self._model = self._load_model()
                    load_ms = (perf_counter()-t)*1000
                t = perf_counter()
                raw = [float(s) for s in self._model.predict(
                    [(scoring_query, r['text']) for r in head],
                    batch_size=self.config.batch_size, show_progress_bar=False,
                    convert_to_numpy=True)]
                if len(raw) != len(head) or not all(math.isfinite(s) for s in raw):
                    raise ValueError("Invalid model scores")
                ranks = {i: rank for rank,i in enumerate(
                    sorted(range(len(head)), key=lambda i: raw[i], reverse=True), 1)}
                w, k = self.config.semantic_rank_weight, self.config.rank_offset
                for i,r in enumerate(head):
                    semantic = w/(k+ranks[i])
                    retrieval = (1-w)/(k+i+1)
                    r.update(retrieval_rank=i+1, semantic_rank=ranks[i],
                             cross_encoder_raw_score=raw[i],
                             cross_encoder_score=0.25*(1+math.tanh(raw[i])),
                             cross_encoder_contribution=semantic,
                             retrieval_rank_contribution=retrieval,
                             aspect_bonus=0.0, rerank_score=semantic+retrieval,
                             rerank_strategy="rank_fusion", scoring_query=scoring_query)
                if self.config.strategy == "learned":
                    import json
                    from pathlib import Path
                    from retrieval.learned_rank import FEATURES, features
                    import numpy as np
                    model_path = (Path(self.config.learned_model_path) if self.config.learned_model_path
                                  else Path(__file__).parent / "models" / "review_ranker_v2.json")
                    model = json.loads(model_path.read_text(encoding="utf8"))
                    if model['version'] != 2 or model['features'] != [FEATURES[i] for i in model['columns']]:
                        raise ValueError("Incompatible learned ranker artifact")
                    if self.config.rerank_window != model['window']:
                        raise ValueError("Learned model requires its trained rerank window")
                    x = features(scoring_query, head, raw)
                    contributions = x[:, model['columns']] * np.asarray(model['coefficients'])
                    for i,r in enumerate(head):
                        r.update(rerank_score=float(contributions[i].sum()),
                                 rerank_strategy="learned",
                                 ranker_version=model['version'],
                                 feature_contributions=dict(zip(model['features'], contributions[i].tolist())))
                head.sort(key=lambda r: r['rerank_score'], reverse=True)
                for i,r in enumerate(tail, len(head)+1):
                    r.update(retrieval_rank=i, rerank_score=head[-1]['rerank_score']-(i-len(head)),
                             rerank_strategy="retrieval_tail", semantic_scored=False)
                records = (head + tail)[:top_k]
                for rank,r in enumerate(records,1):
                    r['final_rank'] = rank
                rerank_ms = (perf_counter()-t)*1000
            return RerankResult(records,input_count,
                                {"model_load":load_ms,"rerank":rerank_ms,
                                 "total":(perf_counter()-started)*1000},
                                {**asdict(self.config),"query_aspects":sorted(aspects)})
        records = []
        return RerankResult(
            candidates=records, input_count=input_count,
            timing_ms={"model_load": load_ms, "rerank": rerank_ms,
                       "total": (perf_counter() - started) * 1000},
            config={**asdict(self.config), "query_aspects": sorted(aspects)},
        )

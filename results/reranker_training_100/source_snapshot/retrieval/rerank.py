"""Re-rank an existing pool of review chunks with MiniLM.

Input : query, candidate dictionaries, and query aspects from query rewrite
Output: ranked candidates with model score, aspect bonus, and final rank

Default: score model-stripped query/raw-text pairs and apply the trained review
ranker to the first 20 candidates. Rank fusion and legacy scoring are optional.
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
    cross_encoder_weight: float = 0.6
    rrf_weight: float = 1.0         # max additive RRF reward
    aspect_weight: float = 1.2      # max reward for covering all query aspects
    aspect_threshold: float = -6.0  # raw model score at which half the reward is kept
    aspect_temperature: float = 1.0  # smaller = sharper transition; must be > 0


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


def _rrf_normalized(chunk: dict) -> float:
    """Only consume an explicitly normalized pure RRF, never generic `score`.

    Hybrid retrieval exports this value before its own aspect bonus. BM25,
    dense-only and uploaded candidates without the field receive no RRF bonus.
    """
    value = chunk.get("rrf_normalized")
    if value is None:
        return 0.0
    if isinstance(value, bool):
        raise ValueError("rrf_normalized must be a finite number in [0, 1]")
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError("rrf_normalized must be a finite number in [0, 1]") from None
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("rrf_normalized must be a finite number in [0, 1]")
    return value


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
        if self.config.strategy not in {"legacy", "rank_fusion", "learned"}:
            raise ValueError("Unknown reranking strategy")
        if self.config.rerank_window < 1 or not isinstance(self.config.rerank_window, int):
            raise ValueError("rerank_window must be a positive integer")
        if not 0 <= self.config.semantic_rank_weight <= 1:
            raise ValueError("semantic_rank_weight must be in [0, 1]")
        if not math.isfinite(self.config.rank_offset) or self.config.rank_offset < 0:
            raise ValueError("rank_offset must be finite and non-negative")
        if top_k is not None and top_k < 0:
            raise ValueError("top_k must be non-negative")
        if not math.isfinite(self.config.cross_encoder_weight) or self.config.cross_encoder_weight < 0:
            raise ValueError("cross_encoder_weight must be finite and non-negative")
        if not math.isfinite(self.config.rrf_weight) or self.config.rrf_weight < 0:
            raise ValueError("rrf_weight must be finite and non-negative")
        if not math.isfinite(self.config.aspect_weight) or self.config.aspect_weight < 0:
            raise ValueError("aspect_weight must be finite and non-negative")
        if not math.isfinite(self.config.aspect_threshold):
            raise ValueError("aspect_threshold must be finite")
        if not math.isfinite(self.config.aspect_temperature) or self.config.aspect_temperature <= 0:
            raise ValueError("aspect_temperature must be finite and positive")
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
                    if model['version'] not in (1,2) or model['features'] != [FEATURES[i] for i in model['columns']]:
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
        if records and top_k != 0:
            rrf_values = [_rrf_normalized(chunk) for chunk in records]
            # CrossEncoder reads the query and original chunk text, without metadata prefixes.
            pairs = [(query.strip(), chunk["text"]) for chunk in records]
            with self._lock:
                if self._model is None:
                    load_started = perf_counter()
                    self._model = self._load_model()
                    load_ms = (perf_counter() - load_started) * 1000
                scoring_started = perf_counter()
                scores = self._model.predict(
                    pairs, batch_size=self.config.batch_size,
                    show_progress_bar=False, convert_to_numpy=True,
                )
                for chunk, score, rrf in zip(records, scores, rrf_values, strict=True):
                    raw_score = float(score)
                    if not math.isfinite(raw_score):
                        raise ValueError("Model returned a non-finite score")
                    # Map the bounded semantic signal to [0, 0.5].
                    score = 0.25 * (1 + math.tanh(raw_score))
                    # Count distinct overlaps; repeated tags do not earn extra reward.
                    matches = len(aspects.intersection(_metadata(chunk).get("aspect_tags") or []))
                    coverage = matches / len(aspects) if aspects else 0.0
                    z = (raw_score - self.config.aspect_threshold) / self.config.aspect_temperature
                    # Keep half the matched-aspect reward even at low model relevance.
                    # sigmoid(z) = (1 + tanh(z / 2)) / 2, without exp overflow.
                    gate = (1 + math.tanh(z / 2)) / 2
                    fixed_bonus = 0.5 * self.config.aspect_weight * coverage
                    gated_bonus = fixed_bonus * gate
                    bonus = fixed_bonus + gated_bonus
                    rrf_bonus = self.config.rrf_weight * rrf
                    semantic = self.config.cross_encoder_weight * score
                    chunk.update(cross_encoder_raw_score=raw_score,
                                 cross_encoder_score=score, cross_encoder_contribution=semantic,
                                 aspect_match_count=matches,
                                 aspect_coverage=coverage, aspect_gate=gate,
                                 aspect_fixed_bonus=fixed_bonus, aspect_gated_bonus=gated_bonus,
                                 aspect_bonus=bonus,
                                 rrf_normalized=rrf, rrf_bonus=rrf_bonus,
                                 rerank_score=semantic + rrf_bonus + bonus)
                # Sort before taking Top-K; Python's stable sort preserves tied order.
                records.sort(key=lambda chunk: chunk["rerank_score"], reverse=True)
                records = records[:top_k]
                for rank, chunk in enumerate(records, 1):
                    chunk["final_rank"] = rank
                rerank_ms = (perf_counter() - scoring_started) * 1000
        else:
            records = []
        return RerankResult(
            candidates=records, input_count=input_count,
            timing_ms={"model_load": load_ms, "rerank": rerank_ms,
                       "total": (perf_counter() - started) * 1000},
            config={**asdict(self.config), "query_aspects": sorted(aspects)},
        )

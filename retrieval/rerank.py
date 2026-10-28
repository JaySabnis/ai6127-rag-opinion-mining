"""Re-rank an existing pool of review chunks with MiniLM.

Input : query, candidate dictionaries, and query aspects from query rewrite
Output: ranked candidates with model score, aspect bonus, and final rank

Steps: add product context -> score (query, chunk) pairs -> apply gated aspect
bonuses -> sort -> return Top-K. This module does not retrieve or filter products.
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
    model_name: str = DEFAULT_MODEL
    revision: str | None = None
    device: str = "cpu"
    batch_size: int = 16
    max_length: int = 512
    local_files_only: bool = False
    aspect_weight: float = 0.5       # maximum reward per distinct matched aspect
    aspect_threshold: float = -5.0  # model score at which half the reward is kept
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


def _model_text(chunk: dict) -> str:
    """Prefix known product metadata for scoring; leave the stored text unchanged."""
    metadata = _metadata(chunk)
    product = " ".join(str(metadata.get(key) or "").strip() for key in ("brand", "model")).strip()
    if product:
        return f"This is a review of {product}.\n\n{chunk['text']}"
    return chunk["text"]


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
               query_aspects: list[str] | None = None) -> list[dict]:
        """Return ranked candidates only; top_k=None keeps all, 0 skips scoring."""
        return self.rerank_with_details(
            query, candidates, top_k=top_k, query_aspects=query_aspects,
        ).candidates

    def rerank_with_details(self, query: str, candidates: list[dict], *,
                            top_k: int | None = None,
                            query_aspects: list[str] | None = None) -> RerankResult:
        """Also return load/inference timings and the aspect settings used."""
        started = perf_counter()
        if not query.strip():
            raise ValueError("query must not be empty")
        if top_k is not None and top_k < 0:
            raise ValueError("top_k must be non-negative")
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
        if records and top_k != 0:
            # CrossEncoder jointly reads the query and each product-prefixed chunk.
            pairs = [(query.strip(), _model_text(chunk)) for chunk in records]
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
                for chunk, score in zip(records, scores, strict=True):
                    score = float(score)
                    if not math.isfinite(score):
                        raise ValueError("Model returned a non-finite score")
                    # Count distinct overlaps; repeated tags do not earn extra reward.
                    matches = len(aspects.intersection(_metadata(chunk).get("aspect_tags") or []))
                    z = (score - self.config.aspect_threshold) / self.config.aspect_temperature
                    # Low model relevance reduces the aspect reward, not the model score.
                    # sigmoid(z) = (1 + tanh(z / 2)) / 2, without exp overflow.
                    gate = (1 + math.tanh(z / 2)) / 2
                    bonus = self.config.aspect_weight * matches * gate
                    chunk.update(cross_encoder_score=score, aspect_match_count=matches,
                                 aspect_gate=gate, aspect_bonus=bonus, rerank_score=score + bonus)
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

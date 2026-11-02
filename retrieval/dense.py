"""Dense retrieval: BGE sentence embeddings + exact FAISS inner-product search.

* Chunks are embedded once with ``BAAI/bge-small-en-v1.5`` (384-d) and
  L2-normalised, so inner product == cosine similarity.
* Queries get the BGE retrieval instruction prefix; documents do not.
* Encoding is written in shards (default 50k chunks each) so a long run can be
  interrupted and resumed, e.g. on a laptop or a Colab session.
* ``IndexFlatIP`` is exact (brute-force) search. At ~460k x 384 floats that is
  ~0.7 GB and tens of milliseconds per query, so no approximate index is needed.
* Metadata filters are applied *inside* the search with ``IDSelectorBatch``,
  so a model filter returns the true top-k for that model instead of
  post-filtering a global top-k.

If FAISS is not installed, an equivalent exact NumPy backend is used (same
results, used by the unit tests).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from retrieval.config import BGE_QUERY_INSTRUCTION, DEFAULT_EMBED_MODEL

try:  # pragma: no cover - availability depends on the machine
    import faiss  # type: ignore

    HAVE_FAISS = True
except ImportError:  # pragma: no cover
    faiss = None
    HAVE_FAISS = False


# --------------------------------------------------------------------- encoders
class Encoder(Protocol):
    name: str
    dim: int

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray: ...

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray: ...


def pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class SentenceTransformerEncoder:
    """Wrapper around sentence-transformers that returns normalised float32 vectors."""

    def __init__(
        self,
        model_name: str = DEFAULT_EMBED_MODEL,
        device: str | None = None,
        batch_size: int = 128,
        query_instruction: str = BGE_QUERY_INSTRUCTION,
    ):
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self.device = device or pick_device()
        self.model = SentenceTransformer(model_name, device=self.device)
        # sentence-transformers renamed this method; support both versions.
        get_dim = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        self.dim = get_dim()
        self.batch_size = batch_size
        self.query_instruction = query_instruction if "bge" in model_name.lower() else ""

    def _encode(self, texts: Sequence[str], show_progress: bool) -> np.ndarray:
        emb = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        )
        return np.asarray(emb, dtype=np.float32)

    def encode_documents(self, texts: Sequence[str], show_progress: bool = True) -> np.ndarray:
        return self._encode(texts, show_progress)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode([self.query_instruction + t for t in texts], show_progress=False)


# --------------------------------------------------------------- corpus encoding
def encode_corpus(
    texts: Sequence[str],
    encoder: Encoder,
    out_dir: str | Path,
    shard_size: int = 50_000,
) -> Path:
    """Embed all texts into ``out_dir/shards/part_XXXXX.npy`` (float16). Resumable."""
    out = Path(out_dir)
    shard_dir = out / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    n = len(texts)
    n_shards = (n + shard_size - 1) // shard_size

    info_path = out / "info.json"
    info = {"model": encoder.name, "dim": int(encoder.dim), "n": n, "shard_size": shard_size}
    if info_path.exists():
        old = json.loads(info_path.read_text())
        if {k: old.get(k) for k in info} != info:
            raise RuntimeError(f"{info_path} was created with different settings {old}; use a fresh --index-dir")
    info_path.write_text(json.dumps(info, indent=2))

    t0 = time.time()
    for s in range(n_shards):
        path = shard_dir / f"part_{s:05d}.npy"
        if path.exists():
            print(f"  shard {s + 1}/{n_shards} exists, skipping")
            continue
        lo, hi = s * shard_size, min(n, (s + 1) * shard_size)
        emb = encoder.encode_documents(texts[lo:hi])
        tmp = path.with_name(path.stem + ".tmp.npy")
        np.save(tmp, emb.astype(np.float16))
        tmp.replace(path)
        print(f"  shard {s + 1}/{n_shards} done ({hi:,}/{n:,} chunks, {time.time() - t0:.0f}s)")
    return shard_dir


def load_embeddings(out_dir: str | Path) -> np.ndarray:
    out = Path(out_dir)
    info = json.loads((out / "info.json").read_text())
    parts = sorted((out / "shards").glob("part_*.npy"))
    expected = (info["n"] + info["shard_size"] - 1) // info["shard_size"]
    if len(parts) != expected:
        raise RuntimeError(f"found {len(parts)} of {expected} shards; finish encoding first")
    emb = np.vstack([np.load(p).astype(np.float32) for p in parts])
    if emb.shape != (info["n"], info["dim"]):
        raise RuntimeError(f"embedding matrix has shape {emb.shape}, expected {(info['n'], info['dim'])}")
    # Re-normalise after the float16 round trip.
    emb /= np.linalg.norm(emb, axis=1, keepdims=True).clip(min=1e-12)
    return emb


# ------------------------------------------------------------------- backends
class _NumpyFlatIP:
    """Exact inner-product search in NumPy. Same results as faiss.IndexFlatIP."""

    def __init__(self, vectors: np.ndarray):
        self.x = np.ascontiguousarray(vectors, dtype=np.float32)
        self.ntotal = self.x.shape[0]

    def search(self, q: np.ndarray, k: int, rows: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        if rows is None:
            sims = self.x @ q
            cand = np.arange(self.ntotal)
        else:
            sims = self.x[rows] @ q
            cand = rows
        k = min(k, sims.shape[0])
        if k == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top], kind="stable")]
        return cand[top].astype(np.int64), sims[top].astype(np.float32)


class _FaissFlatIP:
    def __init__(self, index):
        self.index = index
        self.ntotal = index.ntotal

    def search(self, q: np.ndarray, k: int, rows: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        q2 = np.ascontiguousarray(q.reshape(1, -1), dtype=np.float32)
        if rows is None:
            D, I = self.index.search(q2, min(k, self.ntotal))
        else:
            ids = np.ascontiguousarray(rows, dtype=np.int64)  # must stay alive during search
            if ids.size == 0:
                return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
            try:
                sel = faiss.IDSelectorBatch(ids)
            except TypeError:  # older SWIG signature
                sel = faiss.IDSelectorBatch(ids.size, faiss.swig_ptr(ids))
            D, I = self.index.search(q2, min(k, ids.size), params=faiss.SearchParameters(sel=sel))
        keep = I[0] >= 0
        return I[0][keep].astype(np.int64), D[0][keep].astype(np.float32)


# ---------------------------------------------------------------------- index
class DenseIndex:
    def __init__(self, backend, encoder: Encoder | None, info: dict):
        self.backend = backend
        self.encoder = encoder
        self.info = info

    @property
    def backend_name(self) -> str:
        return "faiss.IndexFlatIP" if isinstance(self.backend, _FaissFlatIP) else "numpy-flat-ip"

    @staticmethod
    def build(out_dir: str | Path, use_faiss: bool | None = None) -> Path:
        """Assemble the shards into a search index file inside ``out_dir``."""
        out = Path(out_dir)
        emb = load_embeddings(out)
        use_faiss = HAVE_FAISS if use_faiss is None else use_faiss
        if use_faiss:
            index = faiss.IndexFlatIP(emb.shape[1])
            index.add(emb)
            path = out / "faiss.index"
            faiss.write_index(index, str(path))
        else:
            path = out / "vectors.npy"
            np.save(path, emb)
        print(f"  dense index written: {path} ({emb.shape[0]:,} x {emb.shape[1]})")
        return path

    @classmethod
    def load(cls, out_dir: str | Path, encoder: Encoder | None = None) -> "DenseIndex":
        out = Path(out_dir)
        info = json.loads((out / "info.json").read_text())
        if HAVE_FAISS and (out / "faiss.index").exists():
            backend = _FaissFlatIP(faiss.read_index(str(out / "faiss.index")))
        elif (out / "vectors.npy").exists():
            backend = _NumpyFlatIP(np.load(out / "vectors.npy"))
        elif (out / "faiss.index").exists():
            raise RuntimeError("faiss.index exists but faiss is not installed (pip install faiss-cpu)")
        else:
            raise FileNotFoundError(f"no dense index in {out}; run the 'faiss' build stage")
        if encoder is not None and encoder.name != info["model"]:
            raise RuntimeError(f"index was built with {info['model']} but the query encoder is {encoder.name}")
        return cls(backend, encoder, info)

    def encode_query(self, query: str) -> np.ndarray:
        if self.encoder is None:
            raise RuntimeError("DenseIndex was loaded without a query encoder")
        return self.encoder.encode_queries([query])[0]

    def search(
        self,
        query: str | None = None,
        k: int = 100,
        rows: np.ndarray | None = None,
        query_vector: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Top-k (row_ids, cosine similarities), best first."""
        q = query_vector if query_vector is not None else self.encode_query(query)
        return self.backend.search(np.asarray(q, dtype=np.float32), k, rows)

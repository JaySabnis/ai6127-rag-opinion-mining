# Retrieval (Group B)

Group B turns the knowledge-base chunks into ranked evidence for generation:

```text
user question
    │
    ▼
query_rewrite.py   DeepSeek: English retrieval query + products + aspects   (optional)
    │
    ▼
retriever.py       BM25 + dense (BGE/FAISS) + RRF  ->  candidate pool, e.g. top 50
    │
    ▼
rerank.py          MiniLM cross-encoder + gated aspect bonus  ->  top 5-10
    │
    ▼
app/streamlit_app.py (UI)  /  generation
```

Part A covers hybrid retrieval (`retriever.py` and the modules it uses);
Part B covers query rewrite and reranking (`query_rewrite.py`, `rerank.py`).

**Agreed design: the LLM identifies, the corpus catalogue checks, rules fall back.**
The DeepSeek rewrite decides which products and aspects the question is about;
`resolve_model()` maps each product name onto a model that exists in the corpus
(so a filter never silently matches nothing); if the rewrite fails, the
retriever's own parser detects the model from the original question.

```python
from retrieval import Retriever
from retrieval.dense import pick_device
from retrieval.query_rewrite import rewrite_query
from retrieval.rerank import CrossEncoderReranker, RerankerConfig

r = Retriever.load("indexes")            # loads meta + BM25 + FAISS + BGE query encoder
# NOT device="cpu" on macOS: CPU torch next to FAISS segfaulted in testing (see 4.)
reranker = CrossEncoderReranker(RerankerConfig(device=pick_device()))   # cuda > mps > cpu

def product_filter(p):                   # LLM product -> corpus model, else brand, else none
    model = r.parser.resolve_model(p["brand"], p["model"])
    if model:
        return {"model": model}
    brand = r.parser.resolve_brand(p["brand"])
    return {"brand": brand} if brand else None

try:
    rw = rewrite_query(question)         # {query, product: [{brand, model}], aspects}
except (RuntimeError, ValueError):
    rw = None                            # no key / network / bad JSON -> fallback

if rw:
    filters = [product_filter(p) for p in rw["product"]] or [None]
    results = [r.retrieve(rw["query"], k=50, auto_filter=False, aspects=rw["aspects"], filters=f)
               for f in filters]         # one retrieval per product (comparisons)
else:
    results = [r.retrieve(question, k=50)]   # parser detects the model itself

for res in results:
    top = reranker.rerank(res.query, [h.to_dict() for h in res.hits],
                          query_aspects=res.parsed.aspects, top_k=10)
```

`Hit.to_dict()` already contains the fields the reranker reads (`chunk_id`,
`doc_id`, `text`, `brand`, `model`, `aspect_tags`).

Run all tests (retrieval, reranker, UI) with:

```bash
uv run python -m unittest discover -s tests -v
```

Check the whole pipeline on the real indexes, and compare ways of matching
LLM product names to the corpus (both print latency per stage):

```bash
uv run python -m scripts.test_pipeline                  # rewrite -> resolve -> retrieve -> rerank, 7 question types
uv run python -m scripts.eval_model_resolution          # parser vs LLM (+ shortlist / full catalogue), 27 hard cases
```

---

# Part A: Hybrid retrieval (BM25 + Dense + RRF)


## 1. Setup and building the indexes

```bash
uv sync                                              # installs the new dependencies
uv run python -m scripts.smoke_test_retrieval        # 1-minute check with real FAISS + BGE
```

Put the knowledge-base output at `data/corpus.jsonl` (`data/` and `indexes/` are git-ignored).

**Try a small index first** (first 5,000 chunks, a few minutes):

```bash
uv run python -m scripts.build_retrieval_indexes --stage all --max-records 5000 --index-dir indexes_small
uv run python -m scripts.search_cli "How is the camera on the Galaxy S10?" --index-dir indexes_small --mode all
```

**Full index** (459,418 chunks):

```bash
uv run python -m scripts.build_retrieval_indexes --stage meta
uv run python -m scripts.build_retrieval_indexes --stage bm25
uv run python -m scripts.build_retrieval_indexes --stage encode --batch-size 256   # longest step
uv run python -m scripts.build_retrieval_indexes --stage faiss
```

* `encode` picks CUDA → Apple MPS → CPU automatically (`--device` to override).
  It writes 50k-chunk shards, so if it is interrupted just run it again and it
  resumes.
* **Colab option:** clone the repo in Colab, upload `indexes/meta.pkl`, run the
  `encode` and `faiss` stages on the GPU, then copy `indexes/dense/` back.
  The encoder must be the same model on both sides (it is checked at load time).

Output layout:

```text
indexes/
├── meta.pkl              row-aligned metadata + text (row i == BM25 doc i == FAISS id i)
├── bm25/                 weights.npz, vocab.json, params.json
└── dense/                info.json, shards/part_*.npy, faiss.index
```

---

## 2. The interface (for reranker / UI / generation)

```python
res = r.retrieve(
    query,
    mode="hybrid",          # "bm25" | "dense" | "hybrid"
    k=10,                   # hits returned
    candidate_k=100,        # depth taken from each ranker before fusion
    filters=None,           # {"model", "brand", "min_rating", "max_rating", "exclude_renewed"};
                            # "model" may be any corpus spelling, it covers all spellings of that phone
    auto_filter=True,       # use the model detected in the query as a filter
    aspect_weight=0.5,      # hybrid aspect boost, 0 = off
    max_per_doc=1,          # at most N chunks per original review, None = keep all
    strip_model=True,       # drop the model name from the searched text once filtered to it
    aspects=None,           # aspects to boost, e.g. rewrite["aspects"]; None = detect from the query
)
```

`res` is a `RetrievalResult`:

| field | meaning |
|---|---|
| `res.hits` | list of `Hit`, best first |
| `res.search_text` | text actually sent to BM25 / the encoder (model name removed when filtered) |
| `res.parsed` | detected `model`, `brand`, `aspects` of the query |
| `res.filters` | filters actually applied (including auto-detected model, canonical spelling) |
| `res.model_variants` | corpus spellings covered by the model filter, e.g. `["Motorola Razr", "Razr"]` |
| `res.n_candidates` | rows allowed by the filters (`None` = whole corpus) |
| `res.timings_ms` | `parse_filter`, `bm25`, `dense_encode`, `dense_search`, `fuse_format`, `total` |
| `res.to_dict()` | JSON-serialisable version |

Each `Hit` carries the chunk and its provenance:

| field | meaning |
|---|---|
| `rank`, `score` | position and score in the requested mode (RRF score for hybrid) |
| `chunk_id`, `doc_id`, `row_id` | ids (unchanged from the knowledge base) |
| `text` | chunk text — feed this to the reranker / LLM |
| `brand`, `model`, `rating`, `timestamp`, `aspect_tags`, `helpful_votes`, `verified`, `is_renewed`, `source_url` | metadata |
| `bm25_score`, `bm25_rank`, `dense_score`, `dense_rank` | per-ranker evidence (`None` if that ranker did not return it) |
| `aspect_boosted` | whether the hybrid aspect boost applied |

**Reranker integration:** retrieve a deeper pool (e.g. `k=50`) and pass
`[h.to_dict() for h in res.hits]` to `CrossEncoderReranker.rerank()`
(see the example at the top and Part B).

Note: `rating` is the overall review rating, **not** the aspect sentiment
(see the knowledge-base handoff document).

---

## 3. How it works (for the report)

**BM25 (sparse).** Implemented directly (not a library) on a precomputed
sparse matrix. Lucene/Robertson form with `k1 = 1.2`, `b = 0.75`:

```text
idf(t)  = ln(1 + (N − df + 0.5) / (df + 0.5))
w(d,t)  = idf(t) · tf·(k1+1) / (tf + k1·(1 − b + b·|d|/avgdl))
score   = Σ_t qtf(t) · w(d,t)
```

`w(d,t)` is query-independent, so it is computed once for every posting and
stored as a CSC matrix; a query reads only its own columns. Tokenisation:
lowercase alphanumerics, English stopwords removed, Snowball stemming (NLTK).
Model tokens such as `s10`, `5g`, `11` are kept. Verified against a brute-force
implementation of the same formula in `tests/test_retrieval.py`.

**Dense.** `BAAI/bge-small-en-v1.5` (384-d), vectors L2-normalised so inner
product = cosine similarity. Queries get the BGE instruction prefix
("Represent this sentence for searching relevant passages: "), passages do not.
FAISS `IndexFlatIP` = exact search; at this corpus size no approximate index is
needed.

**Filters inside the search.** A model filter is applied inside both rankers
(BM25 subset scoring, FAISS `IDSelectorBatch`), so "Galaxy S10 camera" returns
the true top-k among Galaxy S10 chunks instead of filtering a global top-k
(which could leave few or no hits).

**Remove the model name once filtered.** After filtering to "Galaxy S10",
every candidate is a Galaxy S10 review, so the words "Galaxy S10" in the query
no longer separate relevant from irrelevant chunks. Left in, they made BM25
rank chunks that merely repeat the name ("Not as advertised for Galaxy S10")
above chunks about the camera, and pulled the dense query towards generic
"Galaxy S10" reviews. So "How is the camera on the Galaxy S10?" is searched
as "How is the camera on the ?" within the Galaxy S10 subset
(`strip_model=False` turns this off for comparison).

**Hybrid = Reciprocal Rank Fusion** (Cormack et al., 2009):
`RRF(d) = Σ_r 1 / (60 + rank_r(d))`. Ranks rather than scores are fused, so
BM25 scores (unbounded) and cosine similarities need no calibration.

**Aspect boost (soft).** The query's aspects are detected with the
knowledge-base group's own `detect_aspects`. Chunks whose `aspect_tags` match
get `+ w / (60 + 1)` (as if an extra "aspect ranker" put them first). It is a
boost, not a filter, because the tags are keyword-based and noisy.

**Query parsing.** Model names are matched against the models that exist in
the corpus, on token boundaries, longest match first (`iPhone 11 Pro` beats
`iPhone 11`; `iPhone11`, `S10` and `S10+` = `Galaxy S10 Plus` are recognised).
Only distinctive name patterns are used, so ordinary words and specs are not
mistaken for models (see Known limitations).

**Names from the LLM.** `r.parser.resolve_model(brand, model)` maps a product
name produced by the query rewrite onto a corpus model: exact name ignoring
case and spacing ("galaxy s10e" → "Galaxy S10e"), else the same pattern
matching ("Galaxy S10+" → "Galaxy S10 Plus"), else `None` (do not filter).
`resolve_brand()` does the same for brand-only questions.

**One phone, several spellings.** The corpus spells some phones in several
ways ("Galaxy Note 9" / "Galaxy Note9", "Motorola Razr" / "Razr",
"Galaxy S20 FE" / "S20 FE", "G8X Thinq™" / "G8X Thinq"). Within one brand,
names that are equal after ignoring case, spacing, punctuation, a leading
brand name and Samsung's leading "Galaxy" are treated as one phone: 84 groups,
all checked by hand. The most frequent spelling is the canonical name returned
by `detect_model()` / `resolve_model()`, and a model filter covers every
spelling (`res.model_variants`). Effect on the evidence available, e.g.
Motorola Razr 3,487 → 5,748 chunks, Galaxy Note 8 808 → 1,451. Different
phones stay apart ("Moto G Power" vs "Moto G Power (2020)"; Samsung's
"Galaxy Note 10" vs Xiaomi's "Note 10").

**LLM vs rules for product names** (`scripts/eval_model_resolution.py`, 27
hard questions: Chinese brand names, "Note 10+", "iphone11promax", one-word
names, comparisons). Parser alone: 20/27 correct. DeepSeek + `resolve_model()`:
27/27, but only 20/27 of DeepSeek's own model strings were corpus spellings
("Galaxy Note 10+", "Moto G Power 2020", "Note 8 Pro", ...), so without the
check 6 real phones would have matched 0 rows. Putting the catalogue in the
prompt made DeepSeek copy corpus spellings (26/27) but did not change final
accuracy (27/27), while the full catalogue cost ~38.6k prompt tokens instead
of ~430 and ~375 ms more per call; a 20-name shortlist cost ~200 tokens.
Design kept: LLM + `resolve_model()` check, parser as fallback.

**One chunk per review by default.** Overlapping chunks of the same review are
not independent evidence, so `max_per_doc=1` keeps the best chunk per `doc_id`.

---

## 4. Measured performance

**Real index** (459,418 chunks), Apple-silicon MacBook, BGE and reranker on
MPS, `scripts/test_pipeline.py` (7 questions, pool of 50 per product, top 5):

| stage | median | max |
|---|---|---|
| load indexes + BGE (once, at start-up) | 5–7 s | |
| DeepSeek query rewrite | 1,403 ms | 1,770 ms |
| hybrid retrieval (incl. query encoding) | 28–71 ms | 90 ms |
| MiniLM rerank, 50 candidates | 100–114 ms | 235 ms |
| total with rewrite | 1,594 ms | 1,944 ms |
| total without rewrite | 155 ms | 278 ms |

The LLM call is ~88 % of the latency; retrieval + reranking stay under 0.3 s.
The first BGE encode after start-up takes ~300 ms (GPU warm-up), so
`Retriever.load()` does one warm-up encode.

**Platform note (macOS).** Running the reranker on CPU in the same process as
FAISS crashed with a segmentation fault (exit 139, in torch `layer_norm`);
with `device="mps"` the same run passed 7/7. Use `pick_device()` for the
reranker; on a CPU-only machine run `scripts/test_pipeline.py` first.

Micro-benchmarks on synthetic data of the same size (development sandbox):

| step | latency / size |
|---|---|
| BM25 query, whole corpus / 20k-row subset | < 1 ms |
| exact dense search, whole corpus / 20k-row subset (NumPy; FAISS similar or faster) | ~43 ms / ~5 ms |
| BM25 index on disk | ~170 MB |
| dense vectors | ~350 MB (fp16 shards), ~700 MB in RAM (fp32) |

---

## 5. Known limitations

* `aspect_tags` come from keyword matching and are noisy: "fast delivery" is
  tagged `performance` (keyword "fast"); reviews mentioning "Moto G **Power**"
  get `battery`. (The query side strips the model name before detecting
  aspects; the chunk side is unchanged, as the knowledge-base contract says
  not to modify the corpus.)
* The model catalogue (4,191 names) contains noisy names such as "Pre",
  "Rugged", "6.1", "Pcs" and even accessories ("7700mah Battery"). To keep
  auto-filtering safe, a model is only detected from a query when the matched
  words are distinctive: a word plus a number ("Galaxy S10", "G2"), two words
  ("iPhone SE"), or brand + name ("Nokia 3310", "Palm Pre"). Specs such as
  "5G", "128GB", "5000mAh" never count as a model. Consequence for the rule
  parser: one-word names need the brand ("Motorola Razr", not just "Razr"),
  and Chinese brand names are not understood. The LLM rewrite handles these
  (see section 3); the parser is the fallback when the rewrite fails.
* "Galaxy S10" filters to that phone (all its spellings), not the S10e /
  S10 Plus / S10 Lite, which are separate phones in the corpus.
* For comparison questions each product is retrieved separately, but the
  searched text still names the other product, which favours reviews that
  compare the two.
* Chunks with no `model` (empty in 2,534 chunks) are only reachable without a model filter.

---

## 6. Files

| file | role |
|---|---|
| `retrieval/corpus.py` | read corpus, row-aligned metadata, metadata filters |
| `retrieval/text.py` | tokenizer (stopwords, stemming) |
| `retrieval/bm25.py` | BM25 index |
| `retrieval/dense.py` | BGE encoder, resumable sharded encoding, FAISS index |
| `retrieval/query_parser.py` | model / aspect detection in queries |
| `retrieval/hybrid.py` | RRF fusion, aspect boost |
| `retrieval/retriever.py` | `Retriever.retrieve()` |
| `scripts/build_retrieval_indexes.py` | build all indexes |
| `scripts/search_cli.py` | query from the terminal, compare modes, JSON output |
| `scripts/smoke_test_retrieval.py` | quick check with real FAISS + BGE on the test fixture |
| `scripts/test_pipeline.py` | end-to-end check on the real indexes: rewrite → resolve → retrieve → rerank, latency per stage |
| `scripts/eval_model_resolution.py` | experiment: parser vs LLM (+ shortlist / full catalogue) for product names |
| `tests/test_retrieval.py` | unit tests (no model download) |
| `tests/fixtures/make_fixture.py`, `mini_corpus.jsonl` | 19 hand-written synthetic test chunks (incl. one spelling variant) |

---

# Part B: Query rewrite and reranker

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

"""Step 2: clean and de-duplicate the handset reviews.

Input : data/interim/phone_reviews_raw.jsonl  (from 01_extract_phones.py)
Output: data/processed/clean_reviews.jsonl     one cleaned review per line
        reports/02_cleaning_report.json         rows removed at each step (the table for the report)
        reports/02_removed_samples.jsonl        removed examples per reason, for manual checking
        reports/02_kept_short_samples.jsonl     kept short reviews, for manual checking

Principle: when in doubt, keep. A review is only removed when it carries no opinion about the phone,
is not English, is purely about the seller/purchase, or is a copy of the same person's review of the same phone.

Steps, in order (each review is dropped at the first rule it fails):
  1. normalize text (HTML, URLs, Unicode, whitespace); blank auto-titles like "Five Stars"
  2. short reviews (< 5 words): keep if title+text carries an opinion or a phone aspect
     (NLTK VADER, Hu & Liu opinion lexicon, aspect words, emoji); drop empty/filler and non-English ones
  3. drop non-English reviews (stop-word heuristic, langdetect for the uncertain ones)
  4. drop seller-only reviews: every clause is about the seller/shipping/refund/order, none about the phone
  5. drop duplicates ONLY when the same account reviewed the same product (brand + model) with an identical or
     near-identical text (Jaccard >= 0.8 on 5-word shingles). One account = one person, on any device.
     Reviews from different accounts are never removed as duplicates, however similar; identical texts that are
     kept share a `dup_group` id so retrieval can show one copy
  6. add `text_norm`, a microtext-normalized copy of the text (see microtext.py); `text` stays original
"""
import hashlib
import json
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from difflib import get_close_matches
from functools import lru_cache

from langdetect import DetectorFactory, LangDetectException, detect_langs
from nltk.corpus import opinion_lexicon
from nltk.corpus import words as nltk_words
from nltk.sentiment import SentimentIntensityAnalyzer

from common import INTERIM, PROCESSED, REPORTS, clean_text, clean_title, dedup_key, words
from microtext import normalize

MIN_WORDS = 5
NEAR_DUP_THRESHOLD = 0.8
SHINGLE = 5
SAMPLES_PER_REASON = 40

DetectorFactory.seed = 0
FOREIGN_LANGS = {"es", "pt", "it", "fr", "de", "ca", "nl", "pl", "ro", "ru", "tr", "ar", "ja", "ko", "zh-cn", "zh-tw"}
EN_STOP = set("the a an and or but i it is was this that to of in for on with my me you not be have has had "
              "are very so just it's its they phone great good".split())

# ---- lexical resources (NLTK)
VADER = SentimentIntensityAnalyzer()
OPINION = set(opinion_lexicon.positive()) | set(opinion_lexicon.negative())
OPINION_LIST = sorted(OPINION)
EN_VOCAB = {w.lower() for w in nltk_words.words()} | OPINION | set(
    "phone phones ok okay wifi app apps lol sim gb mb nfc android iphone samsung galaxy pixel moto motorola lg "
    "blu nokia sony huawei unlocked smartphone cellphone text texting charger battery selfie email emails "
    "bluetooth gps".split())
ASPECT_RE = re.compile(
    r"\b(camera|photos?|pictures?|pics?|videos?|selfies?|battery|charg\w*|screen|display|touch\w*|sound|audio|"
    r"speakers?|volume|mic|calls?|calling|signal|reception|wi-?fi|bluetooth|gps|apps?|android|ios|software|"
    r"updates?|fast|slow|lags?|laggy|freez\w*|crash\w*|glitch\w*|speed|performance|processor|storage|memory|ram|"
    r"size|small|big|large|light|heavy|design|buttons?|fingerprint|face id|keyboard|texts?|texting|work|works|"
    r"worked|working|function\w*|turn on|power|boot\w*|restart\w*|reset|durable|quality|price|value|money|"
    r"phone|device|cell)\b", re.I)
POSITIVE_PHRASES_RE = re.compile(r"\b(a\+|as advertised|as described|as expected|as promised|as pictured|"
                                 r"exactly what|just what|no complaints|no issues|no problems|meets\b|met the|"
                                 r"thumbs (up|down)|expectations?|like new|as new|still using|pos\b|all ?right)", re.I)
# praise/criticism aimed at the phone itself, recognised even inside a sentence that also mentions the seller
PHONE_PRAISE_RE = re.compile(r"\b(\w+)\s+(phone|device|cellphone|smartphone|iphone|galaxy|pixel)\b|"
                             r"\b(love[sd]?|like[sd]?|hate[sd]?|enjoy\w*)\s+(it|this|the|my)\b", re.I)
SENTIMENT_EMOJI = set("👍👌😊😀😃😄😁😍🥰❤💯🔥⭐🙌👏😎🤩💕💖😘☺🙂✅"   # positive
                      "👎😡😠😞😢😭💩🤬😤😒🙁☹😩😫🤮❌")             # negative
ELONGATED_RE = re.compile(r"(.)\1{2,}")

# ---- seller / purchase-only detection (clause level)
SELLER_RE = re.compile(
    r"\b(sellers?|vendors?|merchant|store|ship\w*|deliver\w*|arriv\w*|package\w*|packag\w*|packed|boxe?s?|"
    r"refund\w*|return\w*|exchang\w*|customer service|customer support|amazon|orders?|ordered|ordering|"
    r"tracking|transaction|business|dealer|wrong (item|one|phone|model|color|colour|size)|never (got|received)|"
    r"not received|mistake|came (early|late|quickly|quick|fast|on time|in time)|on time|timely|"
    r"thanks?|thank you|a\+{2,}|\+{3,})\b", re.I)
# concrete phone features: inside a seller clause these still mean the clause is about the phone
# ("Amazon Renewed phone works great"); generic words like "fast"/"phone" do not ("fast shipping")
PHONE_FEATURE_RE = re.compile(
    r"\b(camera|photos?|pictures?|videos?|selfies?|battery|charg\w*|screen|display|touch\w*|sound|audio|"
    r"speakers?|volume|mic|calls?|signal|reception|wi-?fi|bluetooth|gps|apps?|android|ios|software|updates?|"
    r"lags?|laggy|slow\w*|freez\w*|crash\w*|glitch\w*|performance|processor|storage|memory|ram|buttons?|"
    r"fingerprint|applications?|operat\w*|"
    r"keyboard|works?|worked|working|function\w*|turn on|boot\w*|restart\w*|reset|unlock\w*|locked|carrier|"
    r"gsm|cdma|lte|[345]g|nfc|sim)\b", re.I)
CLAUSE_SPLIT_RE = re.compile(r"[.!?;,\n]+|\s(?:but|and|though|although|however|plus|also|except)\s", re.I)
# words that only thank or praise the transaction, ignored outside seller clauses ("Fast delivery. Thanks!")
PLEASANTRIES = {"thanks", "thank", "thx", "appreciate", "appreciated", "again"}
# frequent one-word Spanish/Portuguese reviews ("Excelente", "muy bueno") that would otherwise pass as English
SPANISH = {"excelente", "bueno", "buena", "buen", "muy", "recomendado", "recomiendo", "telefono", "teléfono",
           "celular", "producto", "perfecto", "gracias", "todo", "bien", "llego", "llegó", "equipo", "calidad",
           "precio", "malo", "mala", "funciona", "recibido", "enviaron", "usado", "otimo", "ótimo"}


def norm_tokens(text):
    return [ELONGATED_RE.sub(r"\1", w.lower()) for w in words(text)]


@lru_cache(maxsize=None)
def spelling_fix(w):
    """Closest opinion word for a misspelling ("exelent" -> "excellent"), or None."""
    if len(w) < 4 or w in SPANISH:
        return None
    m = get_close_matches(w, OPINION_LIST, n=1, cutoff=0.85)
    return m[0] if m else None


def in_vocab(w):
    if w in EN_VOCAB or w.isdigit():
        return True
    for suf in ("s", "es", "ed", "ing", "ly", "'s"):
        if w.endswith(suf) and w[: -len(suf)] in EN_VOCAB:
            return True
    return spelling_fix(w) is not None


def praises_phone(clause):
    """'great phone', 'love it' - an opinion word directly aimed at the phone."""
    for m in PHONE_PRAISE_RE.finditer(clause):
        if m.group(3) or (m.group(1) and m.group(1).lower() in OPINION):
            return True
    return False


def has_opinion_or_feature(text):
    toks = norm_tokens(text)
    # misspellings: also try tokens that are rare English words ("geat" is in the word list but means "great")
    toks = toks + [f for f in (spelling_fix(t) for t in toks if t not in OPINION) if f]
    return (any(t in OPINION for t in toks) or abs(VADER.polarity_scores(" ".join(toks))["compound"]) >= 0.05
            or bool(PHONE_FEATURE_RE.search(text)) or bool(POSITIVE_PHRASES_RE.search(text))
            or any(c in SENTIMENT_EMOJI for c in text))


def short_review_verdict(title, text):
    """'keep', 'non_english' or 'no_content' for a review with < MIN_WORDS words of body text."""
    full = f"{title} {text}".strip()
    toks = norm_tokens(full)
    if not toks:  # emoji / emoticons only: VADER scores ":)" and ":(" directly
        has_sentiment = any(c in SENTIMENT_EMOJI for c in full) or abs(VADER.polarity_scores(full)["compound"]) >= 0.05
        return "keep" if has_sentiment else "no_content"
    if sum(t in SPANISH for t in toks) / len(toks) >= 0.5 or sum(in_vocab(t) for t in toks) / len(toks) < 0.5:
        return "non_english"
    return "keep" if has_opinion_or_feature(full) else "no_content"


def is_seller_only(title, text):
    """True when the review talks about the seller/purchase and no clause says anything about the phone.

    Clauses that mention the seller only count as phone content if they name a concrete phone feature;
    other clauses count if they have an aspect word, an opinion word or a sentiment emoji."""
    full = f"{title}. {text}" if title else text
    if not SELLER_RE.search(full):
        return False
    for clause in CLAUSE_SPLIT_RE.split(full):
        if not clause or not clause.strip():
            continue
        if SELLER_RE.search(clause):
            if PHONE_FEATURE_RE.search(clause) or praises_phone(clause):
                return False
            continue
        toks = [t for t in norm_tokens(clause) if t not in PLEASANTRIES]
        if ASPECT_RE.search(clause) or any(t in OPINION for t in toks) or any(c in SENTIMENT_EMOJI for c in clause):
            return False
    return True


def is_english(text):
    toks = [w.lower() for w in words(text)]
    if not toks:
        return False
    ascii_ratio = sum(c.isascii() for c in text) / len(text)
    stop_ratio = sum(w in EN_STOP for w in toks) / len(toks)
    if ascii_ratio > 0.95 and stop_ratio >= 0.2:
        return True
    report["langdetect_calls"] += 1
    try:
        langs = {l.lang: l.prob for l in detect_langs(text)}
    except LangDetectException:
        return True
    top = max(langs, key=langs.get)
    # langdetect often labels short English text as Afrikaans/Tagalog/...: only trust it for languages
    # that actually occur in this corpus (Spanish is ~95% of the non-English reviews)
    return top == "en" or langs.get("en", 0) >= 0.1 or top not in FOREIGN_LANGS


t0 = time.time()
report = Counter()
samples = defaultdict(list)


def drop(reason, row, **extra):
    report[f"removed_{reason}"] += 1
    if len(samples[reason]) < SAMPLES_PER_REASON:
        samples[reason].append({"reason": reason, "rating": row["rating"], "title": row["title"],
                                "text": row["text"][:300], **extra})


def day(ts):
    return datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


# ---- load, sort by time so the earliest copy of a duplicate is the one kept
rows = [json.loads(line) for line in open(INTERIM / "phone_reviews_raw.jsonl", encoding="utf8")]
rows.sort(key=lambda r: r["timestamp"])
report["input_reviews"] = len(rows)

kept = []
kept_short_samples = []
for r in rows:
    r["title"] = clean_title(r.get("title"))
    r["text"] = clean_text(r.get("text"))
    r["date"] = day(r["timestamp"])
    n = len(words(r["text"]))
    r["is_short"] = n < MIN_WORDS
    if r["is_short"]:
        verdict = short_review_verdict(r["title"], r["text"])
        if verdict != "keep":
            drop(f"short_{verdict}", r)
            continue
        report["kept_short"] += 1
        if len(kept_short_samples) < 200:
            kept_short_samples.append({"rating": r["rating"], "title": r["title"], "text": r["text"]})
    elif not is_english(r["text"]):
        drop("non_english", r)
        continue
    if is_seller_only(r["title"], r["text"]):
        drop("seller_only", r)
        continue
    r["dup_key"] = dedup_key(f"{r['title']} {r['text']}" if r["is_short"] else r["text"])
    r["n_words"] = n
    kept.append(r)
print(f"rule filters done: {len(kept):,} left ({time.time() - t0:.0f}s)", flush=True)


# ---- duplicates: only the same account reviewing the same product with (near-)identical text.
# One account = one person on any device. "Same product" = same brand + model, because one phone is sold under
# several listings (colours, storage, carriers); the listing id is used when the model is unknown.
# Different accounts are never treated as duplicates, however similar their text.
def shingles(text):
    toks = dedup_key(text).split()
    if len(toks) < SHINGLE:
        return {" ".join(toks)}
    return {" ".join(toks[i:i + SHINGLE]) for i in range(len(toks) - SHINGLE + 1)}


def jaccard(a, b):
    return len(a & b) / len(a | b) if a or b else 1.0


by_user_product = defaultdict(list)
for r in kept:
    product = f"{r['brand']}|{r['model']}" if r["model"] else r["parent_asin"]
    by_user_product[(r["user_id"], product)].append(r)

final = []
for group in by_user_product.values():
    accepted = []  # (review, shingle set), earliest first
    for r in group:
        sh = shingles(r["dup_key"])
        exact = next((a for a, _ in accepted if a["dup_key"] == r["dup_key"]), None)
        if exact:
            drop("duplicate_same_user_same_product_exact", r, duplicate_of=exact["text"][:300])
            continue
        near = next((a for a, ash in accepted if jaccard(sh, ash) >= NEAR_DUP_THRESHOLD), None)
        if near:
            drop("duplicate_same_user_same_product_near", r, duplicate_of=near["text"][:300])
            continue
        accepted.append((r, sh))
    final.extend(a for a, _ in accepted)
final.sort(key=lambda r: r["timestamp"])
print(f"duplicates done: {len(final):,} left ({time.time() - t0:.0f}s)", flush=True)

# ---- identical texts kept (other accounts / other products) share a dup_group, so retrieval can show one copy
group_size = Counter(r["dup_key"] for r in final)
report["dup_groups"] = sum(1 for c in group_size.values() if c > 1)
report["kept_identical_text_other_account_or_product"] = sum(c - 1 for c in group_size.values() if c > 1)

with open(PROCESSED / "clean_reviews.jsonl", "w", encoding="utf8") as out:
    for r in final:
        rid = hashlib.sha1(f"{r['parent_asin']}|{r['user_id']}|{r['timestamp']}".encode()).hexdigest()[:12]
        out.write(json.dumps({
            "review_id": rid,
            "parent_asin": r["parent_asin"],
            "asin": r["asin"],
            "brand": r["brand"],
            "model": r["model"],
            "is_renewed": r["is_renewed"],
            "product_title": r["product_title"],
            "found_in": r["found_in"],
            "rating": int(r["rating"]),
            "title": r["title"],
            "text": r["text"],
            # normalized copy for BM25 / embeddings / classifier; `text` stays original for display
            "text_norm": normalize(r["text"], fix_spelling=r["is_short"]),
            "n_words": r["n_words"],
            "is_short": r["is_short"],
            "dup_group": hashlib.sha1(r["dup_key"].encode()).hexdigest()[:10] if group_size[r["dup_key"]] > 1 else None,
            "date": r["date"],
            "verified_purchase": r["verified_purchase"],
            "helpful_vote": r["helpful_vote"],
            # user ids are hashed: the corpus is shared in the submission and needs no raw Amazon ids
            "user_hash": hashlib.sha1(r["user_id"].encode()).hexdigest()[:10],
            "source_url": f"https://www.amazon.com/dp/{r['parent_asin']}",
        }, ensure_ascii=False) + "\n")

report["output_reviews"] = len(final)
report["output_words"] = sum(r["n_words"] for r in final)
report["seconds"] = round(time.time() - t0)
json.dump(dict(report), open(REPORTS / "02_cleaning_report.json", "w"), indent=2)
with open(REPORTS / "02_removed_samples.jsonl", "w", encoding="utf8") as f:
    for reason in samples:
        for s in samples[reason]:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
with open(REPORTS / "02_kept_short_samples.jsonl", "w", encoding="utf8") as f:
    for s in kept_short_samples:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")
print(json.dumps(dict(report), indent=2))

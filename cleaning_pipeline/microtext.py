"""Microtext normalization for review text.

The corpus keeps the original review in `text` (shown to users as evidence) and adds a normalized copy,
`text_norm`, for BM25, embeddings and the sentiment classifier. Normalization is conservative: it never
lowercases (CAPS carry intensity) and only rewrites tokens it is confident about.

  1. emoji -> words          "Love it 👍"            -> "Love it thumbs up"
  2. emoticons -> words      "works :)"              -> "works smile"
  3. elongations             "soooo GRRRREAT"        -> "so GREAT"  (double letter if a real word: "coool" -> "cool", else single)
  4. repeated punctuation    "Terrible!!!!!" -> "Terrible!", "again...couldn't" -> "again... couldn't"
  5. slang / abbreviations   "gr8 phone thx, u wont regret" -> "great phone thanks, you won't regret"
  6. misspelled opinion words, short reviews only   "exelent" -> "excellent"
"""
import re
from difflib import get_close_matches
from functools import lru_cache

import emoji
from nltk.corpus import opinion_lexicon
from nltk.corpus import words as nltk_words

OPINION = set(opinion_lexicon.positive()) | set(opinion_lexicon.negative())
ENGLISH = {w.lower() for w in nltk_words.words()}
VOCAB = ENGLISH | OPINION
OPINION_LIST = sorted(OPINION)

EMOTICONS = [  # order matters: specific forms before general ones
    (r"<3", " love "), (r":'\(", " crying "), (r";-?\)", " wink "), (r"[:=]-?\)+|\(+-?[:=]", " smile "),
    (r"[:;=]-?D\b", " laugh "), (r"[:;=]-?\(+|\)+-?[:;=]", " sad "), (r"[:;=]-?[pP]\b", " playful "),
    (r"(?<![a-z])[:=]-?/(?!/)", " unsure "),
]
EMOTICON_RE = [(re.compile(p), w) for p, w in EMOTICONS]

SLANG = {
    "u": "you", "ur": "your", "gr8": "great", "thx": "thanks", "thnx": "thanks", "tks": "thanks", "ty": "thank you",
    "pls": "please", "plz": "please", "bc": "because", "b/c": "because", "cuz": "because", "coz": "because",
    "w/": "with", "w/o": "without", "idk": "I don't know", "imo": "in my opinion", "imho": "in my opinion",
    "tbh": "to be honest", "omg": "oh my god", "btw": "by the way", "ppl": "people", "txt": "text",
    "msg": "message", "msgs": "messages", "pic": "picture", "pics": "pictures", "batt": "battery",
    "cam": "camera", "abt": "about", "thru": "through", "tho": "though", "luv": "love", "nite": "night",
    "dont": "don't", "cant": "can't", "wont": "won't", "doesnt": "doesn't", "didnt": "didn't", "isnt": "isn't",
    "wasnt": "wasn't", "arent": "aren't", "wouldnt": "wouldn't", "couldnt": "couldn't", "shouldnt": "shouldn't",
    "havent": "haven't", "im": "I'm", "ive": "I've", "youre": "you're", "theyre": "they're", "thats": "that's",
    "whats": "what's", "kinda": "kind of", "gonna": "going to",
    "wanna": "want to", "gotta": "got to", "fav": "favorite", "fave": "favorite", "def": "definitely",
}
# words are rewritten in place, so spacing and things like "4GB/64GB" are left untouched
WORD_RE = re.compile(r"(?<![\w/'])(w/o|w/|b/c|[A-Za-z0-9']+)(?![\w/])")
ELONGATED_RE = re.compile(r"([A-Za-z])\1{2,}")
ROMAN_RE = re.compile(r"^[IVX]+$")
SKIN_TONE_RE = re.compile("[\U0001F3FB-\U0001F3FF]")  # "🤘🏼" -> "🤘", not "... medium-light skin tone"
REPEAT_PUNCT_RE = re.compile(r"([!?,])\1+")
ELLIPSIS_RE = re.compile(r"\.{2,}\s*")  # "again...couldn't" -> "again... couldn't" (keeps the words apart)
WS_RE = re.compile(r"\s+")


def _match_case(src, repl):
    if src.isupper() and len(src) > 1:
        return repl.upper()
    if src[:1].isupper():
        return repl[:1].upper() + repl[1:]
    return repl


def _fix_elongation(tok):
    """'soooo' -> 'so', 'GRRRREAT' -> 'GREAT', 'cooool' -> 'cool': double letter if that is a word, else single."""
    if not ELONGATED_RE.search(tok) or ROMAN_RE.match(tok):  # "Galaxy S III" is not an elongation
        return tok
    two = ELONGATED_RE.sub(r"\1\1", tok)
    one = ELONGATED_RE.sub(r"\1", tok)
    for cand in (two, one):
        if cand.lower() in VOCAB:
            return cand
    return two


@lru_cache(maxsize=None)
def _spelling(tok_lower):
    if len(tok_lower) < 4 or not tok_lower.isalpha() or _is_english(tok_lower):
        return None
    # the opinion lexicon itself contains misspellings ("exellent"), so only correct to real dictionary words
    matches = get_close_matches(tok_lower, OPINION_LIST, n=3, cutoff=0.85)
    return next((m for m in matches if m in ENGLISH), None)


def _is_english(w):
    """Dictionary word, including inflected forms the word list lacks ("damaged", "complaints")."""
    if w in ENGLISH:
        return True
    for suf, add in (("s", ""), ("es", ""), ("ed", ""), ("ed", "e"), ("d", ""), ("ing", ""), ("ing", "e"),
                     ("ly", ""), ("er", ""), ("est", ""), ("ies", "y"), ("ied", "y")):
        if w.endswith(suf) and w[: -len(suf)] + add in ENGLISH:
            return True
    return False


def normalize(text, fix_spelling=False):
    if not text:
        return ""
    text = emoji.demojize(SKIN_TONE_RE.sub("", text), delimiters=(" ", " ")).replace("_", " ")
    for rx, word in EMOTICON_RE:
        text = rx.sub(word, text)
    text = REPEAT_PUNCT_RE.sub(r"\1", text)
    text = ELLIPSIS_RE.sub("... ", text)

    def fix(m):
        tok = _fix_elongation(m.group(1))
        low = tok.lower()
        if low in SLANG:
            return _match_case(tok, SLANG[low])
        if fix_spelling and (corrected := _spelling(low)):
            return _match_case(tok, corrected)
        return tok

    return WS_RE.sub(" ", WORD_RE.sub(fix, text)).strip()

"""Shared helpers for the knowledge-base pipeline: paths, text normalization, brand/model mapping."""
import html
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_REVIEWS = ROOT / "Cell_Phones_and_Accessories.jsonl" / "Cell_Phones_and_Accessories.jsonl"
RAW_META = ROOT / "meta_Cell_Phones_and_Accessories.jsonl" / "meta_Cell_Phones_and_Accessories.jsonl"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
for d in (INTERIM, PROCESSED, REPORTS):
    d.mkdir(parents=True, exist_ok=True)

WORD_RE = re.compile(r"[A-Za-z0-9']+")


def words(text):
    return WORD_RE.findall(text)


# ---------------------------------------------------------------- text cleaning
TAG_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
CTRL_RE = re.compile(r"[\u0000-\u0008\u000b-\u001f\u007f-\u009f​-‏﻿]")
WS_RE = re.compile(r"\s+")
# Amazon auto-generated titles ("Five Stars") leak the star rating into the text, so we blank them
AUTO_TITLE_RE = re.compile(r"^\s*(one|two|three|four|five)\s+stars?\s*$", re.I)


def clean_text(text):
    """HTML unescape, strip tags/URLs/control chars, NFKC-normalize, collapse whitespace.
    Emoji, casing and punctuation are kept on purpose: they carry sentiment."""
    if not text:
        return ""
    text = html.unescape(html.unescape(text))
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.I)
    text = TAG_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = unicodedata.normalize("NFKC", text)
    text = CTRL_RE.sub("", text)
    return WS_RE.sub(" ", text).strip()


def clean_title(title):
    title = clean_text(title)
    return "" if AUTO_TITLE_RE.match(title) else title


def dedup_key(text):
    """Normalized form used for exact-duplicate detection."""
    return " ".join(w.lower() for w in words(text))


# ---------------------------------------------------------------- product filtering
# Items filed under "Cell Phones" that are not handsets (service plans, airtime cards, SIM kits).
# Phones sold *with* an airtime bundle are kept: the title must also not mention a phone.
PLAN_RE = re.compile(
    r"\b(airtime|service plan|day plan|refill card|prepaid card|sim kit|sim card kit|"
    r"activation kit|unlimited talk)\b", re.I)
PHONE_WORD_RE = re.compile(r"phone|galaxy|iphone|pixel|moto|stylo|\blg\b|flip|nokia|alcatel", re.I)


ACCESSORY_RE = re.compile(r"^\s*for\s", re.I)  # "For LG Optimus G Pro ... Kickstand Case"


def is_non_phone(title):
    """For listings inside the 'Cell Phones' category: drop service plans and stray accessories."""
    return (bool(PLAN_RE.search(title)) and not PHONE_WORD_RE.search(title)) or bool(ACCESSORY_RE.search(title))


# ---- phones filed outside "Cell Phones" (Wireless Trade-In, no category, even Chargers/Cases)
# A listing counts as a phone when its title names a phone model AND has a phone spec,
# AND has no accessory word, AND does not start like an accessory title ("For ...", "Compatible ...").
PHONE_MODEL_RE = re.compile(
    r"\b(iphone\s?(\d{1,2}|x[rs]?|se)|galaxy\s?(s\d|note\s?\d|a\d|j\d|m\d|z\s?(fold|flip))|pixel\s?\d|"
    r"moto\s?[gezx]\d*|motorola\s?(one|edge|razr)|razr|oneplus\s?(\d|nord)|redmi|poco\s?[a-z]\d|xiaomi\s?mi|"
    r"nokia\s?\d|lg\s?(stylo|g\d|v\d|k\d)|blu\s?(vivo|studio|g\d|c\d|view|bold)|huawei\s?(p\d|mate|y\d|honor)|"
    r"sony\s?xperia|essential phone|zte\s?(blade|axon|z\d))", re.I)
PHONE_SPEC_RE = re.compile(
    r"\b\d{1,4}\s?gb\b|\bunlocked\b|\(renewed|\brefurbished\b|\bprepaid\b|\bdual sim\b|\bsmartphone\b|"
    r"\bcell ?phone\b|\b(at&t|verizon|t-mobile|sprint|cricket|boost)\b", re.I)
_ACC_WORDS = (
    "case|cover|shell|screen protector|protector|charger|charging|cable|holster|mount|holder|stand|adapter|"
    "glass|skin|stylus|stylus pen|earbud|bud|headset|headphone|earphone|replacement|lens|ring|strap|band|"
    "bracelet|sticker|pouch|wallet|clip|clamp|dock|tripod|selfie stick|power bank|controller|gimbal|watch|"
    "tablet|ipad|packaging|digitizer|lcd|housing|frame|sim tray|keyboard|speaker|car kit|decal|film|armband|"
    "nano sim|unlock service|unlocking service|bumper|sleeve|lanyard|grip|kickstand|battery pack|"
    "battery for|li-?ion|repair|tool|screwdriver|plug|stopper")  # not plain "battery": phones list "5000mAh Battery"
# every accessory word also matches its plural ("Protectors", "Cases", "Headphones")
ACCESSORY_WORD_RE = re.compile(
    rf"\b({_ACC_WORDS})(s|es)?\b|\bsim card(?! included)|\bbox only\b|\btab\b|\bslim fit\b|"
    r"\bfor (apple|samsung|iphone|galaxy|google|pixel|lg|motorola|moto)\b", re.I)  # "Slim Fit ... for Apple iPhone"
ACCESSORY_LEAD_RE = re.compile(r"^\s*(for|compatible|\[|\(|\d+[- ]?(pack|pcs|piece))", re.I)


def is_phone_outside_category(title):
    return bool(PHONE_MODEL_RE.search(title) and PHONE_SPEC_RE.search(title)
                and not ACCESSORY_WORD_RE.search(title) and not ACCESSORY_LEAD_RE.search(title))

# ---------------------------------------------------------------- brand / model
BRAND_ALIASES = {
    "apple": "Apple", "samsung": "Samsung", "google": "Google", "motorola": "Motorola",
    "moto": "Motorola", "oneplus": "OnePlus", "xiaomi": "Xiaomi", "redmi": "Xiaomi", "poco": "Xiaomi",
    "lg": "LG", "lg electronics": "LG", "nokia": "Nokia", "hmd global": "Nokia", "blu": "BLU",
    "tcl": "TCL", "huawei": "Huawei", "sony": "Sony", "essential": "Essential",
    "essential products": "Essential", "zte": "ZTE", "alcatel": "Alcatel", "asus": "ASUS",
    "kyocera": "Kyocera", "ulefone": "Ulefone", "umidigi": "UMIDIGI", "blackview": "Blackview",
    "oukitel": "Oukitel", "doogee": "Doogee", "htc": "HTC", "blackberry": "BlackBerry",
    "realme": "realme", "oppo": "OPPO", "vivo": "vivo", "cat": "CAT", "honor": "Honor",
}
TITLE_BRAND_RE = [  # checked in order against the product title
    (re.compile(r"\biphone\b", re.I), "Apple"),
    (re.compile(r"\bgalaxy\b", re.I), "Samsung"),
    (re.compile(r"\bpixel\b", re.I), "Google"),
    (re.compile(r"\bmoto\b|\bmotorola\b|\brazr\b", re.I), "Motorola"),
    (re.compile(r"\bone ?plus\b", re.I), "OnePlus"),
    (re.compile(r"\bxiaomi\b|\bredmi\b|\bpoco\b", re.I), "Xiaomi"),
]

MODEL_RE = {
    "Apple": re.compile(r"iphone\s*(\d{1,2}|xs|xr|x|se)(\s*(pro max|pro|mini|plus|max))?", re.I),
    "Samsung": re.compile(
        r"galaxy\s*(z\s*(fold|flip)\s*\d?|note\s*\d{1,2}|s\d{1,2}|a\d{1,2}s?|j\d|m\d{2})"
        r"(\s*(ultra|fe|plus|\+|lite|edge|e)\b|\+|e\b)?", re.I),
    "Google": re.compile(r"pixel\s*(\d)(a)?(\s*(xl|pro))?", re.I),
    "OnePlus": re.compile(r"one\s?plus\s*(nord\s*(n?\d{2,3}|ce\s*\d?)?|\d{1,2}[tr]?)(\s*pro)?", re.I),
    "Motorola": re.compile(
        r"(moto\s*(g|e|x|z)\s*(\d{1,2}|fast|power|play|stylus|plus|pure|5g)?(\s*(power|play|plus|5g))?"
        r"|motorola\s*(edge|one|razr)(\s*(\+|plus|5g|ace|fusion|zoom))?|razr)", re.I),
    "Xiaomi": re.compile(r"(redmi\s*note\s*\d{1,2}(\s*pro)?|redmi\s*\d{1,2}[a-z]?|poco\s*[a-z]\d(\s*pro)?|mi\s*\d{1,2}t?(\s*(pro|lite))?)", re.I),
}
YEAR_RE = re.compile(r"\b(201[0-9]|202[0-4])\b")
GEN_RE = re.compile(r"\((\d)(st|nd|rd|th) gen", re.I)


CARRIERS = {"tracfone", "straight talk", "total wireless", "simple mobile", "boost mobile", "verizon",
            "net10", "at&t", "cricket", "t-mobile", "virgin mobile", "metropcs", "amazon renewed"}
CORP_SUFFIX_RE = re.compile(r"[.,]|\b(inc|co|corp|corporation|ltd|llc|electronics|mobile|usa|us)\b", re.I)


def _norm_brand(v):
    return WS_RE.sub(" ", CORP_SUFFIX_RE.sub(" ", v.lower())).strip()


def detect_brand(title, details):
    """Manufacturer brand. Carriers (Tracfone, Straight Talk...) are skipped in favour of the maker in the title."""
    for key in ("Brand", "Manufacturer"):
        v = _norm_brand(details.get(key) or "")
        if v in BRAND_ALIASES:
            return BRAND_ALIASES[v]
    for rx, brand in TITLE_BRAND_RE:
        if rx.search(title):
            return brand
    low = title.lower()
    for alias, brand in BRAND_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", low):
            return brand
    v = (details.get("Brand") or details.get("Manufacturer") or "").strip()
    return v.title() if v and v.lower() not in CARRIERS else "Other"


def tidy_model(s):
    s = WS_RE.sub(" ", s.replace("+", " Plus")).strip()
    return " ".join(w.upper() if w.lower() in {"fe", "xl", "xs", "xr", "se", "5g", "ce"} else
                    w if any(c.isdigit() for c in w) and len(w) <= 3 else w.capitalize() for w in s.split())


def detect_model(brand, title, details):
    """Normalized model family, e.g. 'Galaxy S21 Ultra', 'Moto G Power (2021)'. Empty string if unknown."""
    rx = MODEL_RE.get(brand)
    if rx:
        m = rx.search(title)
        if m:
            model = tidy_model(m.group(0))
            if brand == "Samsung":
                model = re.sub(r"(S\d{1,2}) ?E\b", r"\1e", model)
            if brand == "Motorola":
                y, g = YEAR_RE.search(title), GEN_RE.search(title)
                if y:
                    model += f" ({y.group(1)})"
                elif g:
                    model += f" (Gen {g.group(1)})"
            return model.replace("Iphone", "iPhone")
    model = model_from_title(brand, title)
    if model:
        return model
    name = (details.get("Model Name") or "").strip()
    return tidy_model(name) if name and len(name) <= 40 else ""


# words that end the model name in a listing title ("LG Stylo 4 - 32 GB - Unlocked ...")
TITLE_STOP_RE = re.compile(
    r"\s[-–|,(/:]|\s(\d+\s?gb|\d+(\.\d+)?\s?(\"|”|inch|in\b)|gsm|cdma|unlocked|smartphone|smart phone|cell|phone|"
    r"prepaid|dual|4g|3g|lte|5g|android|factory|with|for|international|global|us\b|version|black|white|blue)", re.I)
CARRIER_PREFIX_RE = re.compile(r"^\s*(" + "|".join(re.escape(c) for c in CARRIERS) + r")\b[\s-]*", re.I)


def model_from_title(brand, title):
    """Fallback: the words after the brand name, up to the first spec/separator word."""
    t = CARRIER_PREFIX_RE.sub("", title)
    aliases = [a for a, b in BRAND_ALIASES.items() if b == brand] + [brand.lower()]
    for a in sorted(aliases, key=len, reverse=True):
        t = re.sub(rf"^\s*{re.escape(a)}\b[\s-]*", "", t, flags=re.I)
    t = TITLE_STOP_RE.split(" " + t, maxsplit=1)[0].strip()
    toks = [w.strip(",.;:") for w in t.split()[:4]]
    toks = [w for w in toks if w]
    if not toks or not any(c.isalnum() for c in toks[0]) or len(" ".join(toks)) < 2:
        return ""
    return tidy_model(" ".join(toks))


def is_renewed(title, store):
    return store == "Amazon Renewed" or bool(re.search(r"\((renewed|refurbished)", title, re.I))

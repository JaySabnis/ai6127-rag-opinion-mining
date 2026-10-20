"""Step 1: extract handset products and their reviews from the raw Amazon Reviews 2023 files.

A listing is a handset when it is in the "Cell Phones" category (minus service plans / stray accessories),
OR when its title is clearly a phone even though Amazon filed it elsewhere (Wireless Trade-In, no category,
even Chargers/Cases) - see common.is_phone_outside_category. Phone-or-not is decided per listing, never
from review text, so a headphone review that mentions an iPhone stays out.

Input : raw meta + review JSONL for Cell_Phones_and_Accessories (McAuley Lab, Amazon Reviews 2023)
Output: data/interim/phone_products.jsonl   one row per handset listing (brand, model, renewed flag)
        data/interim/phone_reviews_raw.jsonl reviews of those listings, joined with product info
        reports/01_extract_report.json
"""
import json
import re
import time
from collections import Counter

from common import (INTERIM, RAW_META, RAW_REVIEWS, REPORTS, detect_brand, detect_model, is_non_phone,
                    is_phone_outside_category, is_renewed)

t0 = time.time()
report = Counter()

# ---- pass 1: handset listings from metadata (categories[1] == "Cell Phones")
products = {}
with open(RAW_META, encoding="utf8") as f, open(INTERIM / "phone_products.jsonl", "w", encoding="utf8") as out:
    for line in f:
        report["meta_items_total"] += 1
        d = json.loads(line)
        cats = d.get("categories") or []
        title = d.get("title") or ""
        if len(cats) >= 2 and cats[1] == "Cell Phones":
            report["meta_items_cell_phones_category"] += 1
            if is_non_phone(title):
                report["meta_items_dropped_not_a_handset"] += 1
                continue
            found_in = "Cell Phones"
        elif is_phone_outside_category(title):
            found_in = " > ".join(cats[1:3]) or "(no category)"
            report["meta_items_phone_outside_category"] += 1
        else:
            continue
        details = d.get("details") or {}
        brand = detect_brand(title, details)
        p = {
            "parent_asin": d["parent_asin"],
            "brand": brand,
            "model": detect_model(brand, title, details),
            "is_renewed": is_renewed(title, d.get("store") or ""),
            "product_title": title,
            "found_in": found_in,
            "store": d.get("store"),
            "price": d.get("price"),
            "average_rating": d.get("average_rating"),
            "rating_number": d.get("rating_number"),
            "date_first_available": details.get("Date First Available"),
        }
        products[p["parent_asin"]] = p
        out.write(json.dumps(p, ensure_ascii=False) + "\n")
report["handset_products"] = len(products)
print(f"pass 1: {len(products):,} handset listings ({time.time() - t0:.0f}s)", flush=True)

# ---- pass 2: reviews of those listings
asin_re = re.compile(r'"parent_asin": "([^"]+)"')
keep_fields = ("rating", "title", "text", "asin", "parent_asin", "user_id", "timestamp",
               "helpful_vote", "verified_purchase")
with open(RAW_REVIEWS, encoding="utf8") as f, open(INTERIM / "phone_reviews_raw.jsonl", "w", encoding="utf8") as out:
    for line in f:
        report["reviews_total"] += 1
        m = asin_re.search(line)
        if not m or m.group(1) not in products:
            continue
        r = json.loads(line)
        p = products[r["parent_asin"]]
        row = {k: r.get(k) for k in keep_fields}
        row.update(brand=p["brand"], model=p["model"], is_renewed=p["is_renewed"],
                   product_title=p["product_title"], found_in=p["found_in"])
        out.write(json.dumps(row, ensure_ascii=False) + "\n")
        report["reviews_of_handsets"] += 1

report["seconds"] = round(time.time() - t0)
json.dump(dict(report), open(REPORTS / "01_extract_report.json", "w"), indent=2)
print(json.dumps(dict(report), indent=2))

# Cleaned smartphone-review corpus

**395,038 English smartphone reviews (22.8M words), 2003–2023.** Built from Amazon Reviews 2023 –
Cell_Phones_and_Accessories (McAuley Lab, https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023).

**Download (Kaggle):** https://kaggle.com/datasets/e7c57394adb18d36e3b2248168253834bb2eff982d8b682c2eefd1736ce08a5b

- `clean_reviews.jsonl` (472 MB): the cleaned corpus, one JSON review per line.
- `phone_products.jsonl` (4 MB): optional product table (see the end of [Fields](#fields)).

## How it was cleaned

1. **Phones only.** Whether a listing is a phone was decided from its product metadata (category + title),
   not from review text. This kept 11,527 phone listings out of 1.29M products, including phones Amazon filed
   under other categories (e.g. "Wireless Trade-In"), and took their 425,731 reviews.
2. **Text cleanup.** Removed HTML, URLs and control characters, and normalized Unicode and whitespace.
   Blanked auto-generated titles ("Five Stars"), which give away the rating.
3. **Short reviews (< 5 words).** Kept if they carry an opinion or a phone feature ("Terrible!", "Dead in two
   months", "👍"), using NLTK VADER and the Hu & Liu opinion lexicon. Removed if empty ("Na") or Spanish (−9,112).
4. **English only.** Removed longer non-English reviews (−12,077).
5. **Seller-only reviews.** Removed reviews about only the seller, shipping or refund, with nothing about
   the phone (−3,621).
6. **Duplicates.** Removed when the **same account** posted an identical or near-identical text
   (Jaccard ≥ 0.8) on the **same phone model** (−5,883). Identical texts from different accounts are kept,
   because they are separate opinions; they share a `dup_group` id.
7. **Microtext normalization.** A normalized copy of every review is stored in `text_norm`. It converts emoji and
   emoticons to words (👍 → "thumbs up", :) → "smile"), shortens elongated words ("soooo" → "so"), collapses
   repeated punctuation ("!!!" → "!"), expands slang ("gr8", "u", "w/o", "dont" → "great", "you",
   "without", "don't") and fixes misspellings in short reviews ("exelent" → "excellent"). Case is kept.
   `text` stays exactly as written.

The star distribution is left as it is (1★ 78k · 2★ 27k · 3★ 31k · 4★ 53k · 5★ 206k), so opinion splits reflect
real buyers. Top brands: Samsung 90k, Apple 70k, Motorola 56k, BLU 38k, LG 34k.

## Fields

| Key | Type | Description |
|---|---|---|
| `review_id` | str | Unique id of the review |
| `parent_asin` | str | Amazon product id (all colours/sizes of one listing) |
| `asin` | str | Amazon id of the exact variant reviewed |
| `brand` | str | Phone maker, e.g. `Samsung`, `Apple` |
| `model` | str | Phone model, e.g. `Galaxy S10`, `iPhone 11 Pro Max` (99% filled; may be null) |
| `is_renewed` | bool | Refurbished/renewed listing |
| `product_title` | str | Amazon listing title |
| `found_in` | str | Amazon category of the listing (`Cell Phones`, or the other category the phone was found in) |
| `rating` | int | Stars, 1–5 |
| `title` | str | Review title (empty if it was an auto-title) |
| `text` | str | Review text, original wording |
| `text_norm` | str | Review text after microtext normalization |
| `n_words` | int | Number of words in `text` |
| `is_short` | bool | `text` has fewer than 5 words |
| `dup_group` | str/null | Shared id for reviews with identical text from different accounts or products |
| `date` | str | Review date, `YYYY-MM-DD` |
| `verified_purchase` | bool | Amazon verified purchase |
| `helpful_vote` | int | "Helpful" votes from other users |
| `user_hash` | str | Anonymized reviewer id |
| `source_url` | str | Amazon product page |

Example:

```json
{"review_id": "ce07282e5976", "parent_asin": "B00009J5VW", "asin": "B00009J5VW", "brand": "Nokia",
 "model": "3650", "is_renewed": false, "product_title": "Nokia 3650 Phone (T-Mobile)", "found_in": "Nokia",
 "rating": 4, "title": "nice phone, little bit delicate for routine use",
 "text": "This is excellent phone with lots of features. ...", "text_norm": "This is excellent phone with lots of features. ...",
 "n_words": 32, "is_short": false, "dup_group": null, "date": "2003-05-02", "verified_purchase": false,
 "helpful_vote": 2, "user_hash": "f14e4f9b75", "source_url": "https://www.amazon.com/dp/B00009J5VW"}
```

### Optional: `phone_products.jsonl`

**Totally optional; it can be skipped.** It is added just in case it is useful for statistics or the UI
(e.g. a product card with price and average rating). It has one row per phone listing (11,527), and joins to the
reviews on `parent_asin`. The review file already contains brand, model and title, so nothing depends on it.

| Key | Description |
|---|---|
| `parent_asin` | Amazon product id (join key) |
| `brand`, `model`, `is_renewed`, `product_title`, `found_in` | Same as in the review file |
| `store` | Seller/store name on Amazon |
| `price` | Listing price in USD at crawl time (missing for 77% of listings) |
| `average_rating`, `rating_number` | Amazon's average stars and number of ratings for the listing |
| `date_first_available` | Date the listing first appeared on Amazon, e.g. "March 8, 2019" |

## Using it

Nothing to install beyond Python + pandas.

- **On your own machine:** open the link above → **Download**, and unzip into `data/`.
- **In a Kaggle notebook:** open the link above → **New Notebook**. The files appear under
  `/kaggle/input/amazon-smartphone-reviews-cleaned-ai6127/`.
- **Kaggle API** (only for accounts added as collaborators on the dataset): `pip install kaggle`, put your
  API token in `~/.kaggle/kaggle.json`, then
  ```
  kaggle datasets download yazhiniyant/amazon-smartphone-reviews-cleaned-ai6127 --unzip -p data
  ```

```python
import pandas as pd
df = pd.read_json("data/clean_reviews.jsonl", lines=True)   # in Kaggle: /kaggle/input/amazon-smartphone-reviews-cleaned-ai6127/clean_reviews.jsonl
df[df.model == "Galaxy S10"].rating.value_counts()
```

## Re-running the cleaning (optional)

Requirements: Python 3.10+ and about 20 GB of free disk space.

1. Download `Cell_Phones_and_Accessories.jsonl` and `meta_Cell_Phones_and_Accessories.jsonl` from the
   Hugging Face page above, and place them in the project root:
   ```
   <root>/Cell_Phones_and_Accessories.jsonl/Cell_Phones_and_Accessories.jsonl
   <root>/meta_Cell_Phones_and_Accessories.jsonl/meta_Cell_Phones_and_Accessories.jsonl
   <root>/cleaning_pipeline/
   ```
2. Install the dependencies (`nltk`, `langdetect`, `emoji`) and the NLTK data:
   ```
   pip install -r cleaning_pipeline/requirements.txt
   python -c "import nltk; [nltk.download(x) for x in ('vader_lexicon','opinion_lexicon','words')]"
   ```
3. Run both steps, in order:
   ```
   cd cleaning_pipeline
   python 01_extract_phones.py   # step 1, phone listings + their reviews  (~4 min)
   python 02_clean.py            # steps 2-7                               (~5 min)
   ```
   Output: `data/processed/clean_reviews.jsonl`. Counts for every rule are in `reports/02_cleaning_report.json`,
   and samples of removed reviews with the reason are in `reports/02_removed_samples.jsonl`.

| File | Purpose |
|---|---|
| `01_extract_phones.py` | Finds phone listings in the metadata and extracts their reviews |
| `02_clean.py` | Cleaning, short-review check, language, seller-only and duplicate rules |
| `microtext.py` | Microtext normalization (`text_norm`) |
| `common.py` | Shared paths, phone/brand/model detection |

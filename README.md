# AI6127 – RAG-based aspect-level opinion mining for smartphone reviews

**Cleaned dataset (Kaggle):** https://kaggle.com/datasets/e7c57394adb18d36e3b2248168253834bb2eff982d8b682c2eefd1736ce08a5b

- Built from Amazon Reviews 2023 – Cell_Phones_and_Accessories (McAuley Lab): 395,038 English smartphone reviews, 22.8M words (2003–2023).
- Phone listings were identified from product metadata (11,527 listings), and their 425,731 reviews were extracted, with brand, model and renewed flag attached.
- Cleaning removed non-English text, contentless short reviews, seller-only reviews and same-account duplicates on the same phone; short reviews with real opinions were kept.
- Microtext normalization (emoji, emoticons, slang, elongations, misspellings) is stored in `text_norm`; the original text is kept in `text`.
- Code, field descriptions and steps to re-run are in [`cleaning_pipeline/`](cleaning_pipeline/README.md).

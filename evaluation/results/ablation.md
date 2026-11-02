### Classification

| Run | Subjectivity macro-F1 | Polarity macro-F1 | Overall macro-F1 | Overall accuracy |
|---|---|---|---|---|
| deepseek_overall | 0.830 | 0.937 | 0.865 | 0.939 |

### RAG (RAGAS)

| Run | Faithfulness | Answer relevancy | Context precision | Context relevance | Queries |
|---|---|---|---|---|---|
| B40_bm25 | 0.950 | 0.812 | 0.429 | 0.787 | 40 |
| B40_dense | 0.981 | 0.863 | 0.611 | 0.969 | 40 |
| B40_learned | 0.967 | 0.842 | 0.836 | 0.988 | 40 |
| B40_learned_checker | 0.750 | 0.771 | 0.703 | 0.988 | 40 |
| B40_learned_checker_fixed | 0.966 | 0.848 | 0.766 | 0.988 | 40 |

### Latency

| Run | Mean total (s) | p95 total (s) | Cold total (s) | Queries/min |
|---|---|---|---|---|
| B40_bm25 | 4.170 | 6.201 | 3.292 | 14.390 |
| B40_dense | 4.814 | 7.260 | 3.191 | 12.460 |
| B40_learned | 6.005 | 7.244 | 4.958 | 9.990 |
| B40_learned_checker | 9.193 | 12.207 | 6.941 | 6.530 |
| B40_learned_checker_fixed | 8.007 | 9.794 | 6.886 | 7.490 |

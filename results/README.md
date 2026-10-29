# Results Index

- [Chinese HTML report](ablation_latest/report_zh.html)
- [English HTML report](ablation_latest/report_en.html)
- [Latest eight-system ablation guide](ablation_latest/README.md)
- [Dedicated training100 results](reranker_training_100/README.md)

The current report uses eight configurations, the model trained on 100 dedicated questions, and 2,632 common evaluation judgments for all four metrics. It reuses saved retrieval results and recomputes metrics; report preparation does not repeat retrieval.

`ablation_40`, `ablation_8`, `reranker_design` and `reranker_learned` retain historical metrics, candidates and reproduction artifacts. Superseded display reports, duplicate result archives and obsolete diagnostics were removed; see `ablation_latest/cleanup_manifest.json`.

Training and evaluation datasets are under `evaluation/`. Delivery archives can be generated locally; they are intentionally excluded from Git because their unpacked contents are available in the repository.

# CLAUDE.md

Research codebase: eXtreme Multi-label Ranking for biomedical Entity Linking (UMLS/PubTator).
Read `AGENTS.md` for the analysis/response conventions — this file is the operational cheat sheet.

## Layout

- `xmr4el/xmr/` — `XModel` (`model.py`, orchestration/persistence) and `HierarchicaMLModel` (`base.py`, the recursive tree train/predict). **The two files that matter.**
- `xmr4el/featurization/` — `Preprocessor` (TSV + PubTator loaders), `TextEncoder` (`emb_flag` 1=tfidf, 2=tfidf+transformer, 3=transformer, 4=`[SEP]`-split), `label_embedding_factory.py` (PIFA).
- `xmr4el/clustering/`, `matcher/`, `ranker/` — each a thin `train.py` + `model.py` pair over the wrappers.
- `xmr4el/models/*_wrapper/` — registry-style wrappers (tfidf, sentence-transformers, SVD/UMAP, balanced k-means, sklearn classifiers). Config `type` strings map here.
- `test/xmr4el/` — argparse **scripts**, not pytest tests. `test/pubtor/` — UMLS DB experiments.
- `.docs/` — pipeline + UMLS schema notes. `.pubtor/` (gitignored) — UMLS Postgres KB layer (models/repository/queries).
- `.models/xmr4el_base_config.json` — the single source of truth for every component's type + kwargs.

## Commands

```bash
python test/xmr4el/test_train_pipeline.py -train_path <pubtator.txt> [-model_config .models/xmr4el_base_config.json] [-ds_len N]
python test/xmr4el/test_evaluate_pipeline.py -xmodel_path <saved_tree_dir> -test_path <pubtator.txt> [-beam_size 5] [-topk 20]
docker build -f xmr4el.dockerfile .   # CUDA 12.8 runtime; postgres.dockerfile for the UMLS db
```

Trained trees are saved to `test/test_data/saved_trees` (gitignored). No test runner, no lint config, no CI.

## Gotchas

- `pyproject.toml` pins RAPIDS (`cuml-cu12`, `cudf-cu12`) and CUDA torch; `requirements.txt` is the CPU-ish subset and the two have drifted. On macOS/CPU install from `requirements.txt`.
- ARM needs `LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1` (the train script sets it itself).
- `README.md` refers to `_test/`; the directory is `test/`.
- DB creds are hardcoded in `db_params.json` / `postgres.dockerfile` (dev-only).
- Known scoring issues are already triaged in `TODO.md` with file:line — check it before debugging prediction quality.

## Conventions

- Smallest change that keeps component boundaries; no new deps.
- Config-driven: add a wrapper + registry entry rather than branching inside the pipeline.
- Verify against the implementation before claiming behavior; say "inferred" when inferring.

## Known defects (audited 2026-09-28, by reading — not yet fixed)

Ranked by impact. `TODO.md` holds the author's own earlier list; this supersedes it where they overlap.

### Tier 1 — caps quality, cheap to fix

1. **`emb_flag: 1` discards the mention.** `.models/xmr4el_base_config.json:73`. Corpus rows are `"mention [SEP] title+abstract"` (`featurization/preprocessor.py:227`); flag 1 TF-IDFs the whole string, so the ~5-token mention is ~2% of a ~250-token vector. Every mention in one abstract gets near-identical features with different gold CUIs. Flag 4 (`featurization/text_encoder.py:205`) exists for exactly this: transformer on the mention side, TF-IDF on the context side. Config-only change.
2. **Small clusters are silently deleted.** `clustering/train.py:37-41` drops clusters with `cnt < min_leaf_size`; their labels get an all-zero `C_node` row, so they can never be proposed (`xmr/base.py:474`) and never reach the next layer (`xmr/base.py:817`). Unlogged recall ceiling. Also `max_leaf_size` is computed at `clustering/train.py:31` and never used.
3. **Ranker curriculum never runs.** `ranker/train.py:310` — `ranker_models.update(...)` is outside the epoch loop, so `existing` is always `None` and each epoch retrains from scratch; the fixed `RandomState(seed)` makes all epochs pick identical negatives. 3 epochs = 3x cost, 1x effect. Compounded by `E_warm: 3` with `n_epochs=3`, so `epoch <= E_warm` always holds (`ranker/train.py:104`) and the hard-negative phase is dead code.
4. **Most labels get no ranker.** `ranker/train.py:164` skips labels with `n_pos < 2` (the PubTator long tail). At predict, `mdl is None -> r = ones(...)` (`xmr/base.py:556`), collapsing them onto the shared cluster score. Use `cosine(x, z_label)` as the fallback instead of `1.0`.

### Tier 2 — design

5. **Eval measures a cosine retriever, not the hierarchy.** `test/xmr4el/test_evaluate_pipeline.py:119` passes `topk_mode="global"`, taking the `xmr/model.py:417` branch that replaces hierarchy scores with cosine over `Z`. Separately `path_logscore` is built at `xmr/base.py:1156` and never folded into the leaf score. (= TODO.md #1, #2.)
6. **Leaf matcher scores are per-cluster, not per-label.** `xmr/base.py:547` (`m = cluster_scores[q_idx, c]`). With `depth: 2`, `n_clusters: 6`, `cut_half_cluster: true` -> 18 leaves, each still scoring only ~3 clusters, so hundreds of labels share one matcher score and (per #4) have no ranker to break the tie. Fix by going deeper or making leaf `C` an identity.
7. **Exposure bias at every layer.** `prepare_layer` restricts each child to `mention_mask` (`xmr/base.py:826`), so child matchers see only in-cluster positives, never out-of-cluster negatives — yet inference routes everything to them. Miscalibrated probabilities are exactly what the beam prunes on. TFN/MAN negatives currently feed only the ranker (`xmr/base.py:415`).
8. **PIFA is context-polluted.** `xmr/model.py:319` averages `"mention [SEP] context"` embeddings per CUI, so frequent CUIs drift to the corpus centroid and clusters become topical, not semantic — and the cosine reranker scores against that same `Z`. Building `Z` from UMLS synonyms (MRCONSO) is what the `.pubtor` KB layer is for.
9. **Dead compute in `fused_predict`.** `xmr/base.py:270` "SINGLE RANKER SHORTCUT" applies one arbitrary label's ranker to every (mention, label) pair. It runs only at the last layer, feeding a next layer that does not exist, and is still written to `fused_scores.npy`.

### Tier 3 — measurement

10. **Label space == the training file.** `organize_pubtator_output` defines labels from train CUIs; eval deletes any gold CUI not in `initial_labels` (`test/xmr4el/test_evaluate_pipeline.py:106`). Results are an in-vocabulary upper bound; zero-shot linking is unmeasured. Print filtered/total.
11. **One weak metric.** `hit_counts` is binary `gold in cand` (`test/xmr4el/test_evaluate_pipeline.py:131`) — candidate-set recall only. No acc@1, MRR, or recall@k curve, so defects 3/4/5 are currently invisible. Derivable from `score_csr` in ~10 lines.
12. **Benchmarks unused.** `datasets/MedMentions/st21pv`, `BC5CDR`, `BioRED` are unpacked but no script reads them. st21pv is the standard comparison point.

### Suggested order

#3 (one-line dedent) -> #1 (config) -> #2 -> #4 -> Tier 3 metrics (so the rest is measurable) -> #5/#6/#7 -> #8 with the KB layer.

### Memory-safe diagnostics

This machine cannot train at full size. All of these are cheap:
- CUI frequency histogram over the pubtator train file -> fraction of labels clearing `n_pos >= 2`.
- Mentions-per-abstract count -> how many rows defect #1 collapses.
- Train/test CUI overlap -> the ceiling from #10.
- Cluster-size distribution: run the clustering config on `Z` alone (13k x 1500 float32 ~ 78MB), no transformer or matcher — counts what #2 deletes.
- End-to-end smoke run at `-ds_len 2000`.

## PECOS comparison (the thesis target)

The claim is XMR4EL vs **PECOS XR-Linear**, head to head. PECOS is currently **not in this repo** — it appears only as prose in `TODO.md`. Before any result is publishable:

**Baseline setup.** `pip install libpecos`, then train `pecos.xmc.xlinear.XLinearModel` on the *same* artifacts XModel builds: `self.X` (instance features), `self.Y` (binarized labels), `self.Z` (PIFA). Same `n_clusters`, same depth, same `min_leaf_size`, same beam size, same metric, same train/test split. Any of these differing makes the delta uninterpretable.

**Equivalence conditions.** A fair comparison requires XMR4EL to not silently deviate from XR-Linear semantics. Defects #5, #6, #7 are all such deviations (score not taken from the hierarchy; leaf scores cluster-level instead of label-level; matcher layers trained without out-of-cluster negatives). Each must be either fixed to match PECOS, or kept as a named design choice with its own ablation row. `TODO.md:34` is the author's note on the same point.

**Ablation ladder.** The per-label ranker, the flag-4 features and the fused scoring are three confounded variables. Report in this order so each is attributable:

1. PECOS XR-Linear, default features
2. PECOS XR-Linear, XMR4EL features (flag 4)
3. XMR4EL tree + matcher only, ranker disabled (`ranker_every_layer: false`, depth-1 leaf ranker off)
4. XMR4EL full (+ per-label ranker, + curriculum negatives)

**Metric alignment.** Each mention carries exactly one gold CUI (`featurization/preprocessor.py:113` -> one-hot `Y` rows), so this is extreme multi-*class*, not multi-label. PECOS papers report precision@k and propensity-scored metrics, which are the wrong target here. Report **acc@1, MRR, recall@k** and say explicitly in the write-up that the single-gold setting is why. Do not silently reuse PECOS's metric suite.

**Scale.** At the ~13k-label disease set neither system is stressed and mature PECOS will likely win outright. `data_datasets/data/train/chemical/labels.txt` carries ~290k labels; full UMLS via the `.pubtor` layer is larger still. The XMR argument only becomes interesting where routing quality and label-embedding quality decide the outcome, so run the headline comparison at the largest label space that fits in memory.

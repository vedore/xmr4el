# Next session — pick up here

State as of 2026-09-29, **session 3**. Branch `repeat`, all changes uncommitted.

Full defect descriptions live in `CLAUDE.md` ("Known defects" + "PECOS comparison"). `results.md`
is the running ablation table. `TODO.md` is the oldest list — superseded where they overlap.

---

## Where we actually are

**Phase 0 done. Phase 1 done (all four items). Nothing is committed yet.**

Working notes for whoever picks this up:

- **The user runs every train/eval themselves.** Do not execute the pipeline. Write the code, then
  hand over the exact command and wait for pasted output. Keep `-ds_len` small (500–1000).
- The venv is `.venv/bin/python` (Python 3.12, uv-managed). `python3` alone has no numpy.
- Self-checks are cheap and safe to run directly:
  `.venv/bin/python -m xmr4el.ranker.train`, `.venv/bin/python -m xmr4el.clustering.train`,
  `.venv/bin/python test/xmr4el/test_evaluate_pipeline.py -selfcheck`.

### Done this session

| Item | Where | Note |
|---|---|---|
| Phase 0 metrics | `test/xmr4el/test_evaluate_pipeline.py` | `gold_rank()` + acc@1 / MRR / recall@k, in-vocab kept/total, `_selfcheck` |
| Phase 0 `topk_mode` | same file | `"global"` → `"per_leaf"`; scores now come from the hierarchy, not cosine over `Z` |
| Phase 1 #1 ranker curriculum | `xmr4el/ranker/train.py:310`, `:198` | `ranker_models.update` moved inside the epoch loop; RNG is `seed + epoch`; `E_warm` 3→1 in config |
| Phase 1 #2 features | `.models/xmr4el_base_config.json` | `emb_flag` 1 → 4 |
| Phase 1 #3 cluster drop | `xmr4el/clustering/train.py` | orphans reassigned to nearest valid centroid; logs raw/valid/reassigned/sizes; `max_leaf_size` dead computation deleted (param kept) |
| Phase 1 #4 ranker fallback | `xmr4el/xmr/base.py:555,565,567` | `r = ones(...)` → `_cos_fallback(x, z_label)`, mapped to [0,1] |
| sklearn 1.9 drift | config + `classifier_wrapper/classifier_model.py` | matcher `eta0: 0.0` removed (now must be > 0); ranker `class_weight: "balanced"` removed (`partial_fit` rejects it) |
| determinism | `models/cluster_wrapper/clustering_model.py` | `seed: 0` in `balancedkmeans` defaults + `np.random.seed`/`torch.manual_seed` before `fit` |

### Two findings that change the plan

1. **Rows 0–3 of `results.md` are inside the noise band and are not publishable.** Row 3 changed no
   routing input yet candidate recall moved −2.1pt. Cause: `kmeans_pytorch` seeded its centroids
   from the global `np.random` state. Now fixed — runs from here are reproducible. Rows 0–3 cannot
   be regenerated without reverting code, so they are superseded by the matrix below rather than
   re-run.
2. **Defect #2 (dropped small clusters) is inert at this configuration.** The new log says
   `reassigned 0/500` at the root and `0/83` at every child: `balanced=True`
   (`clustering_model.py:759`) makes near-equal clusters, so nothing falls under `min_leaf_size: 5`.
   The fix stays in as XR-Linear equivalence insurance, but it buys no accuracy. Do not spend more
   time on it.

### Immediate next step

The user is running `./test/xmr4el/run_ablation.sh` — a seeded 2x2 over `emb_flag` {1, 4} x
`-ds_len` {500, 1000}, logs in `test/test_data/ablation/`. It ends with a summary block.

**When the output is pasted: write those 4 rows into `results.md` as the real baseline table,
replacing rows 0–3.** Then Phase 2.

Watch for: acc@1 vs the random-ordering floor. With 100 candidates/query, random ordering scores
acc@1 ≈ recall@100 / 100 ≈ 0.0039. Every row so far sits at or below that — **scoring has never
beaten random**. If item #4 (cosine fallback) has not lifted acc@1 clearly above the floor, the
cause is defect #6 (leaf matcher scores are per-cluster, not per-label, `xmr/base.py:547`), and
#6 should be pulled forward from Phase 3 ahead of the PECOS baseline.

---

## Plan

Two deliberate reorderings against the old queue:

1. **Metrics before fixes.** The Tier 1 fixes are unverifiable today — the only metric is binary
   candidate-set recall and eval runs `topk_mode="global"`, which replaces hierarchy scores with
   cosine over `Z`. Fix the ranker and the number would not move, and you could not tell whether
   the fix failed or the metric is blind.
2. **PECOS baseline before the big refactor.** The baseline is the risk, not the reward. If
   XR-Linear on your own `X`/`Y`/`Z` wins outright, that must surface in week 1, because it changes
   what the thesis argues.

Keep `results.md` with one row per change and the exact command that produced it. That table is the
ablation appendix, written as a side effect instead of reconstructed at the end.

### Phase 0 — make it measurable — **DONE**

- acc@1, MRR, recall@k from `score_csr` in `test/xmr4el/test_evaluate_pipeline.py` (~15 lines, no
  extra compute). Keep the existing hit count as recall@candidates.
- Drop `topk_mode="global"` (`test_evaluate_pipeline.py:119`). One argument. Until it goes, every
  number measures a cosine retriever over `Z`, not the hierarchy.
- Print filtered/total from `filter_labels_and_inputs`, so the 57.6% in-vocabulary ceiling appears
  in every run instead of being invisible.
- Write down *now*, while it is fresh: the metric is acc@1 / MRR / recall@k and **not** precision@k,
  because each mention carries exactly one gold CUI (`preprocessor.py:113` → one-hot `Y` rows). An
  examiner will ask why the PECOS metric suite was not reused.

**Exit:** a baseline row in `results.md` from a smoke run (`-ds_len 2000`).

### Phase 1 — Tier 1 bugs — **DONE** (see table above; #2 turned out inert)

In this order; re-run eval after each and record the delta row.

1. **`ranker/train.py:310`** — `ranker_models.update({...})` sits outside the `for epoch` loop, so
   `existing` is always `None` and every epoch retrains from scratch. Move it inside, right after
   `results = Parallel(...)(...)`. One indent level. Then set `E_warm` to 1 or 2 in
   `.models/xmr4el_base_config.json` (currently 3 with `n_epochs=3`, so `epoch <= E_warm` always
   holds and the hard-negative phase is dead code), and derive the negative-sampling RNG per epoch
   instead of from the fixed seed (`ranker/train.py:196`) so the epochs stop drawing identical
   negatives. Currently 3x the cost for 1x the effect.
2. **`.models/xmr4el_base_config.json:73`** — `emb_flag: 1` → `4`. Config only, biggest expected
   single jump. At 46.4 mentions/abstract, flag 1 gives 46 rows near-identical features and
   different gold CUIs.
3. **`clustering/train.py:37-41`** — labels in clusters below `min_leaf_size` get an all-zero
   `C_node` row and become permanently unreachable. **Reassign them to the nearest valid cluster**
   (against the clustering model's centroids, ~4 lines) rather than allowing undersized leaves:
   XR-Linear never drops labels, and dropping them breaks the equivalence Phase 2 depends on. Log
   the reassigned count — that is the missing cluster-size diagnostic. While in the file,
   `max_leaf_size` is computed at line 31 and never used: wire it up or delete it.
4. **`ranker/train.py:164`** — labels with `n_pos < 2` (38.4% of them) get no ranker and tie on the
   shared cluster score at `xmr/base.py:556`. Replace the `r = ones(...)` fallback with
   `cosine(x, z_label)`.

Leave one runnable assert-based check behind for #1 and #3 — both degrade silently rather than
crashing.

**Exit:** four `results.md` rows, each attributable to one change.

### Phase 2 — PECOS baseline — **NEXT**

`pip install libpecos`, train `pecos.xmc.xlinear.XLinearModel` on the same artifacts XModel already
persists: `self.X`, `self.Y`, `self.Z`. Same `n_clusters`, depth, `min_leaf_size`, beam size,
metric, split. Any of these differing makes the delta uninterpretable.

Gives rows 1–2 of the ablation ladder in `CLAUDE.md`:
1. PECOS XR-Linear, default features
2. PECOS XR-Linear, XMR4EL features (flag 4)

**Exit:** a head-to-head number on st21pv dev. This is the go/no-go for the thesis framing.

### Phase 3 — converge on XLinear semantics

Not "fixes" — **deletions**. Each item removes an accidental deviation, so baseline and system end
up differing by exactly one component (the per-label ranker), which is the cleanest comparison
available and the easiest to defend in the write-up. `TODO.md:34` is the author's note on the same
point.

- Fold `path_logscore` into the final leaf score (`xmr/base.py:1156`); it is built and never used.
- Make the leaf matcher label-level rather than cluster-level (`xmr/base.py:547`).
- Add out-of-cluster negatives to the matcher layers, not just the ranker (`xmr/base.py:826`).
- Delete the dead `fused_predict` single-ranker shortcut (`xmr/base.py:270`).

**Exit:** rows 3–4 of the ladder (tree+matcher with ranker disabled; full system).

### Phase 4 — the novel contribution

- Build `Z` from UMLS MRCONSO synonyms via `.pubtor` instead of PIFA over mention+context
  (`xmr/model.py:319`). This is what decouples the label space from the training file — i.e. what
  turns this into real EL rather than "rank the 18.5k CUIs I have seen", and what makes the 42.4%
  zero-shot slice scorable instead of deleted.
- Re-run the ladder with the zero-shot slice **included**, reported separately. A zero-shot column
  is a result no amount of tuning on the in-vocabulary set can produce.
- Only then consider `MedMentions/full` for scale, or the MeSH-mapped secondary sets.

---

## Open decisions

- **Per-label ranker: keep and test.** Settled. It is the one deliberate addition on top of
  XLinear semantics, and the row 3 vs row 4 delta is what measures it. This is why defects #3 and
  #4 must be fixed properly rather than worked around.
- **Headline label space: st21pv (18.5k).** Settled, replacing the earlier "disease 13k vs chemical
  290k" question — those came from `data_datasets/`, which is no longer the plan. st21pv buys an
  external comparison point, which is worth more than raw label count. If a scaling argument is
  needed later it comes from `MedMentions/full`, not from a second corpus.
- **Defect #2 fix: reassign to nearest valid cluster.** Settled, see Phase 1 #3.
- Still open: whether `path_logscore` folding happens as sum-of-logs or a learned weight — decide
  with a number in Phase 3, not in advance.

## Housekeeping

- `.gitignore` now covers `/data_datasets` and `/data_datasets.zip`. Done.
- `LICENSE` and `CONTRIBUTING.md` are empty files.
- `README.md` refers to `_test/`; the directory is `test/`.

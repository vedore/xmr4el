# Next session — pick up here

State as of 2026-09-29. Code still unchanged; the two sessions so far were read-only analysis plus
this plan. Branch `repeat`.

Full defect descriptions live in `CLAUDE.md` ("Known defects" + "PECOS comparison"). This file is
the work queue and the state you would otherwise re-derive. `TODO.md` is the older list —
superseded where they overlap, still correct on its own items.

---

## The one thing to remember

The thesis is **not** wrong. The premise (EL as XMR, tree-routed candidates, PIFA hierarchy) is
sound. What is broken is the implementation and the measurement — which together mean the thesis
is currently **untested**, not disproved. Three reasons:

1. There is no PECOS baseline in the repo at all.
2. The places where XMR4EL diverges from XR-Linear semantics are accidents, not design choices.
3. The only metric is candidate-set recall, which cannot see the ranker working or failing.

---

## Datasets — decided

**Everything runs on `datasets/`, not `data_datasets/`.** MedMentions ST21pv is the headline; it is
the standard biomedical EL benchmark, has published comparison numbers, and ships its own splits.

```
datasets/MedMentions/st21pv/data/corpus_pubtator_train.txt   2,635 docs  122,241 mentions  18,520 CUIs
datasets/MedMentions/st21pv/data/corpus_pubtator_dev.txt       878 docs   40,884 mentions   8,643 CUIs
datasets/MedMentions/st21pv/data/corpus_pubtator_test.txt      879 docs   40,157 mentions   8,457 CUIs
```

Use **dev** for all iteration; touch **test** only for the final table.

Secondary sets, not drop-in:
- `datasets/MedMentions/full/` — same corpus, no CUI filtering, ~3x the label space. The scaling
  run, if one is needed. Splits are PMID lists, not separate files.
- `datasets/BC5CDR`, `datasets/BioRED`, `datasets/ncbi_disease` — label spaces are MeSH / NCBI-Gene
  IDs, **not CUIs** (`BioRED/Train.PubTator` col 6 is e.g. `D003924`). Each needs a mapping through
  MRCONSO before it can share the pipeline. Out of scope until Phase 4.

**Loader caveat:** st21pv col 6 is `UMLS:C1519254`, prefixed. `preprocessor.py:227` takes
`parts[5]` verbatim, so labels become `"UMLS:Cxxxxxxx"`. Harmless for train/test string matching
(both sides carry the prefix) but it will not join to the `.pubtor` UMLS KB. Strip the prefix in
the loader when Phase 4 starts, or earlier if it is cheaper to do once.

---

## Diagnostics — three of four already done

Measured on st21pv train/test with shell one-liners. No script needed; the fourth is folded into
fix #2 below.

| Diagnostic | Result | Means |
|---|---|---|
| Labels clearing `n_pos >= 2` | **11,418 / 18,520 (61.6%)** | 38.4% of labels get **no ranker** and fall back to `r = ones(...)` — defect #4 |
| Mentions per abstract | **46.4** | Defect #1 collapses ~46 rows per abstract into near-identical TF-IDF vectors with different gold CUIs |
| Test CUIs present in train | **4,867 / 8,457 (57.6%)** | **42.4% of test CUIs are zero-shot and currently deleted by eval** (`test_evaluate_pipeline.py:106`) — defect #10 |
| Cluster-size distribution | not run | Folded into fix #2: the fix has to log the dropped count anyway |

The zero-shot number is the headline finding. Nearly half the test set is being thrown away before
scoring, and the result is reported as if it were the whole set.

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

### Phase 0 — make it measurable

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

### Phase 1 — Tier 1 bugs

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

### Phase 2 — PECOS baseline

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

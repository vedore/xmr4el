# Results archive (invalid)

Every dev number here predates the label-mapping and embedding-row-order fixes and is **invalid**.
Kept only as history of what was tried. Current results: `docs/results.md`.

## Session 5: every dev number below is invalid (label order)

Model label index `j` is `sorted(labels)[j]` (`MultiLabelBinarizer.classes_`), but eval mapped names with
`initial_labels` in input order: 0 of 500 positions matched. Rows 0-11, including the post-fix rows 8-11,
scored dev against a permuted label list. Fixed in `xmr4el/xmr/model.py` (applies to existing trees on
load). Train-side routing figures (4.79x matcher, 3.81x cosine) used the internal `Y` and still stand.
See `STATUS.md` session 5.

## ROOT CAUSE FOUND (session 4): layer models were loaded out of training order

**`HierarchicaMLModel.load` appended each layer's models in raw `plistdir()` filesystem order while
`child_index_map` indexes them by training order.** Layers themselves were sorted
(`xmr4el/xmr/base.py:813`); the models *inside* a layer never were. So the beam sent every query to a
child holding a **different cluster's labels**.

Measured on the `flag4_ds500` tree before the fix: `child_index_map[0][0]` is the identity
`{0:0,...,5:5}`, yet 5 of 6 root clusters mapped to a child whose label set was **entirely disjoint**
(overlap 0). Exactly 1 of 6 lined up -- which is why routing measured at chance.

Root matcher vs full traversal, same 2000 stored *training* rows, `K=6`:

| beam | chance | root matcher | before fix | after fix |
|---|---|---|---|---|
| 1 | 0.167 | 0.797 (**4.79x**) | 0.152 (0.91x) | **0.797 (4.79x)** |
| 2 | 0.333 | 0.933 (2.80x) | 0.295 (0.89x) | **0.933 (2.80x)** |
| 3 | 0.500 | 0.976 (1.95x) | 0.501 (1.00x) | **0.976 (1.95x)** |

The root matcher was always a good router -- 80% top-1 cluster accuracy. The traversal discarded it.

Fixed at `xmr4el/xmr/base.py:820` (sort by the `ml_<n>` index `save()` writes at line 778), with the
invariant asserted at load time so it cannot regress silently.

### Every number in this file predates the fix and is invalid

Evaluation always goes through `XModel.load`, so the permutation affected **all** of it: rows 0-7,
both beam sweeps, the "routing is at chance" correction and the "`emb_flag` is the routing
bottleneck" table. The flag1-vs-flag4 delta may well survive -- flag 4 was above chance *despite* a
1-in-6 router -- but it has to be re-measured, not assumed. Nothing here is citable until the tables
are regenerated.

The three diagnostics that remain valid are the ones that never touched the traversal: the encoder
is exact (`cos = 1.0000` re-encoding stored rows, verified only where row order is known), the
matcher routes train at 4.79x chance, and cosine-over-`Z` routes at 3.81x -- so `Z` and the
clustering carry real routing signal and **defect #8 is not the bottleneck it looked like**.

Note `_X`, `_Y` and `_training_texts` are in three different orders: `X[i]` pairs with `Y[i]`, but
re-encoded `_training_texts[i]` matches `X[i]` at only ~0.36 cosine and `Y.argmax` never equals the
`_training_texts` group index. Any diagnostic pairing texts to rows is wrong.

## Runs

### Post-fix baseline (valid): eval-only on the 2026-09-29 trees

Eval-only re-runs with the fixed loader (`xmr4el/xmr/base.py:820`), 2026-09-30. The trees predate #6 (cluster-level leaf); their saved configs (`test/test_data/ablation/config_flag*_ds500.json`) already show
the score-domain fix (ranker `log_loss`, `neg_mult 5`), unverified against the tree itself. st21pv dev, `-ds_len 500` (500 labels), in-vocab 7530/40884 (347 unique gold
CUIs), `-beam_size 5`. Command: the eval loop in `STATUS.md` "Immediate next step (session 4)".

| # | Tree | `-topk` | cand/q | R@cand | random R@cand | ratio | acc@1 | MRR | R@5 | R@20 | R@50 | R@100 |
|---|------|---------|--------|--------|---------------|-------|-------|-----|-----|------|------|-------|
| 8 | flag1 `15-16-12` | 20 | 100 | 0.3934 | 0.204 (100 of 417) | 1.93x | 0.0015 | 0.0145 | 0.0042 | 0.0952 | 0.2040 | 0.3934 |
| 9 | flag1 `15-16-12` | 0 | 417.0 | 0.8509 | 0.833 (5/6 clusters) | **1.02x** | 0.0003 | 0.0100 | 0.0037 | 0.0463 | 0.0853 | 0.2016 |
| 10 | flag4 `15-58-50` | 20 | 100 | 0.2641 | 0.176 (100 of 417) | 1.50x | 0.0023 | 0.0120 | 0.0084 | 0.0622 | 0.1537 | 0.2641 |
| 11 | flag4 `15-58-50` | 0 | 416.9 | 0.7341 | 0.833 (5/6 clusters) | **0.88x** | 0.0013 | 0.0122 | 0.0092 | 0.0507 | 0.1077 | 0.1910 |

Chance at `-topk 0` is exactly 5/6 whatever the cluster sizes are: the beam drops 1 of 6 root clusters, and the leaf layer
does not prune (hardcoded `beam_size=100`, `base.py:1082`).

Reading:
- **Root routing on dev is at chance for flag 1 (1.02x) and below chance for flag 4 (0.88x).** Flag 4 misses 2002 of 7530
  queries against 1255 expected at chance. On *train* rows the same flag-4 root matcher scores 4.79x chance, so the
  matcher does not generalise from train to dev. The inferred cause is memorising abstract context: 46 mentions share one
  abstract's TF-IDF, and dev abstracts are unseen. This is inferred, not verified.
- **The pre-fix beam sweep was measuring noise.** Before the fix, child `c` held a different cluster's labels, so every
  candidate set was 5 effectively random clusters. That explains why both flags sat at ~0.83-0.87 then. Its conclusion
  that "flag 4 routes above chance" is **reversed** here: flag 4 is worse than flag 1 on every candidate metric.
- **The per-leaf cut carries signal and the global order does not.** Leaf-wise top-20 reaches 1.5-1.9x random. A global
  top-100 over the same 417 candidates is ~1.0x. Leaf scores are not comparable across leaves (`path_logscore` is never
  folded into the leaf score, defect #5).
- **Ordering is at or below random in all four rows.** For example, flag 1 acc@1 is 0.0015 against a random line of 0.0039.
  So the score-domain fix alone did not produce ordering signal; #6 is not measured yet.

### Baseline: Phase 1 complete, seeded (rows 4-7)

First reproducible rows. Clustering is seeded from `clustering_model.py`, so these rerun
byte-identical. Produced by `./test/xmr4el/run_ablation.sh` with the config **as of Phase 1 exit**:
ranker `loss: hinge`, `neg_mult: 45`, `E_warm: 1`, `emb_flag` set per row. `-beam_size 5 -topk 20`,
st21pv dev, 100 candidates/query.

| # | Change | Train | in-vocab | acc@1 | MRR | R@5 | R@20 | R@50 | R@100 | R@cand |
|---|--------|-------|----------|-------|-----|-----|------|------|-------|--------|
| 4 | `emb_flag 1` | `-ds_len 500` (500 labels) | 7530/40884 (18.4%) | 0.0011 | 0.0125 | 0.0094 | 0.0462 | 0.1473 | 0.3588 | 0.3588 |
| 5 | `emb_flag 4` | `-ds_len 500` | same | 0.0000 | 0.0131 | 0.0092 | 0.0493 | 0.1166 | 0.3892 | 0.3892 |
| 6 | `emb_flag 1` | `-ds_len 1000` (1000 labels) | 11661/40884 (28.5%) | 0.0026 | 0.0147 | 0.0213 | 0.0385 | 0.0873 | 0.1907 | 0.1907 |
| 7 | `emb_flag 4` | `-ds_len 1000` | same | 0.0003 | 0.0126 | 0.0069 | 0.0914 | 0.1691 | 0.2763 | 0.2763 |

**Reading: routing works, scoring is exactly uniform noise.** Against the random-ordering
prediction `R@cand x k/100`:

| row | R@cand | k=5 pred/act | k=20 pred/act | k=50 pred/act |
|---|---|---|---|---|
| 4 | .359 | .018 / .009 | .072 / .046 | .179 / .147 |
| 5 | .389 | .019 / .009 | .078 / .049 | .195 / .117 |
| 6 | .191 | .010 / .021 | .038 / .039 | .095 / .087 |
| 7 | .276 | .014 / .007 | .055 / .091 | .138 / .169 |

Actual tracks the random line at every k, and at `-ds_len 500` sits *below* it. The ranker
contributes nothing to ordering; all four rows' signal is routing recall alone. **acc@1 and MRR are
not interpretable in any row above** and must not be reported as results.

Mechanism, confirmed by reading (`xmr4el/xmr/base.py:547-578`): ranker `loss: hinge` selects
`is_hinge` -> `r = expit(mdl.decision_function(...))`. At 45 negatives per positive with no
reweighting every margin is strongly negative, so `expit -> ~0`, then `clip(r, eps, 1.0)` floors
every ranker-backed label at `eps`. Meanwhile the 38.4% of labels with no ranker take
`_cos_fallback`, which returns `[0,1]`-mapped cosine (~0.5) — so the untrained long tail
systematically outranks every trained label. `m = cluster_scores[q_idx, c]` (`base.py:553`) is
cluster-constant, so within a cluster nothing breaks the tie but CSR index order. This is
STATUS appendix #4 ("predict_proba, decision_function, expit and fixed-alpha fusion are mixed without
calibration") observed end to end.

**`-topk 20` is per-leaf, not global.** `beam_size 5 x topk 20` is the exact `candidates/query =
100.0`, and `recall@candidates` is measured on `leaf_global_labels` *after* that cut
(`base.py:1090`), i.e. through the broken scores. At `-ds_len 500` (~14 labels/leaf -> 70 < 100) no
cut happens, so rows 4-5's `R@cand` is the true beam ceiling; rows 6-7 (~140 candidates) are
depressed by noise-driven truncation, which is why `R@cand` *falls* from ds500 to ds1000.

### Correction (session 4): the beam does not route

The `-topk 0` ceiling pass at `-ds_len 1000` returns **834 candidates/query out of 1000 labels**
and `recall@candidates = 0.814`. Chance for a set that size is 834/1000 = **0.834**. Routing is at
chance — the hierarchy contributes nothing.

The arithmetic is deterministic, not noise: `n_clusters: 6` at the root and `-beam_size 5` keeps
5/6 of the label space, and the leaf call uses a hardcoded `beam_size=100` (`xmr4el/xmr/base.py:1082`)
so nothing is pruned below the root. `5/6 x 1000 = 833`. **The tree prunes exactly one cluster.**

This inverts the reading above. `recall@candidates` at `-topk 20` is 0.191 from 100 candidates;
selecting those 100 from the 834 at random would give `100/834 x 0.814 = 0.098`. So the ~1.9x is
produced entirely by the **fused score** truncating 834 -> 100, not by the hierarchy. "Routing
works, scoring is noise" is backwards: **routing is at chance and the fused score carries weak
signal** (~2x at k=100) that decays to nothing by k=1.

Consequence for the plan: at `beam 5 / n_clusters 6` no ablation can measure routing, because there
is barely any routing to measure. Routing quality at chance points at defect #8 (PIFA over
`"mention [SEP] context"`, `xmr/model.py:319`, makes clusters topical rather than semantic), not at
the ranker.

### Ties are gone and it did not help

Same log: `distinct scores/query: 100.0 of 100.0`. The tie signature that motivated defect #6 is
absent — every candidate now gets its own score — yet `acc@1 = 0.0017` against a random line of
0.0019, and `recall@50 = 0.075` against 0.095. **Breaking the ties did not create signal.** Whatever
ordering the fused score produces is distinct-but-arbitrary, so #6 was not the bottleneck it was
ranked as.

*Caveat: this run spanned live edits to `base.py` and `test_evaluate_pipeline.py`, so its rows are
not attributable and are not tabled. The two findings above survive it — one is config arithmetic,
the other is a qualitative presence/absence.*

**`emb_flag 4` improves candidate selection** (not routing -- see the correction above). Normalized
by chance (`100 / n_labels`): ds1000 flag 4 is 2.8x chance vs flag 1's 1.9x; ds500 1.9x vs 1.8x. The ds500 pair is a fair comparison (no truncation,
same label space). ds500 vs ds1000 is **not** comparable — different label spaces and different
in-vocab filters (18.4% vs 28.5%).

### Routing measured properly: `emb_flag` is the routing bottleneck (defect #1 confirmed)

Beam sweep at `-topk 0` on the two `-ds_len 500` trees, same 500-label space, both trained **before**
the #6 edit landed, differing only in `emb_flag`. With `depth: 2` the candidate set is exactly the
union of the beamed root clusters' labels, so this isolates the **root matcher** — no leaf scoring,
no ranker, immune to everything else changed this session. Chance = `nnz / 500`.

| beam | flag1 meas | chance | ratio | flag4 meas | chance | ratio | delta |
|---|---|---|---|---|---|---|---|
| 1 | 0.1420 | 0.1675 | **0.85** | 0.1452 | 0.1664 | **0.87** | +0.3pt |
| 2 | 0.2818 | 0.3349 | **0.84** | 0.3645 | 0.3326 | **1.10** | +8.3pt |
| 3 | 0.4576 | 0.5016 | **0.91** | 0.5513 | 0.4989 | **1.11** | +9.4pt |
| 5 | 0.8250 | 0.8340 | **0.99** | 0.8668 | 0.8319 | **1.04** | +4.2pt |

**`emb_flag 1` routes below chance at every beam.** Defect #1 confirmed end to end: flag 1 TF-IDFs
`"mention [SEP] title+abstract"` whole (`featurization/preprocessor.py:227`), so the ~5-token mention
is ~2% of a ~250-token vector, every mention in an abstract gets near-identical features with
different gold CUIs, and the root matcher cannot route them. The `-ds_len 1000` flag1 tree gives the
same picture (0.85-0.99).

**`emb_flag 4` crosses above chance** (1.10x at beam 2, 1.11x at beam 3), worth +8 to +9pt of
candidate recall at the selective beams that matter. This is the first single-variable, same-code,
same-label-space delta in the project and the first result that is safely attributable.

**But 1.1x chance is a weak router, and it caps everything downstream.** A hierarchy that concentrates
gold only 10% better than random cluster selection cannot support acc@1, no matter how good the leaf
scoring is — which explains why the score-domain fix and #6 both landed on the random line. Features
were necessary but are not sufficient; defect #8 (PIFA over `"mention [SEP] context"`,
`xmr/model.py:319`, giving topical rather than semantic clusters) is **not** exonerated.

**Also: `beam_size: 5` at `n_clusters: 6` is not a search, it is 83% of the label space.** Every
`recall@candidates` figure in rows 0-7 was measured at that setting. The meaningful operating points
are beam 1-2. Note beam 1 is *below* chance even for flag 4 (0.87) while beam 2-3 are above it: the
matcher's top-1 cluster pick is anti-informative while its top-2/3 are informative, which is a
calibration symptom worth a look if routing becomes the focus.

### Superseded: rows 0-3 (unseeded clustering)

Kept for traceability only. `kmeans_pytorch` seeded centroids from the global `np.random` state, so
the tree differed between runs and the +2.8 / +1.6 / -2.1 pt deltas below are inside a >= 2pt noise
band. Not regenerable without reverting code; superseded by rows 4-7. Defect #2 (dropped small
clusters) was also found inert here: `reassigned 0/500` at every node, because `balanced=True`
(`clustering_model.py:759`) never produces a cluster under `min_leaf_size: 5`.

| # | Change | Train | in-vocab | acc@1 | MRR | R@5 | R@20 | R@100 | R@cand |
|---|--------|-------|----------|-------|-----|-----|------|-------|--------|
| 0 | Phase 0 baseline (metrics added, `topk_mode` back to `per_leaf`) | `-ds_len 500` (500 labels) | 7530/40884 (18.4%) | 0.0005 | 0.0125 | 0.0028 | 0.0660 | 0.3441 | 0.3441 |
| 1 | Phase 1 #1: ranker warm-starts across epochs, per-epoch negatives, `E_warm` 3→1 | `-ds_len 500` | same | 0.0001 | 0.0132 | 0.0066 | 0.0655 | 0.3718 | 0.3718 |
| 2 | Phase 1 #2: `emb_flag` 1→4 (transformer on mention, TF-IDF on context) | `-ds_len 500` | same | 0.0000 | 0.0110 | 0.0033 | 0.0398 | 0.3875 | 0.3875 |
| 3 | Phase 1 #3: undersized clusters reassigned instead of dropped | `-ds_len 500` | same | 0.0017 | 0.0102 | 0.0041 | 0.0278 | 0.3667 | 0.3667 |

Run 0, `xmodel_2026-09-29_11-12-59`, st21pv dev, `-beam_size 5 -topk 20`, 100 candidates/query
(5 leaves x topk 20). Two config fixes were needed first to make anything run on sklearn 1.9:
matcher `eta0: 0.0` removed (now required `> 0`) and ranker `class_weight: "balanced"` removed
(`partial_fit` rejects it). The note originally written here — that `neg_mult: 45` "already fixes
the pos:neg ratio" — is **wrong**: `N_neg = min(neg_mult * n_pos, n_neg_avail)`
(`xmr4el/ranker/train.py:91`), so 45 *creates* a 45:1 imbalance with nothing compensating for it.
That is half the cause of the dead ranker described above.

**Reading of row 0.** `recall@100 == recall@candidates` — every candidate the tree proposes is
scored, nothing is lost after retrieval. But acc@1 is 0.05% while 34% of gold labels are *in* the
candidate set: ranking inside the candidate set is no better than random (random would be
0.34/100 = 0.34%; it scores 0.05%, i.e. slightly worse than random). That is the signature of
defects #4 and #6 — most labels have no ranker and fall back to `r = ones(...)`, and the leaf
matcher score is per-cluster, so hundreds of labels share one score and tie. Phase 1 targets
exactly this. The 18.4% in-vocabulary figure is an artefact of the 500-label smoke run, not the
real ceiling (57.6% at full train size).

**Reading of row 1.** Candidate recall +2.8pt (0.344 → 0.372) — the ranker feeds the fused score
that routing prunes on (`ranker_every_layer: true`), so a genuinely-trained ranker improves
retrieval. Ordering is unchanged: MRR +0.0007, acc@1 moves from 4 hits to 1 hit out of 7530, which
is noise. As expected — item #1 only makes the rankers that exist better, and per the 61.6%
diagnostic, 38.4% of labels have no ranker at all (item #4).

**Reading of row 2.** Candidate recall +1.6pt (0.372 → 0.388); feature width 1500 → 2268. Ordering
metrics move down slightly (MRR 0.0132 → 0.0110, acc@1 to 0), but every ordering number so far is
within noise of random — with 100 candidates, random ordering scores acc@1 ≈ R@100/100 ≈ 0.0039 and
all three rows sit *below* that. Retrieval is improving monotonically, scoring is not a signal yet.
Nothing can be concluded about features until items #4 and #6 make the score informative.

**Reading of row 3 — defect #2 is inert at this configuration, and rows 0-3 are inside the noise
band.** The new log says `reassigned 0/500` at the root and `0/83` at every child: balanced k-means
(`balanced=True`, `clustering_model.py:759`) produces near-equal clusters by construction (sizes
`[85,83,83,83,83,83]` then `[18,13,13,13,13,13]`), so nothing ever falls under `min_leaf_size: 5`.
The pruning branch only fires when `n_points / n_clusters < min_leaf_size`, which at the full 18.5k
label space will not happen either. The fix stays in as insurance (and as an XR-Linear equivalence
requirement — XR-Linear never drops labels) but it buys nothing measurable here.

More important: row 3 changed **no** routing input, yet candidate recall moved -2.1pt
(0.388 → 0.367). That bounds run-to-run noise at >= 2pt, which is the same order as the +2.8 and
+1.6 attributed to rows 1 and 2. **Those two deltas are not safely attributable.** Cause found:
`kmeans_pytorch` seeds its initial centroids from the global `np.random` state, so the whole tree
differed between runs. Fixed (`clustering_model.py`, `seed: 0` in the `balancedkmeans` defaults,
`np.random.seed` + `torch.manual_seed` before `fit`); from row 4 on, reruns are byte-identical and
deltas mean something. Rows 0-3 should be regenerated before they go in the write-up.


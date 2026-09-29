# Results

One row per change, with the exact command that produced it. This table is the ablation appendix.

## Metric choice

Each mention carries exactly one gold CUI (`featurization/preprocessor.py:113` → one-hot `Y` rows),
so this is extreme multi-**class**, not multi-label. Reported metrics are **acc@1, MRR, recall@k**.
PECOS's precision@k / propensity-scored suite is **not** reused: with a single gold label,
precision@k is just recall@k / k and carries no extra information.

`recall@candidates` is the old binary `gold in cand` hit count, kept so earlier numbers stay
comparable. `In-vocabulary mentions` is the ceiling from defect #10 — mentions whose gold CUI is
absent from the training label space are deleted before scoring
(`test/xmr4el/test_evaluate_pipeline.py`), so every row below is an in-vocabulary upper bound.

## Runs

| # | Change | Train | in-vocab | acc@1 | MRR | R@5 | R@20 | R@100 | R@cand |
|---|--------|-------|----------|-------|-----|-----|------|-------|--------|
| 0 | Phase 0 baseline (metrics added, `topk_mode` back to `per_leaf`) | `-ds_len 500` (500 labels) | 7530/40884 (18.4%) | 0.0005 | 0.0125 | 0.0028 | 0.0660 | 0.3441 | 0.3441 |
| 1 | Phase 1 #1: ranker warm-starts across epochs, per-epoch negatives, `E_warm` 3→1 | `-ds_len 500` | same | 0.0001 | 0.0132 | 0.0066 | 0.0655 | 0.3718 | 0.3718 |
| 2 | Phase 1 #2: `emb_flag` 1→4 (transformer on mention, TF-IDF on context) | `-ds_len 500` | same | 0.0000 | 0.0110 | 0.0033 | 0.0398 | 0.3875 | 0.3875 |
| 3 | Phase 1 #3: undersized clusters reassigned instead of dropped | `-ds_len 500` | same | 0.0017 | 0.0102 | 0.0041 | 0.0278 | 0.3667 | 0.3667 |

Run 0, `xmodel_2026-09-29_11-12-59`, st21pv dev, `-beam_size 5 -topk 20`, 100 candidates/query
(5 leaves x topk 20). Two config fixes were needed first to make anything run on sklearn 1.9:
matcher `eta0: 0.0` removed (now required `> 0`) and ranker `class_weight: "balanced"` removed
(`partial_fit` rejects it; `neg_mult: 45` already fixes the pos:neg ratio).

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

## Commands

```bash
./test/xmr4el/run_ablation.sh    # seeded 2x2: emb_flag {1,4} x -ds_len {500,1000}
```

```bash
.venv/bin/python test/xmr4el/test_train_pipeline.py \
  -train_path datasets/MedMentions/st21pv/data/corpus_pubtator_train.txt \
  -model_config .models/xmr4el_base_config.json -ds_len 500

.venv/bin/python test/xmr4el/test_evaluate_pipeline.py \
  -xmodel_path test/test_data/saved_trees/<run> \
  -test_path datasets/MedMentions/st21pv/data/corpus_pubtator_dev.txt \
  -beam_size 5 -topk 20

.venv/bin/python test/xmr4el/test_evaluate_pipeline.py -selfcheck   # asserts on gold_rank
.venv/bin/python -m xmr4el.ranker.train        # asserts epochs warm-start + draw new negatives
.venv/bin/python -m xmr4el.clustering.train    # asserts no label is dropped from C_node
```

# Status and next steps

History: `docs/status_log.md`. Valid results: `docs/results.md`.
Implemented behavior, synthetic checks, pipeline observations, PECOS notes: `docs/pipeline.md`.
Commands: `docs/results.md` § Commands.

## Resume here

Last updated 2026-10-09 (R done; plan E written: E1 next).

**NEXT: user decides the order (E3 / run wrapper / E4-E5, see "Open" in plan E).** Reranker dropped.

**E2 result (2026-10-09, `e2_dev_eval.log`, reranker `..._13-27-32`, 5 folds, 2 epochs, 2720 s = +1617 s for the
fold trees): rejected.** `not seen 1` w 0.25 0.9080 vs R2b 0.9108; unseen 0.6967 vs 0.7009; seen >1 0.8113 vs
0.8491. Epoch 2 loss still 0.036: the model memorizes the 4.2k train (query, gold) pairs whatever the negatives.
Of {R, c1, c2} R stays best on dev; no new test run. Across the three, unseen at w 0.25 spans 0.6946-0.7009
(6 rows): the reranker's unseen gain is ~+20-26 dev rows and +6 on test; the training set (4.2k BC5CDR rows) is
the limit, not the negatives or epochs. E4 retrains R's recipe (in-sample, 2 epochs). `-folds` stays (tested, opt-in).

**E1 result (2026-10-09, `e1_dev_eval.log`, reranker `..._12-35-14`, 1 epoch): rejected.** `not seen 1` w 0.25
0.9064 vs R2b 0.9108; unseen 0.6946 vs 0.7009; seen >1 0.7877 vs 0.8491 (the 2-epoch model learned the seen >1
string conventions, 1 epoch does not). Epoch 2's low loss was not the problem: E2 keeps 2 epochs.
Cost view (user asked vs PECOS): PECOS train 34.6 s on our exported features ~ our hierarchy (31-33 s on the Mac,
127 s on the server); PECOS predict 2.3 s vs tree eval ~15-30 s. The reranker adds ~18 min training (2 epochs) and
~3 min per 4.3k dev rows (~250 pairs/s); scoring only `not seen 1` rows (~27%) would cut that to ~50 s (eval scores
all rows today for the scope table). E2's folds add ~20 min of training only.

**E. Next round (plan 2026-10-09; user: run every open option).** Order keeps each tree change in ONE retrain: a
reranker is tied to its tree, so tree-changing code (E3) lands after the reranker work on the current tree.
Baseline = R2b dev: tree `10-18-51` w 0 0.9020; reranker `..._10-37-32`, `not seen 1` w 0.25 = 0.9108 (unseen 0.7009).
Test is run once per final system only (R3 = 0.9229, level with the tree's hybrid 0.9227; unseen +6/938 = noise).
- E1 (no code): option c1, reranker `-epochs 1` on `10-18-51` (epoch 2 loss 0.03 = memorized train lists). User:
  `train_reranker.py -xmodel_path <10-18-51> -train_path datasets/BC5CDR/disease/train.pubtator -epochs 1`, then
  dev `evaluate.py` with `-train_path train_plus_ctd -reranker_path <new> -rerank_w 0 0.1 0.25 0.5 1 inf`.
  Compare per scope with R2b; key number = unseen at the chosen w.
- E2 (code done 2026-10-09, pytest 43): option c2, out-of-fold candidates. `train_reranker.py -folds N
  -tree_train_path <train_plus_ctd>` -> `rerank.out_of_fold_top_k`: train PMIDs split at random (seed 0) into N
  folds; per fold a tree with the saved tree's config (`XModel.__init__` params read off the loaded tree) on
  tree_train_path minus the fold's docs (CTD pseudo-docs stay in every fold tree; all 500 BC5CDR train PMIDs are in
  train_plus_ctd), fold rows ranked by it, labels mapped to the saved tree's indices by name. Gold missing from the
  fold top K: row kept (gold is always inserted at position 0, negatives = top K-1); the printed "gold in tree top K"
  is now the out-of-fold rate. Reader rows carry `docs` (PMID). Fold trees are not saved. Test
  `test_out_of_fold_top_k_never_uses_own_document` (doc-unique labels never ranked out of fold, ranked in sample).
  Single factor vs R: 2 epochs. User (after E1 finishes, `git pull` first): command in the chat / below.
  `python3 scripts/train_reranker.py -xmodel_path outputs/saved_trees/xmodel_2026-10-09_10-18-51 -train_path
  datasets/BC5CDR/disease/train.pubtator -folds 5 -tree_train_path datasets/BC5CDR/disease/train_plus_ctd.pubtator
  2>&1 | tee outputs/logs/e2_train.log` then dev eval as E1 (log `e2_dev_eval.log`). Cost ~5 x 4 min fold trees
  (SapBERT cache hits) + 18 min reranker + 4 min eval. Check: 5 "Fold f/5" lines, held-out rows ~1/5 of the BC5CDR
  rows each; out-of-fold gold-in-top-10 well below R1's in-sample 1.0000 (dev R@10 0.9724 is the reference).
  Pick the best of {R, c1, c2} on dev; test once only if it beats R2b on dev.
- E3 (code, no runs; tree-changing changes batched): #8 candidate-only kNN products (`scoring.py`; result
  identical: test = full kNN at random rows); #9 no `fused_scores`/`label_embeddings` in saved nodes (format change);
  abbreviations: first an offline analysis of `u3_errors.tsv` `abbrev` rows + the reader's expansion (missed SFs:
  oab, edds, ptld) -> a fix plan for the user to approve -> code.
- E4 (user run): BC5CDR retrain with E3 -> dev eval (vs `10-18-51` w 0: abbreviation effect; #8/#9 are
  result-neutral) -> test once. Tree only (reranker dropped, user 2026-10-09).
- E5 (user run): MedMentions st21pv, generality of the tree vs PECOS (no reranker). New tree (base config; pre-C2
  trees do not load), dev eval, PECOS on the exported features, test once.
- Reranker dropped from the system (user 2026-10-09): +0.5 pt on test (= the tree's hybrid) for ~18 min training and
  ~3 min per 4.3k rows. Reported as a rejected experiment; code (`rerank.py`, `train_reranker.py`, eval scopes) kept.
- Open (user 2026-10-09): order of E3-E5 vs `docs/experiment_management.md` (untracked design report). Claude's
  view: E3 code first (no runs, no overlap), then a minimal run wrapper (one experiment JSON -> train + eval ->
  run dir with resolved config, commit, metrics.json; stdlib only), then E4/E5 through it; the report's phases 2-5
  (scheduler, lifecycle, parallel, TUI, Pydantic) only when a real grid needs them.

**R3 result (2026-10-09, `r3_test_eval.log`, test 4399/4410 rows).** Tree w 0: acc@1 0.9177 / MRR 0.9408 / seen 1
0.9851 / seen >1 0.7184 / unseen 0.7186 / hybrid 0.9227. `not seen 1` w 0.25: **0.9229** / MRR 0.9443 / seen >1 0.8161
(+17 rows) / unseen 0.7249 (+6 rows) / seen 1 unchanged = +23 rows. Smaller than dev (+38, unseen +26): on test the
gain is mostly seen >1 and ends level with the tree's hybrid (0.9229 vs 0.9227); the unseen gain (+6 of 938) is
within noise. Ungated w 0.25: 0.9200, seen 1 -13 rows (the gate holds on test). `docs/results.md` test table updated.


**R2b result (2026-10-09, `r2b_dev_eval.log`, same tree + reranker).** Check ok: every scope's w 0 row = R2 w 0;
`unseen` rows keep seen groups at w 0 and match `all rows` on unseen. acc@1 all / seen 1 / seen >1 / unseen, MRR:
- `not seen 1` w 0.25: **0.9108** / 0.9787 / 0.8491 / 0.7009, MRR 0.9374 (+38 rows vs w 0 0.9020; unseen +26,
  seen >1 +12, seen 1 untouched). Chosen. w 0.1-0.5 all 0.9089-0.9108: a plateau, not a knife edge.
- `all rows` w 0.25: 0.9099 / 0.9774 (-4 rows, above floor 0.9767) / 0.8491 / 0.7009: also passes; the gate adds
  ~4 rows. `unseen` w 0.25: 0.9080. w >= 1 worse in every scope.
- Why the gate is principled, not tuning: on seen-1 strings the tree (0.9787) is already at the tree-or-dict ceiling
  (0.9793), so the reranker can only lose there; the gate uses train strings only (like `hybrid`).
- Selection: 21 configs (3 scopes x 7 w) picked on dev -> test once. Generality not shown: w and scope are BC5CDR
  dev choices; a second dataset (MedMentions) with the same recipe is the real check.
R3 (server, in container): `python3 scripts/evaluate.py -xmodel_path outputs/saved_trees/xmodel_2026-10-09_10-18-51
-test_path datasets/BC5CDR/disease/test.pubtator -train_path datasets/BC5CDR/disease/train_plus_ctd.pubtator
-reranker_path outputs/rerankers/xmodel_2026-10-09_10-18-51_2026-10-09_10-37-32 -rerank_w 0 0.25 2>&1 | tee
outputs/logs/r3_test_eval.log`; report the `not seen 1` w 0.25 row (w 0 row = tree on test).

**R1 + R2 result (2026-10-09, remote CUDA, commit `b92e36c`, tree `xmodel_2026-10-09_10-18-51`, reranker
`..._10-37-32`; logs `r1_tree.log`, `r1_train.log`, `r1_dev_eval.log`).**
- Tree: cache miss (new server), 128 leaves of 91-92 labels, run 326 s. Tree code = C2's (`1789a29..b92e36c` touches
  only rerank/eval files).
- w 0 row: acc@1 0.9020 / MRR 0.9319 / unseen 0.6737 / seen 1 0.9787 / hybrid 0.9048 vs C2 `17-28-02` 0.9045 / 0.9335
  / 0.6821 / 0.9783: -11 rows (unseen -8), more than C2's +2/-3; cause = fresh CUDA encoding + k-means ties (same
  code). This tree's w 0 row is R's baseline (as planned).
- R1: 4236/4236 rows kept, gold in tree top 10 = 1.0000 on train (in-sample kNN puts gold first); loss epoch 1 2.27 ->
  0.45, epoch 2 0.03 (memorizes the train lists); 1103 s at ~79 pairs/s.
- Sweep (acc@1 all / seen 1 / seen >1 / unseen, MRR): w 0.5 0.9055 / 0.9739 / 0.8538 / 0.6915, MRR 0.9347;
  w 1 0.8959; w 2 0.8880; inf 0.8762 / 0.9564 / 0.8538 / 0.6170.
- Pass rule vs w 0: best w 0.5 gains all +15 rows, unseen +17, seen >1 +13 (= dict 0.8538 for every w > 0), but
  seen 1 -15 rows (0.9739 < floor 0.9767): **fail, no R3**. The reranker alone breaks exact-string matches
  (seen 1 0.9564 at inf).
- Follow-ups (user picks; first two are eval-only on the same reranker):
  (a) [user 2026-10-09: do it -> R2b] apply the reranker only to rows whose mention string is unseen in train (seen strings keep the tree; uses
      train strings like `hybrid`). Arithmetic from the w 0.5 groups: all ~0.9090 (+30 rows vs w 0), seen 1 unchanged;
      or also on seen >1 (it gives the dict's 0.8538 there).
  (b) finer w (0.1, 0.25) on all rows: seen 1 loss may shrink faster than the unseen gain.
  (c) retrain: 1 epoch, or out-of-sample negatives (tree top K on held-out folds), against the memorization.

**D. Third audit (2026-10-09; all 9 findings checked against code, all real). D1 + D2 done (pytest 42, selfchecks).**
- D1 (no result change): `dict_to_pubtator.py` refuses `-out` == `-ctd_path` (`os.path.samefile`, as split CLI);
  `pecos_compare.py export` refuses a non-empty `-out` (before loading the tree); `train_reranker.py` requires
  `-k >= 2`, `-epochs`/`-rows_per_batch >= 1` and exits before model load when 0 rows are kept; `CrossEncoderReranker.fit`
  raises on empty / misaligned / K < 2 input; `beam_sweep.py` exits non-zero listing failed beams;
  `JointOvRLogistic.fit` logs a warning when `n_iter_ >= max_iter` (fit unchanged).
- D2: `gold_rank` argsort `kind="stable"` (= `top_k`): base ranks on tied scores now equal the reranker w 0 ranks.
  Exact float ties are rare; recorded rows are expected unchanged (not re-run).
- Regressions: `tests/test_audit_regressions.py` (converter alias, non-empty export dir, failing sweep child,
  max_iter warning), `tests/test_rerank.py` (20-way tie: `gold_rank` == w 0 rank; fit rejects empty / K 1).
- Also fixed: `test_fit_score_save_load` was flaky (unseeded init on MPS; the toy plateaus at 2 tied labels for ~40%
  of inits): test model now on CPU with seed 1 (deterministic).
- Deferred: #2 dense matcher (already deferred, C); #8 candidate-only kNN products (BC5CDR train rows are small;
  do when a large-corpus run needs it); #9 drop `fused_scores`/`label_embeddings` from saved nodes (tree-format
  change -> retrain; bundle with the next planned retrain); pyproject `joblib` line and `docs/full_cuda_comparison.md`
  stale claims (touch with those files). Pickle load = trusted-provenance boundary, no change.

**R code done (committed; pytest 36 + selfchecks).** `xmr4el/rerank.py`: `label_texts` (label j = Y column j:
top 5 distinct `mention_key` strings by count, ties first-seen -> BC5CDR strings before CTD names),
`top_k`, `rerank_order`, `CrossEncoderReranker` (`AutoModelForSequenceClassification`, num_labels 1: SapBERT
pooler + linear; AdamW 2e-5, linear warmup 10%, listwise CE, 4 rows x K per step, dynamic padding, max_length 96).
`scripts/train_reranker.py`: negatives = tree top K (its predict_config, beta 10) minus gold, cut to K-1; rows with
< K-1 candidates dropped (no knn top-up: tree gives ~hundreds per row). `evaluate.py -reranker_path [-rerank_k]
[-rerank_w ...]`: logits once, one MRR + acc@1-per-group row per w (default 0 0.5 1 2 inf); load refuses a reranker
of another tree. Synthetic e2e (tfidf tree, tiny BERT): w 0 row = tree metrics. Tiny-BERT toy needs ~800 steps
without dropout to learn query-label matching (test uses 100 epochs, ~6 s).
User runs on Docker (user 2026-10-09: fresh tree, image rebuild; logs r1_tree / r1_train / r1_dev_eval), run inside a persistent interactive container `xmr4el-shell`:
  train.py (train_plus_ctd, bc5cdr config) -> T = newest tree -> train_reranker.py -> evaluate.py dev with `-reranker_path`.
- Pasted output: tree train = cache miss on a new server is fine; R1 check `gold in tree top 10` ~0.97+ on train and loss falling;
  R2 check w 0 row ~ 0.9045 / MRR 0.9335 / unseen 0.6821 / seen 1 0.9783 within noise (new tree, PIFA k-means ties),
  then apply the R pass rule against the new tree's w 0 row; pick w; R3 test with `-rerank_w <w>` only.

**C2+C3 passed (pytest 33 + 4 selfchecks).** Plain attributes instead of pass-through properties; `NodeInput`,
`prepare_layer` -> `[(c, NodeInput)]`, `tree.train` helpers inlined; shape docstrings; PIFA = `normalize(Y.T @ X)`;
`xmodel.json` next to `xmodel.pkl`. Trees before C2 do not load (`15-38-58` dead). Tree `17-28-02` (`c2_train.log`,
`c2_dev_eval.log`): cache hit, hierarchy 32.7 s, run 59.1 s; dev acc@1 0.9045 / MRR 0.9335 / unseen 0.6821 /
hybrid 0.9066 / @cand 0.9907 vs B1 0.9041 / 0.9335 / 0.6800 / 0.9062 / 0.9914 (+2 / -3 rows: PIFA k-means ties):
within noise, no results.md row. R uses `17-28-02`; R's pass baseline is this tree (w 0 row of the sweep = 0.9045 / unseen 0.6821 / seen 1 0.9783).

**C1 passed (pytest 32 + 4 selfchecks; `c1_dev_eval.log` on `15-38-58` identical to B1, eval 7 s).** As planned, plus: both configs set `max_leaf_size: 200`
(never read by code) -> key deleted; `Clustering.train` lost its `local_to_global_idx` arg (only fed
`cluster_to_labels`); 7 save fallbacks removed (tree.py had one too) and the `.joblib`/`.pkl` load branches in
`HierarchicalMLModel.load`; load asserts -> `FileNotFoundError` (missing file) / `ValueError` (content). Old trees
still load (stale attrs ignored). Next: C2+C3 code.

**C. Second audit (2026-10-08, plan; 10 findings checked against code, all real).** Scores: quality 6.5,
readability 6, architecture 6. Architecture kept (pipeline boundaries are fine); the fixes are local.
- C1 (no result change): delete unused `max_leaf_size` (XModel/node/clusterers; configs set it but nothing reads it);
  `label_max_cos(..., rows, cols)` scores only candidate pairs in query chunks for `XModel.predict` (full
  matrix kept for diagnostics); split CLI: all three official PMID lists or none, ratios >= 0 summing to 1,
  drop `test_ratio`; load `assert`s -> `raise ValueError`, `beam_size >= 1` / `topk >= 0` checked in
  `resolve_predict_config`; diagnose_unseen kNN-free pool with `topk=0`; diagnose_routing guard
  `root.is_last_layer`; remove the 6 `joblib.dump` save fallbacks (raise `TypeError` before writing);
  docs: `filter_labels_and_inputs` (flat labels), `reduction.py`, sparse/ndarray hints, stale tests/README lines
  (rankers, `per_leaf`, `early_stopping`); `Y_binazer` -> `Y_binary`; junk comments; delete unused
  `cluster_to_labels`. Regressions in `tests/test_audit_regressions.py` (partial split, beam 0, save without
  `.save`, candidate kNN = full kNN at random rows).
- C2 (readability; changes saved-tree format -> retrain): plain attributes instead of the ~19 pass-through
  property pairs (xmodel, node, clusterers, tree, matcher, encoder; keep `local_to_global_idx` setter and
  `is_empty`); `tree.train`: `NodeInput` NamedTuple for the positional child tuples, inline
  `_accumulate_children` / `_save_ml_for_layer` / `_finalize_layer`; shape lines (`X` n x d, `Y` n x L,
  `Z` L x d, `C` L x K) at API entry points; PIFA = `normalize(Y.T @ X)` (drops the 1e-10, ~1e-7 shift, may
  flip k-means ties). Pass: user retrain + dev eval (B1 commands) equal to B1 0.9041 / MRR 0.9335 /
  unseen 0.6800 / hybrid 0.9062 within noise; no results.md row.
- C3 (user 2026-10-08): `XModel.save` also writes `xmodel.json` next to `xmodel.pkl`: every key of the pickled
  `state` (after C2's attribute rename). JSON-native values (configs, scalars, strings) as is; arrays/sparse
  matrices as `{"type", "shape", "dtype"}` (+ `nnz` for sparse); label lists as `{"type", "len", "first"}`;
  anything else as `repr`. Write-only (load still reads the pkl); one helper in xmodel.py, `json.dump(...,
  indent=2, default=...)`. Test: synthetic save -> json loads, keys == pkl state keys. Lands with C2's retrain.
- Do not run `/code-review` (user 2026-10-08).
- Deferred: dense classifier (#3; features are dense by design: `ponytail:` comment with the memory ceiling);
  all-empty context block (#4; only CTD-only training hits it, no planned run does); `wrapper.model.model`
  flattening and encoder `getattr(f"{prefix}...")` access (touch when editing those files).

**B. Bug audit fixes (2026-10-08, 14 findings):** knn fused before topk and rows re-sorted
(`tree.rank_rows`); split CLI rejects PMID overlap and output = input; fitted block widths
(`TextEncoder.block_widths`, `XModel.feature_blocks`, used by knn and the diagnostics); private cache tmp file;
TSV keeps "NA"/"NULL"; PubTator doc keeps title/abstract whitespace and the empty-title separator (BC5CDR,
CTD, MedMentions texts unchanged: checked no trailing whitespace / empty titles); hierarchy save refuses a
non-empty dir; sparse `fused_scores` unwrapped on load; same-second saves get `_1`; evaluate exits on 0
in-vocabulary rows; diagnostics use `xm.Z` (routing refuses one-layer trees); diagnose_unseen makes `-out`'s
dir; window 0 = empty context. Regressions `tests/test_audit_regressions.py`; pytest 28 + selfchecks pass.
Trees saved before B have no `block_widths` -> no knn eval: retrain.
**B1 passed:** tree `15-38-58` (`b1_train.log`, `b1_dev_eval.log`): cache hit, run 61.0 s; dev identical to N6
(0.9041 / MRR 0.9335 / unseen 0.6800 / hybrid 0.9062): no results.md row. `13-16-09` deleted; R uses `15-38-58`. R moved to tree `17-28-02` (C2).

**N6 passed:** tree `13-16-09` (`outputs/logs/n6_train.log`, `n6_dev_eval.log`, `n6_test_eval.log`): cache hit,
hierarchy 31.9 s, run 64.1 s; bare eval prints `search beam 10, topk 0, knn beta 10`; dev and test identical to the
knn10 references (dev 0.9041 / unseen 0.6800 / hybrid 0.9062; test 0.9163 / hybrid 0.9213): no results.md row.
User deletes tree `12-21-30` (no longer loads). U3 uses `13-16-09`.
**U3 code done (uncommitted):** `diagnose_unseen.py` had a latent N3 bug: `xm.predict` without `knn_beta` took
the tree's beta 10, so the fusion screen fused twice. Now: `-knn_beta`/`-beam_size` (default the tree's) set the
ranking behind acc@1, per-group rank buckets and the TSV (one `xm.predict`, = evaluate.py); the screen re-scores
`knn_beta=0` scores. TSV = every error, columns `group`, `abbrev` (mention text changed by the tree's
`abbrev_expansion`, from an abbrev-free reload), `gold_in_string_set`, `pred_in_string_set`, mention 1-NN for all
rows. Check: screen row beta 10 must equal the header per group. Synthetic end-to-end (tfidf tree, forced
errors, abbrevs) ok; pytest 23 + selfchecks pass.
**U3 D1/D2/D4 result** (`outputs/logs/u3_diag.log`, `u3_errors.tsv`, tree `13-16-09`, dev, beta 10): header = screen
row beta 10 per group (check ok). Errors 413: seen 1 68, seen >1 40, unseen 305.
D1 unseen (n 953): rank 1 0.680, 2-5 0.187, 6-20 0.064, >20 0.031, not cand 0.038 -> mostly scoring; routing cap 36 rows.
D2 unseen errors (hand read of 62 of 305; rough shares): pred a MeSH parent/child/sibling of gold or near-synonym
   (renal failure vs nephropathy, birth defects vs congenital anomalies, ischemia vs myocardial ischemia) ~55%;
   paraphrase needing knowledge (analgesia, writhing, startle, decrease of hr, contralateral rotation) ~20%;
   abbreviation ~13% (30/305 flagged `abbrev`, e.g. "ptld post-transplant ..." not cand; unexpanded SFs oab, edds);
   doubtful annotation ~10% (renal impairment and decreased renal function get opposite golds).
D4 seen >1: pred inside the string's train label set 40/40; gold inside 33/40; dict picks gold on 11/40 (psychosis
   7 rows: gold D011605 is the minority 2/9, so a prior hurts there). Seen 1 errors: gold outside the string set 66/68
   (dict wrong too; unfixable by a prior). Ceiling of a string prior ~ +9..11 rows = +0.002..0.0025 all (= hybrid gap).
User 2026-10-08: abbreviations later; D3/D4 screen deferred. Next = R (reranker) plan below.

**R. Candidate reranker (plan 2026-10-08; fixes D2's ~55% near-synonym / granularity errors).**
Target: gold in the tree's top K but not at 1. Dev headroom: acc@1 0.9041 vs R@10 0.9749 (unseen: rank 2-5 0.187,
6-20 0.064). Not the old per-label rankers (`docs/results.md`: separately trained per-label models had scores that were
not comparable across labels): one shared model scores every (mention, label) pair, so scores compare across labels.
- Model: cross-encoder, input `mention [SEP] context window` paired with `label text` = the label's most frequent
  distinct train mention strings (up to ~5, `; `-joined). Dataset-generic: on BC5CDR+CTD the annotators' strings come
  first, then CTD MEDIC names (one row each); on MedMentions only train mentions (no UMLS). Head = linear on [CLS]
  -> one logit. Init SapBERT (user 2026-10-08).
- Training data: the 4,182 BC5CDR train mentions (697 labels; they carry the annotation conventions and context)
  only (user 2026-10-08: no CTD rows; they lack context and SapBERT already covers synonymy). Per row: gold + K-1 negatives = the tree's top-K
  on that row (beta 10, gold removed) topped up with mention-knn neighbours; listwise softmax CE over the K.
  In-sample tree candidates are fine because the reranker never sees the tree score in training.
- Scoring: final = log(tree score) + w * reranker logit over the top K (K 10 default; 20 if R@20 helps);
  w swept in one eval (0, 0.5, 1, 2, inf) on dev, then fixed and saved; test once.
- Code (Claude): `xmr4el/rerank.py` (`CrossEncoderReranker`: fit, score pairs, save/load; `torch_device()`),
  `scripts/train_reranker.py -xmodel_path -train_path` -> `outputs/rerankers/<tree>_<ts>/` (records the tree name;
  load refuses another tree), `evaluate.py -reranker_path [-rerank_k] [-rerank_w]` with a w sweep table per
  string group. Label index -> name via `initial_labels` only. Synthetic test: tiny BERT config (random init,
  no download) learns a separable toy task; save/load round trip; index mapping at random rows.
- Runs (user): R1 train reranker (log `r1_train.log`); R2 dev eval with the w sweep (`r1_dev_eval.log`);
  pass = unseen and overall acc@1 above 0.6800 / 0.9041 beyond noise, seen 1 not below 0.9783 - 0.002;
  R3 test once, `docs/results.md` row (system comparison vs PECOS from here on: `docs/pipeline.md` note).
- Cost, measured on the user's Mac (MPS, SapBERT, seq 96 padded, batch 4 rows x 10): train 38 pairs/s = 18 min
  per epoch (41.8k pairs); dynamic padding should give ~1.5x -> 2-3 epochs ~25-55 min. Inference 130 pairs/s ->
  dev 5.5 min at K 10; the eval scores once and sweeps w from the same scores.
- Decided (user 2026-10-08): SapBERT init, Mac MPS, BC5CDR rows only, K 10.

**Logging cleanup (2026-10-07, commit `1ccaf0b`):** INFO reports stage/layer/node timings and shapes;
training `-verbose` enables DEBUG, `-quiet` keeps warnings/errors. Evaluation always
preserves warnings. Details and log capture: `scripts/README.md` § Train.
Logs before `1ccaf0b` (e.g. `outputs/logs/speed_baseline_bc5cdr_ctd.log`) use the old format
(`Training ML: Number 0`, no elapsed fields): compare stage spans from timestamps, never message names.

**Training speed (in progress).** Benchmark = BC5CDR-disease, `train_plus_ctd.pubtator`,
`configs/xmr4el_bc5cdr_config.json` (depth 2: root -> 16 leaves of 734 labels, 89,929 rows, rankers off).
PECOS 34.6 s is on exported features, so it compares to our hierarchy time, not the run time.
A. Baseline (`1ccaf0b`, `outputs/logs/speed_baseline2_bc5cdr_ctd.log`, tree `14-59-30`, mps): run 562.4 s =
   SapBERT 197.2 s (87,016 distinct of 89,929 rows) + TF-IDF/SVD 27.4 s + PIFA 4.0 s + hierarchy 330.5 s
   (root 78.7 s: clustering 37.7 s, matcher ~41 s; 16 leaves 251.4 s, 11.0-22.5 s each, leaf = matcher only).
   Leaf profile (synthetic 6000x2304x734, `JointOvRLogistic.fit`, scipy L-BFGS-B): 44% scipy `setulb`, 51% objective,
   of which numpy single-threaded logaddexp+expit ~48 ms vs two GEMMs ~15 ms each (Accelerate).
   Done (uncommitted): `JointOvRLogistic.fit` uses torch `LBFGS` (CPU, float32, strong Wolfe, history 10,
   stop at max|grad| <= tol). Synthetic: leaf 16.9 -> 3.9 s, root shape 4.3 -> 3.1 s, top-1 agreement 0.999;
   liblinear parity test, pytest (22) and selfchecks pass. Not bit-identical by design.
B. Passed: tree `15-15-09` (`outputs/logs/speed_torchlbfgs_bc5cdr_ctd.log`) has every dev metric of `13-47-38` at
   beam 10 (acc@1 0.845, `docs/results.md`); hierarchy 330.5 -> 197.9 s (leaves 251.4 -> 118.9 s), run 454.4 s.
   Root unchanged at 78.7 s. Solver iterations from the pickled `n_iter_`: root ~280, leaves 90-123.
   Not committed yet (user decides; commit it apart from C for attribution).
C. Done (uncommitted): root clustering. `kmeans_pytorch.pairwise_cosine` broadcasts an (N, K, d) tensor
   (1.7 GB per iteration at 11744x16x2304); `clusterers.py` swaps in a matmul version (`_pairwise_cosine`, installed
   inside `BalancedKMeans.train` because the package import stays deferred). Synthetic root-sized fit 31.7 -> 12.5 s,
   identical clusters (ARI 1.0); test `tests/hierarchy/test_clusterers.py::test_pairwise_cosine_patch`.
   Also done (uncommitted, same run): SapBERT embedding cache. `Transformer.transform` saves the distinct-text
   embeddings to `outputs/cache/transformers/<sha256 of model, pooling, max_seq_length, dtype, texts>.npy` and loads
   them on a later run with the same distinct texts (log `Transformer cache hit`, no `Transformer completed`).
   Train and eval sets get separate files (~270 MB for the 87,016 train strings). Pytest redirects the cache to
   tmp (`tests/conftest.py`); test `tests/features/test_transformers.py::test_transform_cache`. The first run
   after this change fills the cache (same time as before); later runs skip ~200 s. Compare speed by
   `Hierarchy completed`, not `Run completed`, once runs mix cache hits and misses.
   User: train + eval as in B with log `outputs/logs/speed_kmeansmm_bc5cdr_ctd.log`:
   `python scripts/train.py -train_path datasets/BC5CDR/disease/train_plus_ctd.pubtator -model_config configs/xmr4el_bc5cdr_config.json 2>&1 | tee outputs/logs/speed_kmeansmm_bc5cdr_ctd.log`
   `python scripts/evaluate.py -xmodel_path outputs/saved_trees/<run> -test_path datasets/BC5CDR/disease/dev.pubtator -train_path datasets/BC5CDR/disease/train_plus_ctd.pubtator -beam_size 10 -topk 0 -alpha 0 -path_score`
   Pass = `Clustering completed` sizes all 734 and root node well below 78.7 s, dev metrics = 0.845 +- noise.
C2. User, after C: same train + eval again to confirm cache hits (code frozen until both finish):
   `python scripts/train.py -train_path datasets/BC5CDR/disease/train_plus_ctd.pubtator -model_config configs/xmr4el_bc5cdr_config.json 2>&1 | tee outputs/logs/speed_cachehit_bc5cdr_ctd.log`
   then the eval of B/C with `2>&1 | tee outputs/logs/speed_cachehit_eval.log`. Pass = both logs show
   `Transformer cache hit` and no `Transformer completed`, `Encoding completed` ~25-30 s (TF-IDF/SVD only),
   dev metrics identical to the C tree (same features; only mps run-to-run noise is gone).
C/C2 passed. C (`speed_kmeansmm_bc5cdr_ctd.log`): sizes all 734, root node 78.7 -> 62.0 s, hierarchy 196.1 s.
   C2 (`speed_cachehit_bc5cdr_ctd.log`, tree `16-03-37`): `Transformer cache hit`, `Encoding completed` 24.1 s,
   root clustering ~16.3 s (node start -> `Clustering completed`), root node 50.2 s, hierarchy 173.2 s, run 203.0 s;
   dev beam 10: acc@1 0.8474, MRR 0.8900, R@10 0.9580, hybrid 0.8862.
E. Done (uncommitted): `BalancedBisect` in `clusterers.py` (type `balancedbisect`, kwargs `n_clusters`, `iter_limit` 20,
   `seed`; subclasses `BalancedKMeans` only for save/load/labels). Synthetic 11744x2304 x16: 0.66 s.
   Test `tests/hierarchy/test_clusterers.py::test_balanced_bisect`; pytest 25 pass.
   Config `configs/xmr4el_bc5cdr_bisect_config.json` = bc5cdr config with `balancedbisect`, `iter_limit` 20.
   User: `python scripts/train.py -train_path datasets/BC5CDR/disease/train_plus_ctd.pubtator -model_config configs/xmr4el_bc5cdr_bisect_config.json 2>&1 | tee outputs/logs/speed_bisect_bc5cdr_ctd.log`
   then the eval of B/C on the new tree with `2>&1 | tee outputs/logs/speed_bisect_eval.log`.
   Pass = sizes all 734, root clustering < ~2 s, dev acc@1 within noise of 0.847 (C2). If pass: switch all configs to
   `balancedbisect`, delete `BalancedKMeans` (move its save/load/labels into `BalancedBisect`), `_pairwise_cosine`,
   its test, the bisect config, and the `kmeans-pytorch` dep (+ numba if nothing else uses it). If acc drops: try
   `iter_limit` 50 before giving up.
E result: tree `16-12-24` (`speed_bisect_bc5cdr_ctd.log`): sizes all 734, root clustering 0.49 s (was ~16.3 s),
   root node 35.8 s, hierarchy 144.1 s, run 174.7 s. Dev beam 10: acc@1 0.8409 (C2 0.8474), MRR 0.8851, R@10 0.9556,
   R@cand 0.9940, hybrid 0.8811. Offline on the saved root Z (mean cosine to own centroid): kmeans_pytorch 0.5540,
   bisect 0.5509-0.5519 over seeds 0-2; iter_limit 50/200 gives the same, so `iter_limit` is not the cause.
   Gap = greedy splits are never refined jointly. Unknown: is -0.65 pt clustering-seed noise?
E2. User: bisect seeds 1 and 2 (`configs/xmr4el_bc5cdr_bisect_seed{1,2}_config.json`, temporary), train + eval each,
   logs `outputs/logs/speed_bisect_seed{1,2}_*.log`. Rule: both acc@1 >= ~0.845 -> seed noise, E passes (do the
   cleanup above). Both ~0.841 -> real drop: add a joint balanced refinement after the bisect (assign to 16 centroids
   under the size cap, a few rounds) and recheck the offline objective against 0.5540 before asking for a run.
E2 result: bisect seed 1 tree `16-20-11` acc@1 0.8323 (MRR 0.8814, hybrid 0.8783), seed 2 tree `16-24-12` 0.8334
   (MRR 0.8821, hybrid 0.8815); with seed 0 0.8409 -> mean 0.836 vs kmeans_pytorch 0.847: real drop.
   kmeans_pytorch = joint 16-way balanced Lloyd with an approximate auction assignment (eps = range/50), 400 iters.
E3. Done (uncommitted): after the bisect, `refine_iter` (default 20) joint rounds: centroids -> `_balanced_assign`
   (dual prices on over-full clusters, 300 steps, then greedy by score - price under the bisect sizes) until stable.
   Offline root objective (mean cosine to own centroid): bisect 0.5509-0.5519, plain greedy refinement ~0.5545
   (not monotone), scipy linprog exact 0.5584 (38 s/round, rejected), priced greedy 0.5572-0.5586 over seeds 0-2,
   3.0 s total; kmeans_pytorch 0.5540. pytest 25 pass.
   User: seeds 0 and 1 (`configs/xmr4el_bc5cdr_bisect_config.json`, `..._bisect_seed1_config.json`), train + eval each,
   logs `outputs/logs/speed_refine_seed{0,1}_*.log`. Rule: both acc@1 >= ~0.845 -> E passes (cleanup as in E, also
   delete the seed configs). Still ~0.836 -> the objective is not the cause; compare cluster contents vs kmeans_pytorch.
E3 seed 0: tree `17-07-13` (`speed_refine_seed0_*.log`): sizes all 734, root node 35.3 s, hierarchy 148.6 s,
   run 180.9 s; dev beam 10 acc@1 0.8530, MRR 0.8954, R@10 0.9621, R@cand 0.9937, hybrid 0.8918 (kmeans_pytorch C2:
   0.8474 / 0.8900 / 0.9580 / 0.9921 / 0.8862). Seed 1 tree `17-11-12` is truncated (`OSError: No space left on
   device` in `XModel.save`, ~1.7 GB per BC5CDR+CTD tree); rerun only the seed 1 command after freeing space.
E3 seed 1: tree `17-20-52` acc@1 0.8430, MRR 0.8888, hybrid 0.8873. Seeds 0/1 mean 0.848 vs kmeans_pytorch 0.847: pass.
E cleanup done: the new algorithm IS `BalancedKMeans` (type `balancedkmeans`, so configs keep their type);
   `kmeans-pytorch`, numba, llvmlite, `_pairwise_cosine` and its test, the bisect/seed configs are gone; configs use
   `iter_limit` 20 (bisect iterations; 50/200 gave the same objective), cuda config lost `device`. `uv.lock` updated.
   Real root (11744x2304, 16): 3.0 s (was ~16.3 s patched, ~37.7 s originally). pytest 24 pass.
E4. User: delete all saved trees (results stay in `docs/results.md`; `outputs/cache` keeps SapBERT embeddings), then
   train + eval BC5CDR with `configs/xmr4el_bc5cdr_config.json`, logs `outputs/logs/bkmeans_bc5cdr_ctd*.log`.
   Pass = sizes all 734, cache hit, acc@1 ~0.853 (= `17-07-13`, same seed 0 code path; mps noise only). Then add the
   E row to `docs/results.md` and suggest commits (B, C, E separately).
E4 passed: tree `17-28-48` (`bkmeans_bc5cdr_ctd*.log`) = `17-07-13` metrics exactly (acc@1 0.8530, MRR 0.8954,
   hybrid 0.8918), cache hit, sizes all 734, root clustering 3.0 s, hierarchy 148.4 s, run 178.5 s. Row in `docs/results.md`.
   Committed. Later: rerun the BC5CDR test section (`docs/results.md`) on `17-28-48`.
F. Node training speed (2026-10-07). All node time is the matcher's torch L-BFGS; time = iterations x cost/iter.
   `17-28-48` `n_iter_` (from `matcher/classifier_model.pkl`): root 258 (node 34.9 s, clustering 3.0 s), leaves 83-129
   (3.4-13.0 s, 113 s total). Synthetic (scratch benchmark, same closure): converges in ~40 iterations, so real data
   is the slow tail of the max|grad| <= 1e-4 rule. MPS vs CPU: root shape 90000x2305x16 6.1 -> 2.8 s (2.2x),
   leaf 6000x2305x734 2.1 -> 1.8 s (no gain); top-1 agreement 1.0.
   Why PECOS is 34.6 s (hierarchy only; ours 148.4 s): (1) leaf size. PECOS tree 8 -> 128 -> labels (~92 labels/leaf)
   and each leaf trains only on the rows whose gold is in that cluster (TFN negatives), so one pass over a layer
   costs ~89,929 x 92 x d; our 16 leaves of 734 cost ~89,929 x 734 x d, 8x more. (2) Solver: per-label liblinear dual
   coordinate descent, squared hinge, loose eps (PECOS defaults L2R_L2LOSS_SVC_DUAL, eps 0.1, max_iter 100, all
   threads; from PECOS docs, pecos is not installed locally) vs our joint logistic L-BFGS to max|grad| <= 1e-4
   (83-258 iterations, each 1+ line-search evals). (3) C++ threaded per label. Not the features: PECOS ran on ours.
   Plan, cheapest first, one change per run (all on `17-28-48`'s seed 0 path; baseline acc@1 0.8530, hierarchy 148.4 s):
F1. User: config-only `tol` 1e-4 -> 1e-3 (`configs/xmr4el_bc5cdr_tol1e3_config.json`, temporary), train + eval, logs
   `outputs/logs/tol1e3_bc5cdr_ctd*.log`. Same seed/code as `17-28-48`, so any metric change is the tol effect.
   Pass = acc@1 within ~0.002 of 0.8530 and hierarchy well below 148.4 s -> make 1e-3 the `JointOvRLogistic` default
   and the configs' value. Read `n_iter_` per node from the new tree (`matcher/classifier_model.pkl`).
F2. User: config-only smaller leaves, PECOS's main lever: root `n_clusters` 16 -> 128 (~92 labels/leaf, same depth-2
   code; `configs/xmr4el_bc5cdr_k128_config.json`, temporary), train + eval as F1 with logs `k128_bc5cdr_ctd*.log`.
   Expect leaves ~8x less work; root matcher has 128 targets (more cost, X reads dominate at 16). Watch R@cand
   (beam 10 x ~92 = ~920 candidates vs 1000 now) and acc@1; if routing loses gold, also try beam 20 at eval only.
   PECOS reaches 0.847 with this leaf size at beam 10. Pass = acc@1 >= ~0.847 and hierarchy far below 148.4 s.
   Combine with F1's tol if both pass (one confirming run).
F1 result (2026-10-08, tree `10-50-53`, `tol1e3_bc5cdr_ctd*.log`): metrics = `17-28-48` exactly, hierarchy 149.7 s, every
   node's `n_iter_` identical. Cause: the unscaled objective's gradient (sum over rows) never reached tol, so every fit
   stopped at the float32 loss stall (`tolerance_change` 1e-9); synthetic final max|grad| 0.26 for any tol <= 1e-2.
   So "slow tail of the 1e-4 rule" above was wrong. Fix (uncommitted): `JointOvRLogistic` divides loss and grad by n
   (sklearn lbfgs scaling, same minimizer). Synthetic 6000x2304x734: stall 36 iters -> 24 at tol 1e-4, 9 at 1e-2,
   top-1 agreement 1.0. pytest 24 pass. Tree `10-50-53` is redundant (user may delete).
F1b. User: same F1 command (tol 1e-3 config, now effective) on the scaled code, logs `tol1e3s_bc5cdr_ctd*.log`.
   Pass = acc@1 within ~0.002 of 0.8530, hierarchy well below 148.4 s -> 1e-3 default. Fail -> retry base config (1e-4).
F1b passed: tree `11-04-56` acc@1 0.8513 (-0.0017), MRR 0.8945, hybrid 0.8913, hierarchy 100.8 s, run 131.6 s;
   `n_iter_` root 118, leaves 62-93. Done (uncommitted): tol 1e-3 is the `JointOvRLogistic` default and in every
   joint-matcher config (incl. k128); temp tol1e3 config deleted; row in `docs/results.md`. New baseline for F2:
   `11-04-56` (acc@1 0.8513, hierarchy 100.8 s). F2 now runs with tol 1e-3 (no separate combine run).
F2 passed: tree `11-09-56` (`k128_bc5cdr_ctd*.log`), sizes 91-92, acc@1 0.8560, MRR 0.8977, hybrid 0.8873 (-0.004),
   unseen 0.595 (-0.018), seen 1-label 0.941 (+0.012), R@cand 0.9916 (917 cand), R@100 0.977 (0.987 at 16).
   Hierarchy 75.1 s: root clustering 17.6 s, root matcher 22.5 s, 128 leaves 34.9 s total; `n_iter_` 7-132, median 31.
   Row in `docs/results.md`. Not yet the configs' default.
F2b. User: eval only, `11-09-56` at beam 20 (log `k128_bc5cdr_ctd_eval_b20.log`). If unseen/hybrid/R@100 recover to
   the 16-leaf level at acc@1 >= 0.856, k128 + beam 20 is the setting; else decide k128 (acc@1) vs 16 (hybrid).
   Then the root is 40 of 75 s: next targets are clustering at k=128 (17.6 s; own balanced k-means) and F3 (root matcher on MPS).
F2b: beam 20 changes only R@cand (0.9916 -> 0.9940, 1835 cand); every ranked metric identical, eval 41 -> 85 s. So the
   lower R@20-R@100 at 128 leaves is cross-leaf scoring, not routing. Decision (Claude default, metrics within the
   seed-0/1 spread of ~0.01): 128 leaves, beam 10. `configs/xmr4el_bc5cdr_config.json` `n_clusters` 16 -> 128; k128
   config deleted. Uncommitted.
F2c. Done (uncommitted): `_balanced_assign` (k-means refinement, 20 calls per fit) was 2/3 Python greedy loop over
   n*k entries, 1/3 numpy float64 price loop. Now: price loop in torch float64 (bit-identical p), greedy replaced by
   vectorized row-proposing deferred acceptance (same matching: common weights -> unique stable matching, same tie
   order). Synthetic 11744x2304: k=128 17.6 -> 4.9 s, k=16 1.8 -> 1.0 s, labels identical to `HEAD` at both.
   Test `tests/hierarchy/test_clusterers.py::test_balanced_assign_matches_greedy`; pytest 25 pass.
   User: confirm run with the bc5cdr config (now k128), logs `k128c_bc5cdr_ctd*.log`.
   Pass = dev metrics identical to `11-09-56` (acc@1 0.8560, MRR 0.8977), root clustering ~5 s (17.6), hierarchy ~62 s.
   Then: commit (tol fix, k128 config, clustering separately), F3 if the root matcher (22.5 s) still dominates,
   rerun the BC5CDR test section on the new tree.
F2c passed: tree `11-31-05` (`k128c_bc5cdr_ctd*.log`) = `11-09-56` metrics exactly; root clustering 5.0 s, root node
   32.0 s (matcher ~27 s), 128 leaves 33.7 s, hierarchy 65.8 s, run 96.3 s (cache hit). Commits (uncommitted now):
   1. `classifiers.py` + joint-matcher configs' tol + `docs/results.md` `11-04-56` lines: scaled objective, tol 1e-3.
   2. `configs/xmr4el_bc5cdr_config.json` n_clusters 128.
   3. `clusterers.py` + `test_clusterers.py`: vectorized balanced assignment.
   STATUS.md / results.md go with the last commit. Trees `10-50-53`, `11-04-56`, `11-09-56` are superseded (user may delete).
   Next: F3 (root matcher ~27 s on MPS, synthetic 2.2x) and the BC5CDR test-section rerun on `11-31-05`.
F3. Done (uncommitted): `xmr4el.torch_device()` (cuda -> mps -> cpu; shared with `Transformer`). `JointOvRLogistic.fit`
   runs on it when n * L >= `GPU_MIN_ENTRIES` (5M), else CPU as before; Linux CPU-only = unchanged path.
   Synthetic per L-BFGS iteration on MPS: 89929x128 0.20 -> 0.12 s; 20000x128 break-even; 2064x92 leaves 2x slower,
   so at k=128 only the root qualifies. CUDA break-even not measured (ponytail note at the constant). pytest 25 pass.
   User: bc5cdr config train + eval, logs `mps_bc5cdr_ctd*.log`. Not bit-identical (MPS float32 reductions).
   Pass = acc@1 within ~0.002 of 0.8560, root node well below 32.0 s (matcher ~27 s -> ~16 s expected).
F3 passed: tree `11-45-47` (`mps_bc5cdr_ctd*.log`) acc@1 0.8551, MRR 0.8972, hybrid 0.8873; root node 32.0 -> 20.1 s
   (matcher ~27 -> ~15 s, 124 iterations), leaves 34.9 s, hierarchy 54.6 s, run 85.5 s (cache hit).
F5. Leaves are now 35 of 55 s (128 x ~0.27 s) while a synthetic leaf-sized fit is 0.04-0.16 s: profile one leaf's
   `MLModel.train` path for non-solver overhead before touching the solver.
G. Eval speed (plan 2026-10-08; eval of `11-45-47` = 42 s wall, stages unmeasured: eval logs WARNING only).
   Suspects from code (`scripts/evaluate.py`, `XModel.predict`, `Tree.predict`):
   - Routing: one `matcher_model.predict_proba(x_row)` per query per internal node (`tree.py` ~521): 4305 single-row
     calls at the root, each re-densifying one row (`JointOvRLogistic._dense`) plus Python trail dicts.
   - Load: `XModel.load` of 129 nodes + the SapBERT model, even when the eval encodings are a cache hit.
   - Test encoding: SapBERT cache hit? + TF-IDF/SVD transform.
   - `-train_path` breakdown: re-parses the 89,929-row train PubTator (with abbreviation expansion).
   - `evaluate.py`: three Python loops of `score_csr.getrow(qi)` over 4305 rows (gold rank, @cand, tree top-1).
G0. User (first step next session, code frozen): profile one eval with stage logs, nothing else changes:
   `python -m cProfile -o outputs/logs/eval_profile.prof scripts/evaluate.py -xmodel_path outputs/saved_trees/xmodel_2026-10-08_11-45-47 -test_path datasets/BC5CDR/disease/dev.pubtator -train_path datasets/BC5CDR/disease/train_plus_ctd.pubtator -beam_size 10 -topk 0 -alpha 0 -path_score -verbose 2>&1 | tee outputs/logs/eval_profile.log`
   Claude then reads the .prof (pstats, cumtime) + the INFO/DEBUG stage lines and ranks the suspects by seconds.
G1. Fix the largest stages, one at a time, cheapest first. Likely: batch routing per layer (group the beam by parent
   node, one `predict_proba` GEMM per node, same top-k/trail logic); lazy SapBERT load on cache hit; vectorized
   CSR loops in `evaluate.py` (indptr slices); cache or skip the train re-parse. Pass for each = the eval output
   identical to `11-45-47`'s (acc@1 0.8551, MRR 0.8972, hybrid 0.8873; batched GEMM may move float32 ties: then
   within ~0.001) and lower wall time. Synthetic check first where possible (README selfchecks).
G0 result (`outputs/logs/eval_profile.{log,prof}`, `11-45-47`): 52 s = load + parse ~2 s, encoding 0.3 s (cache hit),
   routing 14.3 s (`augment_features` 13.7 s: 43,050 single-row sparse hstack + sklearn `normalize`), ranking 35.5 s
   (`predict_labels` 32.5 s, of which `_cos_fallback` 25.9 s: per label, re-slices the leaf rows and recomputes their
   norms, 11,744 calls; row norms 16.3 s, matvec 9.5 s, slicing 3.2 s). Load, train re-parse, evaluate.py loops: minor.
G1a. Done (uncommitted): `scoring.py` `_cos_fallback(q_idx, li)` computes `X_query @ Z.T`, row norms and per-vector
   Z norms once per leaf call and gathers. Bit-identical to the per-label code on random float32/float64 CSR
   (per-vector `norm(z)`; `norm(Z, axis=1)` differs by 1 ulp). pytest 25 pass.
   User: `python scripts/evaluate.py -xmodel_path outputs/saved_trees/xmodel_2026-10-08_11-45-47 -test_path datasets/BC5CDR/disease/dev.pubtator -train_path datasets/BC5CDR/disease/train_plus_ctd.pubtator -beam_size 10 -topk 0 -alpha 0 -path_score -verbose 2>&1 | tee outputs/logs/g1a_eval.log`
   Pass = output identical to `eval_profile.log` (acc@1 0.8551, MRR 0.8972, hybrid 0.8873, every recall), Ranking
   35.5 -> ~7 s. Then commit G1a, then G1b: batch routing per layer (group rows by parent node, one predict_proba +
   one `augment_features` per node; same top-k/trail logic).
G1a passed (`outputs/logs/g1a_eval.log`): output identical to `eval_profile.log`; ranking 35.5 -> 6.6 s, wall 15 s
   (was 42 s unprofiled). Without cProfile routing is 6.3 s (14.3 s was profiler overhead on many small calls),
   so G1b gains ~5 s at most. Committed.
G1b. Done (uncommitted): `HierarchicalMLModel.predict` routes all queries layer by layer: beam entries of every
   query in one list (query order, then beam order) with `X_beam` row i = entry i; one `predict_proba` per node over
   its entries, one `augment_features` per layer over the kept candidates; same top-k, stable sort and cut per query,
   same leaf batch order. Removed `_init_beam_first_layer`, `_stack_batch`, `_add_beam_to_pending`.
   Synthetic old (HEAD) vs new, depth 2 and 3, 300 queries, 4 settings each: routes and score CSR identical except
   depth 3 beam 1: 1 of 300 queries differs by 1 float32 ulp in a layer-1 matcher prob (batched vs single-row GEMM).
   Random root-shaped GEMM (4305x2305x128) batched = single-row exactly, so the BC5CDR eval should be identical.
   pytest 25 pass.
   User: `python scripts/evaluate.py -xmodel_path outputs/saved_trees/xmodel_2026-10-08_11-45-47 -test_path datasets/BC5CDR/disease/dev.pubtator -train_path datasets/BC5CDR/disease/train_plus_ctd.pubtator -beam_size 10 -topk 0 -alpha 0 -path_score -verbose 2>&1 | tee outputs/logs/g1b_eval.log`
   Pass = output identical to `g1a_eval.log` (else within ~0.001: float32 ulp), `Routing completed` 6.3 s -> ~1 s.
G1b passed (`outputs/logs/g1b_eval.log`): output identical to `g1a_eval.log`; routing 6.3 -> 1.6 s, wall 15 -> 10 s.
   Committed.
F5 result (synthetic leaf 630x2307, 92 labels, bc5cdr config, torch + transformers imported): `MLModel.train` + save
   0.174 s, of which 3 `gc.collect()` per node 0.14 s (47 ms each; more in the real process's larger heap); solver
   ~0.03 s. Done (uncommitted): removed the 3 `gc.collect()` calls in `MLModel.train` (refcounting frees the `del`ed
   arrays; only cycles needed gc). Per-layer `collect()` in `tree.py` kept. Synthetic leaf 0.174 -> 0.032 s. pytest 25 pass.
   User: `python scripts/train.py -train_path datasets/BC5CDR/disease/train_plus_ctd.pubtator -model_config configs/xmr4el_bc5cdr_config.json 2>&1 | tee outputs/logs/nogc_bc5cdr_ctd.log && python scripts/evaluate.py -xmodel_path outputs/saved_trees/$(ls -t outputs/saved_trees | head -1) -test_path datasets/BC5CDR/disease/dev.pubtator -train_path datasets/BC5CDR/disease/train_plus_ctd.pubtator -beam_size 10 -topk 0 -alpha 0 -path_score 2>&1 | tee outputs/logs/nogc_bc5cdr_ctd_eval.log`
   Pass = dev metrics = `11-45-47` (acc@1 0.8551, MRR 0.8972, hybrid 0.8873; gc does not touch numerics, only MPS
   root noise ~0.002 possible), leaf layer 34.9 -> ~5-8 s, hierarchy 54.6 -> ~25 s. Then commit, add row to results.
F5 passed: tree `12-21-30` (`nogc_bc5cdr_ctd*.log`) dev metrics = `11-45-47` exactly; root node 19.5 s (clustering
   ~5 s, matcher ~14 s), leaf layer 34.9 -> 11.3 s (~0.09 s/leaf vs synthetic 0.03 s), hierarchy 31.1 s, run 61.6 s
   (encoding 25.1 s, cache hit). Row in `docs/results.md`. Committed.
F6 (candidates, user picks): run 61.6 s = encoding 25.1 s (TF-IDF/SVD on cache hit) + root 19.5 s + leaves 11.3 s.
   PECOS's 34.6 s compares to our hierarchy 31.1 s: training speed target met. Remaining options: encoding (SVD
   fit, the largest stage now), leaf residual (~0.06 s/leaf: temp save/load), or back to accuracy work.
   Test section rerun on `12-21-30` done (acc@1 0.873, hybrid 0.909 vs PECOS 0.862 / 0.904; `docs/results.md`).
U. Unseen strings (2026-10-08, user priority). Dev only (test stays for final numbers).
U0. `scripts/diagnose_unseen.py` (new, selfcheck ok): on dev unseen-string rows, tree acc@1 + gold-rank buckets
   (not in candidates = routing loss), flat 1-NN over the tree's train rows and nearest-z per feature block
   (all / mention=SapBERT / char / context), and `outputs/logs/unseen_errors.tsv` (tree errors with each label's most
   frequent train mention and the mention-block 1-NN prediction). User:
   `python scripts/diagnose_unseen.py -xmodel_path outputs/saved_trees/xmodel_2026-10-08_12-21-30 -test_path datasets/BC5CDR/disease/dev.pubtator -train_path datasets/BC5CDR/disease/train_plus_ctd.pubtator 2>&1 | tee outputs/logs/unseen_diag.log`
   Read: tree acc@1 must equal the eval's unseen 0.5950 (same rows; else the script is wrong). Then: routing loss
   large -> beam/root; gold in cand but ranked low -> leaf scoring; SapBERT 1-NN >> tree -> add a mention-kNN
   scorer for unseen strings (SapBERT's own method); all baselines ~tree -> read the TSV (label granularity, CTD noise).
U0 result (`outputs/logs/unseen_diag.log`, `unseen_errors.tsv`; tree acc@1 0.5950 = eval, script consistent):
   gold rank 1 0.595, 2-5 0.207, 6-20 0.072, >20 0.088, not cand 0.038 -> routing is not the problem, leaf order is.
   Flat 1-NN over train rows (acc@1 on the 953 rows): all 0.600, mention (SapBERT) 0.638, char 0.517, context 0.025;
   nearest z: 0.594 / 0.619 / 0.408 / 0.020. Tree wrong + mention 1-NN right 0.143, the reverse 0.100 (oracle 0.738).
   Errors: often a more specific CTD label ("obese" -> obesity, morbid; hypomagnesemia -> hypomagnesemia 4, renal),
   abbreviation-appended keys ("ob obese", "dm diabetes mellitus"), some annotation-level choices. Label-prior idea
   rejected: gold more frequent in BC5CDR train than pred in 41% of errors (pred 35%), 42% of error golds CTD-only.
U1. `diagnose_unseen.py` now also screens fusion on all dev rows (no training): each tree candidate scored
   log(tree score) + beta * knn, knn = max SapBERT-block cosine to the label's train rows; beta 0/5/10/20/40/inf,
   printed as the string breakdown + hybrid. Selfcheck ok. User: same command as U0 (log `unseen_diag2.log`):
   `python scripts/diagnose_unseen.py -xmodel_path outputs/saved_trees/xmodel_2026-10-08_12-21-30 -test_path datasets/BC5CDR/disease/dev.pubtator -train_path datasets/BC5CDR/disease/train_plus_ctd.pubtator 2>&1 | tee outputs/logs/unseen_diag2.log`
   Check: beta 0 row ~ eval (all 0.8551, unseen 0.5950; argmax tie order may move it by <0.001). Pick beta on dev by
   hybrid; if unseen gains >= ~0.03 without losing hybrid, implement it in `Tree.predict`/`evaluate.py` as a
   `-knn_beta` option (train rows' mention block kept in the tree), then confirm once on test.
U1 result (`unseen_diag2.log`, dev): beta 0 = eval exactly (0.8551 / unseen 0.5950 / hybrid 0.8873). beta 5: all
   0.9015, unseen 0.6758, hybrid 0.9052; beta 10: all 0.9041, seen 1 0.9783, seen >1 0.8113, unseen 0.6800, hybrid
   0.9062; 20: 0.9003 / 0.6621; 40: 0.8976 / 0.6506; knn only within candidates: 0.8846 (seen >1 0.590). Fusion lifts
   every group; the tree alone (0.904) now ~ the old hybrid. Chosen beta 10 (5 within 0.003: flat optimum).
U2. Done (uncommitted): `XModel.predict(knn_beta=0.0)`: candidates' scores x exp(beta * knn), knn =
   `scoring.label_max_cos` (max cosine of the query's `mention_block()` to each label's rows of `self.X`, labels from
   `self.Y`); candidates and routes unchanged. `evaluate.py -knn_beta`; `diagnose_unseen.py` reuses `label_max_cos`.
   Test: `test_pipeline_persistence` checks the exact formula (its 6-d synthetic features tie cosines, so no argmax
   check). pytest 25 pass. No retrain: X and Y are in `xmodel.pkl`.
   User: `python scripts/evaluate.py -xmodel_path outputs/saved_trees/xmodel_2026-10-08_12-21-30 -test_path datasets/BC5CDR/disease/dev.pubtator -train_path datasets/BC5CDR/disease/train_plus_ctd.pubtator -beam_size 10 -topk 0 -alpha 0 -path_score -knn_beta 10 2>&1 | tee outputs/logs/knn10_dev_eval.log && python scripts/evaluate.py -xmodel_path outputs/saved_trees/xmodel_2026-10-08_12-21-30 -test_path datasets/BC5CDR/disease/test.pubtator -train_path datasets/BC5CDR/disease/train_plus_ctd.pubtator -beam_size 10 -topk 0 -alpha 0 -path_score -knn_beta 10 2>&1 | tee outputs/logs/knn10_test_eval.log`
   Pass = dev = U1 beta 10 row (0.9041 / unseen 0.6800 / hybrid 0.9062; tie order may move <0.001). Test is
   reported as is (beta chosen on dev only). Then: results.md rows (dev + test), commit.
U2 passed: dev = U1 beta 10 row exactly (0.9041, MRR 0.9335, unseen 0.6800, hybrid 0.9062); test (run once) 0.9163,
   MRR 0.9402, seen 1 0.9845, seen >1 0.7299, unseen 0.7122, hybrid 0.9213 (was 0.873 / 0.653 / 0.909). Rows in
   `docs/results.md`. Committed. Literature test 93.2 / 93.5 (protocol not matched).
N. Normalize how the model runs (plan 2026-10-08, user decisions: remove rankers; keep a TF-IDF-only feature
   option for plain TSV but under a descriptive name, not flag 1/6; keep configs base + bc5cdr only).
   Goal: one canonical train path and one canonical predict path, settings stored in the config/tree, so a bare
   `evaluate.py` reproduces the reported numbers. Today a bare eval uses beam 5, topk 20, alpha 1, no path score.
   Key fact: at the validated `-alpha 0` the leaf score is the matcher probability; the cosine/ranker fusion
   (`scorer`, `alpha`, `fusion`, `p`, `eps`) only costs time (6.3 s of the 9 s dev eval), except an lp-fusion
   cube/cube-root round trip and an eps clip that may move float ties.
   N1 predict path (inference only, verify on `12-21-30`): `XModel.predict(X_text, beam_size, topk=0, knn_beta)`;
      leaf score = matcher prob x path prob (path score always on); remove alpha/fusion/p/eps/scorer/topk_mode/
      include_global_path/n_jobs/return_matrix; `topk` = global cut of the final row; return the score CSR only
      (evaluate's @cand = row nnz, same set as final_path labels). `evaluate.py` keeps -beam_size, -knn_beta, -topk as
      overrides, defaults read from the tree's config (N3). Drop evaluate's ranker report and tie detector branch.
      Pass = dev output = `knn10_dev_eval.log` (<= 0.001 if the fusion round trip moved ties); eval ~3 s.
      N1 passed, committed (dev identical to `knn10_dev_eval.log`, eval 6 s): `HierarchicalMLModel.predict(X, beam_size, topk=0)` returns the CSR only; each visited
      leaf gives its `LEAF_CANDIDATES` = 100 best labels (the old leaf `beam_size=100`), score = matcher prob x
      exp(path_logscore), rows sorted; `XModel.predict(X_text, beam_size, topk, knn_beta)`. Deleted
      `scoring.predict_labels`, `MLModel.predict`, `eval.split_by_ranker`, evaluate's -alpha/-scorer/-path_score,
      ranker report and tie detector; @cand = mean(rank > 0). Evaluate defaults for now: beam 5, topk 0, knn 0
      (N3 moves them to the config). Tests rewritten (`test_no_rankers` checks the exact root x leaf product;
      `test_scoring` now tests `label_max_cos`); pytest 25 + 4 selfchecks pass. README, scripts/README, pipeline.md
      eval docs updated.
   N2 training path: delete rankers (`ranker.py`, `Ranker`, `ranker_config`, `cur_config`, `ranker_every_layer`,
      `train_rankers`, `MLModel.ranker_model`, `_topb_sparse`/M_MAN, matcher `m_node` if only rankers read it), leaf
      label embeddings if only cosine/rankers read them, matcher types `sklearnsgdclassifier` and
      `sklearnlogisticregression` (+ the leaf SGD early-stopping override), clusterer `sklearnkmeans` (tests move to
      `balancedkmeans`; the liblinear parity test calls sklearn directly). Each deletion: grep every reader first.
      N2 done, committed: deleted `ranker.py`, ranker/cur/train_rankers/ranker_every_layer/n_workers args (XModel,
      tree, node, configs), `m_node`, sklearn LR/SGD classifiers, `sklearnkmeans`, the child-Z zero pad, leaf
      label embeddings (internal nodes keep Z: clustering + diagnostics read the root's). Also deleted now (would
      not load): configs flag1, flag6_sapbert, full_cuda, bc5cdr_dict (README/CLAUDE.md flag1 lines fixed in N3).
      Tests: `test_no_rankers.py` -> `test_tree.py`; `test_ranker.py`, `test_matcher.py` (SGD override) deleted;
      reassignment test now balanced (62 labels, 4 clusters, min_leaf 16). pytest 23 + 5 selfchecks pass.
      Depth-2 training is unchanged (rankers were off, leaves never cluster), so no run for N2 alone.
   N3-N5 done, committed together: `XModel(features="tfidf"|"sapbert_char_context", predict_config={...})`,
      `TextEncoder(features=...)` (other names raise; no old-flag message). `predict_config` defaults beam 10,
      topk 0, knn 0, merged with the config's; base config beam 2 / knn 0, bc5cdr beam 10 / knn 10.
      `XModel.predict` and `evaluate.py` args default None -> `resolve_predict_config`. Plain TSV: copy the base
      config with `"features": "tfidf"` (README § Grouped TSV). N4: diagnose scripts use `features`; others needed
      no change (`screen_leaf_scorer.py` kept: it screens sklearn scorers, not removed code). N5: README,
      scripts/README, pipeline.md, AGENTS.md, CLAUDE.md, full_cuda note. pytest 23 + 5 selfchecks pass.
   N3 config: `emb_flag` 1/6 -> `"features": "tfidf"` (TF-IDF->SVD of the text; plain TSV) | `"sapbert_char_context"`
      (today's flag 6); joint matcher for both; `"predict_config": {"beam_size": 10, "topk": 0, "knn_beta": 10}` (bc5cdr;
      base keeps knn_beta 0 until measured on MedMentions, U3 D5). Delete configs bc5cdr_dict, flag6_sapbert, full_cuda,
      flag1 (their rows stay in `docs/results.md`; `docs/full_cuda_comparison.md` gets a note). The TSV option stays
      covered by `test_pipeline_persistence` (already the TF-IDF path).
   N4 scripts: adapt `diagnose_routing.py`, `diagnose_unseen.py` (blocks from the features name), `beam_sweep.py`,
      PECOS baselines; delete `screen_leaf_scorer.py` if it only screens removed scorers (its results stay).
   N5 docs: README, `scripts/README.md`, `docs/pipeline.md` table, CLAUDE.md commands (`emb_flag` lines) updated.
   N6 verification: N1 on `12-21-30` (eval only). N2/N3 rename breaks old trees (no compat code): one bc5cdr retrain +
      bare dev eval (no flags) = `knn10_dev_eval.log` within MPS root noise (~0.002), then test once, then the user
      deletes trees `11-31-05`, `11-45-47`, `12-21-30`. pytest + all selfchecks after each step. One commit per step.
U3. Next-session diagnosis plan (2026-10-08). Dev only; test once per retained change. Tree `12-21-30`, base eval
   `-beam_size 10 -topk 0 -alpha 0 -path_score -knn_beta 10` (dev 0.9041; unseen 0.6800 n=953, seen >1 0.8113 n=212,
   seen 1 0.9783 n=3140 = dict). Remaining dev errors ~413: unseen ~305, seen >1 ~40, seen 1 ~68.
   Code first (Claude, then selfcheck): `diagnose_unseen.py -knn_beta b` so rank buckets, groups and the TSV describe
   the fused ranking; add TSV columns `abbrev` (mention carries an appended expansion), `group` (seen 1 / seen >1 /
   unseen), `gold_in_string_set` (gold among the labels train gives this exact string). One user run answers D1-D4.
   D1 unseen, where is gold: rank buckets after fusion. Mostly 2-5 -> scoring (D2/D3); `not cand` (cap 0.038) -> routing.
   D2 unseen, error types from the TSV (Claude reads ~60): pred more specific than gold (CTD child vs MeSH parent),
      abbreviation-appended key ("ob obese"), composite/partial mention, annotation-level (unfixable). Shares decide U4.
   D3 unseen, knn variants (screen, no training): label score = max (now) vs mean of top-2/3 rows; knn over train rows
      vs CTD-name rows only (row source in `train_plus_ctd.pubtator` doc ids); SapBERT on the raw mention vs the
      abbreviation-appended text (needs a SapBERT encode of ~1k dev strings: cache miss, ~10 s). Beta re-swept per variant.
   D4 seen >1 (dict 0.854 > tree 0.811): is the tree's pick inside the string's train label set? Inside -> screen a
      string prior: score x share(label | exact string)^gamma, gamma 0/0.5/1/2. Outside -> knn ties at cos 1.0 for every
      label with that string; check whether context separates them (context block alone is 0.025: likely not).
   D5 (optional, generality): knn fusion on MedMentions (no saved tree; one train + eval with the base config, user
      decides). A dataset-specific beta would weaken the claim.
   Decision rule: implement the largest measured gain first, one change per run, keep beta 10 unless D3 re-sweeps.
F4. Only if F1-F3 are not enough: a PECOS-style per-label solver (dual CD, squared hinge). Big change: the model
   becomes an SVM, and routing/scoring use sigmoid probabilities today.
E plan (original): Goal: root clustering ~38 s -> ~1-2 s and drop the git-pinned
   `kmeans-pytorch` dependency (+ numba). Read C's `Clustering completed` timing first; if the patched root
   clustering is already < ~5 s, E is only worth it for dropping the dependency.
   - Method: PECOS-style recursive balanced spherical 2-means. Normalize Z rows (eps for zero rows; never drop rows,
     `ClusteringTrainer.train` asserts one label per row). Split a set of n points into k clusters: if k == 1 stop;
     k_l = k // 2, k_r = k - k_l, n_l = round(n * k_l / k). Init 2 centroids from 2 seeded random points; iterate
     (max `iter_limit`, stop when the assignment stops changing): score = Z @ (c_l - c_r), the n_l highest go left,
     the rest right; centroids = normalized means. Recurse on both sides. Handles any `n_clusters` (6, 8, 16), and
     sizes differ by at most 1 at each split (16 x 734 on BC5CDR+CTD).
   - Code: new `ClusteringModel` subclass in `clusterers.py`, config type `balancedbisect`, kwargs `n_clusters`,
     `iter_limit` (default 20), `seed`; numpy only; same `{"cluster_labels": ...}` model dict, so
     save/load/`labels()` are copied from `BalancedKMeans`. Keep `balancedkmeans` until E passes (A/B in one codebase).
   - Tests (`tests/hierarchy/test_clusterers.py`): sizes balanced for n_clusters 6 and 16 on non-divisible n;
     separated blobs recovered exactly; same seed -> same labels; zero-norm row kept.
   - Verify: pytest + selfchecks; user trains `configs/xmr4el_bc5cdr_config.json` with only the clustering type
     changed (copy to `configs/xmr4el_bc5cdr_bisect_config.json`), eval beam 10 dev. Pass = dev acc@1 within
     noise of 0.845 (one factor changed; the cache makes features identical, so the delta is the clustering).
     Also check the MedMentions base config (n_clusters 6) at 500 labels before switching all configs.
   - If it passes: switch every config to `balancedbisect`, delete `BalancedKMeans`, `_pairwise_cosine`, its test,
     the `kmeans_pytorch` assert in `tests/integration/test_config_and_imports.py`, and `kmeans-pytorch`/numba from
     `pyproject.toml` (`uv lock`). Trees saved with `balancedkmeans` then stop loading (type lookup); retrain, no compat code.
D. Later: root matcher (~40 s): ~280 L-BFGS iterations x ~110 ms of two memory-bound GEMMs (89929x2305 by 16).
   Candidate: run the fit on mps when the node is large (synthetic root shape: mps 1.9 s vs cpu 3.1 s; leaves
   were not faster on mps). Then SapBERT 197-222 s on mps (batch size / fp16; outside the PECOS comparison).
   Node parallelism only if leaves still dominate. `gc.collect()` x3 in `MLModel.train` (`node.py`) can go anytime.

**Done: XMR4EL vs PECOS on BC5CDR** (`docs/results.md` § BC5CDR, § BC5CDR-disease dev/test).
- Data: `datasets/BC5CDR/CDR_Data/CDR.Corpus.v010516/`; disease-only splits `datasets/BC5CDR/disease/{train,dev,test}.pubtator`;
  CTD MEDIC dictionary `datasets/CTD/ctd_disease.pubtator` (`scripts/dict_to_pubtator.py`), merged as
  `train_plus_ctd.pubtator` (11,744 labels). Reader drops `-1` ids, splits composites via column 7
  (`tests/data/test_readers.py::test_bc5cdr_ids`).
- PECOS needs `-threshold 0` (default 0.1 cost it 0.075 on dev). On Mac: `docker build/run --platform linux/amd64`
  (`docs/full_cuda_comparison.md`).
- Selected: `configs/xmr4el_bc5cdr_config.json` (depth 2) on train + CTD, beam 10, hybrid (dictionary if string seen).
  Depth 3 (`xmr4el_bc5cdr_dict_config.json`) loses ~0.08-0.09. Test: XMR4EL 0.860 / hybrid 0.906 vs PECOS
  0.862 / 0.904 (tie). Literature test acc@1 BioSyn 93.2, SapBERT 93.5 (protocol not matched).
- Open accuracy gaps (after speed): unseen strings 0.64; tree below the dictionary on seen 1-label strings
  (0.930 vs 0.985); optional CTD cost on the 3631 shared dev rows.
- Earlier MedMentions PECOS rows used `threshold` 0.1: rerun with 0 before claiming parity or a win there.

**Current best, MedMentions (session 11-12):** base config = flag 6 (SapBERT mention | char TF-IDF->SVD 768 |
context window 10 TF-IDF->SVD 768) + `abbrev_expansion` append + matcher `jointlogisticregression`. Eval with
`-beam_size 2 -alpha 0 -path_score`. Dev, beam 2: 500 labels 0.862 / MRR 0.896 (PECOS 0.861, `threshold` 0.1);
1000 labels 0.843 / 0.884 (PECOS 0.838, `threshold` 0.1). Weak group: unseen strings 0.53-0.58.
Train 375.7 s at 1000 labels vs PECOS 12.4 s.

## Future work

- Fused matcher/ranker cluster scores (`MLModel.fused_predict` -> `scoring.fused_cluster_scores`): removed
  2026-10-07, it had no callers. Recover with `git show b5c56a8:xmr4el/learning/scoring.py`
  (`fused_cluster_scores`) and `git show b5c56a8:xmr4el/hierarchy/node.py` (`fused_predict`).
- Unseen strings (abbreviations not defined in the doc, generic labels); a candidate reranker on top of
  matcher x path score (must beat 0.843 at 1000 labels); encoder swap (flat screen first); training-time
  stage breakdown; full-label MedMentions run (user does not want it yet).

## Scope and working rules

- Train/evaluate on local PubTator or grouped TSV + label files. UMLS/KB integration is out of scope.
- The user runs corpus training/evaluation. Agents may run synthetic checks and provide commands.
- Commands: `python` on the user's machine (venv active; agents running checks use `.venv/bin/python`), `python3` inside Docker containers.
- Preserve datasets and `outputs/`. Dependencies are managed with `uv`.
- No compatibility code for old saved trees: a tree the current code cannot load is retrained.
- Use 500–1000 labels for diagnosis, not for claims about full-label-space performance.
- Change one experimental factor at a time. Record bundled changes as a bundle without attributing
  the outcome to one component.

## Interpretation rules

- Report uniform random-cluster chance beside routing: top-1 is `1/K`, top-2 is `2/K`
  for a single root with `K` clusters. At `K=6`, these are 0.167 and 0.333.
  The diagnostic also reports a majority-cluster reference to expose mention-frequency skew.
- `-topk 0` removes the final per-leaf cut, but the internal leaf matcher still selects at most
  100 clusters/labels. Candidate recall is a pure routing ceiling only when that limit is inactive.
- The evaluator's random-ordering line is approximate when candidate counts vary. For exact
  comparisons, average the per-query expectation `hit_i * min(k, n_i) / n_i`, zero for empty rows.
  Do not diagnose a broken ranker from a small gap to the approximate line.
- Metrics exclude out-of-vocabulary gold labels. Record kept/total alongside every row and compare
  different vocabulary sizes separately. Coverage times conditional accuracy gives all-mention
  accuracy when excluded labels count as misses.

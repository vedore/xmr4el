# Status and next steps

History: `docs/status_log.md`. Valid results: `docs/results.md`.
Implemented behavior, synthetic checks, pipeline observations, PECOS notes: `docs/pipeline.md`.
Commands: `docs/results.md` § Commands.

## Resume here

Last updated 2026-10-08 (training speed F done through F3 (`11-45-47`, hierarchy 54.6 s); eval speed G0 profiled; G1a cosine batching passed and committed (`194e5db`), eval 42 -> 15 s; G1b routing batch passed and committed, eval 15 -> 10 s; F5 gc fix passed and committed: tree `12-21-30`, hierarchy 31.1 s, run 61.6 s, eval 9 s, metrics unchanged).

**Runs:** none in flight. Saved trees: `12-21-30` (current), `11-45-47`, `11-31-05` (superseded; user may delete). Next: see F6 below. Saved trees on disk: `11-45-47` (current), `11-31-05` (superseded).

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
   Also pending: BC5CDR test-section rerun on the current tree.
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

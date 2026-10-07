# Status and next steps

History: `docs/status_log.md`. Valid results: `docs/results.md`.
Implemented behavior, synthetic checks, pipeline observations, PECOS notes: `docs/pipeline.md`.
Commands: `docs/results.md` § Commands.

## Resume here

Last updated 2026-10-07 (training speed: C and C2 passed; E done: E4 passed, tree `17-28-48` acc@1 0.853 in `docs/results.md`; next = commits B/C/E, then the test-set rerun).

**Runs:** none in flight. All older saved trees were deleted; only `17-28-48` exists. Uncommitted: torch L-BFGS (B), SapBERT cache (C), own balanced k-means (E; replaces kmeans-pytorch).

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
   Next: user commits B, C, E separately; then rerun the BC5CDR test section (`docs/results.md`) on `17-28-48`.
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

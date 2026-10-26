# Architecture and cleanup assessment

Inspected 2026-10-23. Archived 2026-10-26: historical record of the 2026-10-23 cleanup. The terminal-interface section predates the TUI (`docs/design/tui_architecture.md`).

## Purpose and boundary

XMR4EL trains and evaluates hierarchical label rankers for supplied biomedical mentions on local files. Its users are researchers and lab colleagues running experiments on laptops or GPU servers. It does not detect mentions or serve a knowledge base.

Essential workflow: read and group training texts by concept ID; fit text features; construct sparse targets and dense PIFA label embeddings; cluster labels; train node matchers; save a tree; rank new mentions; evaluate held-out PubTator annotations. Diagnostics, controlled baselines, and recorded experiment provenance support that research purpose.

Keep the Python API and thin scripts over separate data, feature, hierarchy, learning, and evaluation modules. Configuration stays explicit JSON. Persistence and logging belong alongside the components they describe. There is no need for backend discovery, an application framework, a database, a service, or a generic job scheduler. A future terminal interface should call this API and run existing scripts in child processes.

## Verified component inventory

| Area | Classification | Evidence and decision |
| --- | --- | --- |
| `xmr4el/xmodel.py` | KEEP | `train`, `predict`, `predict_encoded`, `save`, `load` coordinate the actual pipeline; `initial_labels` defines sorted output columns. |
| `data/readers.py` | SIMPLIFY | PubTator validation, context windows, abbreviation expansion, grouping, and TSV loading are used. pandas is used only to group two TSV columns; use `csv` and sorted integer keys. |
| `data/splits.py`, conversion scripts | KEEP | PMID splits, JSONL export, CTD dictionary input, BioC conversion, and entity filtering have CLI callers and tests. They are data preparation, not a second model pipeline. |
| `features/encoder.py` | SIMPLIFY | Two real compositions: TF-IDF, or transformer mention + character mention + context blocks. Keep normalization and fitted widths; remove unreachable fallback branches and redundant forwarding methods. |
| `features/vectorizers.py`, `features/reduction.py` | CONSOLIDATE | Each has one backend hidden behind a registry, metaclass, subclass, and outer wrapper. Use one concrete adapter for TF-IDF and one for SVD; keep configs and persistence. |
| `features/transformers.py` | KEEP | Frozen HF encoder, revision resolution, deduplication, cache, pooling, truncation, and OOM recovery have callers and offline tests. Arbitrary HF/local checkpoints are a working research capability. |
| `features/label_embeddings.py` | KEEP | PIFA and sorted binarization are called by `XModel._fit`; row ownership matters more than replacing a small static utility. |
| `hierarchy/clusterers.py` | SIMPLIFY | Balanced spherical k-means and undersized-cluster reassignment are active. Replace automatic registration with explicit backend selection; retain the algorithm/assignment boundary. |
| `learning/classifiers.py` | CONSOLIDATE | Only joint OvR logistic regression is selectable. Keep the Torch solver, label blocking, sparse features, CPU/GPU selection and fallback; remove the extra estimator wrapper and ignored compatibility arguments. |
| `learning/matcher.py`, node label maps | REMOVE / SIMPLIFY | `MatcherTrainer` has one caller that discards three of four return values. Its global-to-local round trip is identity because node `Y` columns already follow node label order. Train directly on local `Y @ C`; retain filtering of rows without a positive label. |
| `hierarchy/node.py`, `hierarchy/tree.py` | KEEP | Fixed-depth node training, temporary node storage, child mapping validation, batched beam search and path probabilities implement the core method. Remove unused inverse maps, not local-to-global output ownership. |
| `learning/scoring.py`, `eval.py` | KEEP | Shared mention-kNN, candidate fusion, ranking metrics, coverage and dictionary breakdowns have several callers. Tie semantics differ intentionally between screens and evaluation. |
| `rerank.py`, `train_reranker.py` | KEEP | Working optional cross-encoder and document-fold negative generation; called from evaluation and covered by tests. No evidence supports deleting it. |
| `scripts/train.py`, `evaluate.py`, `run_experiment.py` | KEEP / SIMPLIFY | Real entry points; tracked runs validate specs and write provenance, metrics and completion/failure records. Remove the ineffective in-process `LD_PRELOAD` mutation from training. |
| Diagnostic, screening and latency scripts | KEEP | Evaluate saved trees or isolated scoring hypotheses. They are explicitly documented research workflows, not abandoned runtime plugins. |
| PECOS/scispaCy scripts and Dockerfiles | KEEP / INVESTIGATE | Separate external baseline runtimes with shell callers and recorded comparisons. Existing project notes identify them as local research artifacts. Do not remove them from this development checkout. |
| `scripts/run_*.sh` | INVESTIGATE | Experiment recipes depend on Linux, Docker, particular artifacts and latest-run globbing. Some replace exports. Preserve; do not expose these as general terminal-interface operations. |
| `configs/` | KEEP | Model presets, equivalence variants, train/eval specs, and eval-only sweeps have concrete script references. Repeated JSON values alone do not justify inheritance. |
| `tests/`, CI, packaging, main Dockerfile | KEEP | Synthetic offline suite, installed-wheel smoke, lint and locked dependencies support reproducibility. No configured type checker exists. |
| Historical reports, thesis, results, status | KEEP / INVESTIGATE | Research records and proposals are not runtime code. Retain them, label superseded architectural descriptions, and leave existing user edits intact. |
| Four registry metaclasses; no-op destructors; load-without-config defaults | REMOVE | No alternate backend or subclass registration caller exists. Destructors only assign `None`; missing config files indicate incomplete artifacts, not a supported migration path. |

## Resulting execution and data contracts

`scripts/train.py` loads one model config, reads local files, optionally takes the first N label groups, calls `XModel.train`, and saves under `outputs/saved_trees/`. `scripts/run_experiment.py` adds run validation, hashes, git/package metadata, logs, and repeated split evaluation; eval-only mode loads an existing tree. These are synchronous processes. There are no repository-owned background services.

| Layer | Input → output | Owner |
| --- | --- | --- |
| Read/group | PubTator or numeric group-ID TSV + aligned label file → grouped texts/IDs | `data/readers.py` |
| Features | texts → normalized float32 dense or CSR `X`; fitted TF-IDF/SVD and transformer revision | `features/encoder.py` and adapters |
| Label representation | row ownership → sparse one-hot `Y`, sorted IDs, dense float32 `Z = normalize(Y.T @ X)` | `features/label_embeddings.py`, `XModel._fit` |
| Hierarchy | `Z` → node-local cluster matrix `C`, local-to-global indices, child map | `hierarchy/clusterers.py`, `tree.py` |
| Matching | node-local `X`, `Y @ C` → sigmoid probabilities over clusters or leaf labels | `learning/matcher.py`, `classifiers.py` |
| Inference | query features → beam candidates → leaf probability × path probability → optional mention-kNN → sparse ranked scores | `tree.py`, `XModel.predict_encoded` |
| Evaluation | PubTator gold + scores → acc@1, MRR, recall, candidate recall, coverage, optional string breakdown | `eval.py` |

Preprocessing semantics, feature balance, PIFA, cluster membership, matcher supervision and ranking formulas are preserved by the cleanup. Removing inverse maps changes internal plumbing only: node-local `Y` already has exactly the node's ordered label columns. Tests must check noncontiguous global IDs, unlabeled rows, score persistence, and input/config ownership.

## Dependencies and artifacts

Retain NumPy/SciPy for arrays and sparse matrices, scikit-learn for TF-IDF/SVD/normalization, Torch for joint logistic optimization and encoders, transformers/Hugging Face Hub for model/tokenizer loading and revision resolution, and threadpoolctl for bounded BLAS work. Remove pandas after its only caller is replaced. pytest remains a development dependency. setuptools/wheel build the package; uv locks environments. PECOS and scispaCy stay in their separate containers rather than project dependencies.

Tree directories contain encoder and hierarchy components, inference state in `xmodel.pkl`, a readable JSON summary, and lazy research state in `research.pkl`. A completed `xmodel.pkl` is the tree publication marker. Optional kNN state avoids loading training arrays when saved. Model files use pickle and require trusted sources. Rerankers store HF model/tokenizer files and metadata tied to a saved tree ID. HF downloads are the core external integration; no corpus or KB service is queried by the pipeline.

Internal adapter layouts change. Previously saved trees require retraining; no migration code is added. No saved trees, caches, datasets, exports, user work, or research results are deleted.

## Terminal interface readiness

Implemented operations and reusable state:

| Workflow | Existing API / command | State and artifacts to expose |
| --- | --- | --- |
| Validate and launch experiment | `check_spec`, `run`; `scripts/run_experiment.py -spec ...` | invalid spec before work; `running → completed / failed`; run log, provenance, split metrics, tree reference |
| Evaluate existing model | same runner with `-xmodel_path`; `evaluate_tree`; `scripts/evaluate.py` | loaded model + search overrides → scores/metrics; no training required |
| Predict supplied mention | `XModel.load`, `predict`, `initial_labels` | unloaded → loaded → scored; mention/context input; ranked concept IDs and scores |
| Inspect/compare runs | files in `outputs/runs/` | spec, git, input hashes, package versions, metrics and status; require equal eval hashes for fair comparisons |
| Prepare inputs | split/conversion scripts | input paths + output paths → prepared files; retain overwrite guards |
| Diagnose / tune | diagnostic scripts, eval specs, baseline scripts | saved tree → routing, coverage, error or latency reports; tune on dev |
| Train/evaluate reranker | `train_reranker.py`, `evaluate.py -reranker_path` | matching tree ID, candidate K, fold/training metadata, logits/fusion metrics |

Core data models are grouped texts/labels, flat corpus/label/span/document rows, model/search JSON, `X/Y/Z/C`, ordered label IDs, node inputs, ranked CSR scores, and run/metric dictionaries. Keep UI state outside model objects. Run training in a child process so loading/encoding/optimization cannot block the UI; consume logs and durable artifacts.

Missing (not implemented here): installed command, run index/browser, structured progress events, cancellation state, process ownership, resumable training, and beam-path explanation API. SIGTERM in the runner records failure; SIGKILL can leave `running`. The planned light/full terminal modes in `STATUS.md` remain future work; do not claim a working TUI.

## Remaining decisions and validation

Preserve research choices whose removal could change results: routing feature augmentation, branching/depth, block balance, abbreviations, optional kNN and reranking. Known limits remain dense label/weight memory, child training without out-of-cluster negatives, nondeterministic ties at candidate cuts, dictionary/context mismatch, corpus leakage beyond identical file paths, and model cache identity for edited local checkpoints. These need measured research changes, not cleanup guesses.

Baseline before editing: **100 pytest tests passed**. Corpus training/evaluation, external baseline containers, accelerator hardware, and the thesis contents are outside the synthetic validation scope.

## Applied cleanup

- Removed four registry metaclasses and their dictionaries; folded `Tfidf`, `SklearnTruncatedSVD`, and `JointLogisticRegression` into their existing public adapters. `BalancedKMeans` is the algorithm behind the concrete clustering adapter and returns assignments directly.
- Removed `MatcherTrainer`, unused inverse label maps across model/tree/node/matcher callers, ignored classifier `dtype`/`onevsrest` arguments, no-op destructors, and fallback loading without component configs.
- Matcher persistence now writes its classifier directly instead of an empty extra state pickle and nested directory. Estimator adapters persist native models and explicit JSON configs.
- Removed redundant encoder forwarding methods and impossible model-class/model-fit fallback branches. Missing encoder components now report their path with `FileNotFoundError`.
- Replaced pandas grouping with `csv` plus sorted integer group IDs. Retained quoted tabs/newlines, literal NA/NULL strings, empty texts, row order within groups, truncation, and label-count checks; malformed columns or group IDs raise explicit errors.
- Removed pandas from project requirements and regenerated the lock, also dropping python-dateutil, pytz, six and tzdata. Other locked package versions stayed unchanged.
- Removed training's ineffective in-process `LD_PRELOAD` assignment; the dynamic loader needs such settings before Python starts.
- Updated current documentation and marked historical architecture inventories as superseded. Preserved the pre-existing `STATUS.md` modification and untracked `1`.

Runtime Python shrank by **684 lines**. Added 16 regression cases for adapter round trips/backend rejection, TSV formats/errors, and local target order/unlabeled-row filtering through persistence. Existing end-to-end tests still cover sorting, float32, sparse/dense paths, routing, kNN, and saved trees. No framework, dependency, CLI, or TUI was added.

## Validation results

| Check | Result |
| --- | --- |
| `.venv/bin/python -m pytest -q` | **116 passed in 58.86 seconds** (baseline: 100 passed). |
| `uvx --offline ruff check xmr4el scripts tests` | Passed. |
| `uv lock --check --offline`, `git diff --check` | Passed. |
| `.venv/bin/python -m compileall -q xmr4el scripts tests` | Passed; source syntax and removed-dependency references also checked. |
| Six self-checks | Passed: evaluation, routing, unseen strings, feature screen, leaf scorer screen, PECOS comparison math (no external PECOS import). |
| CLI startup | 15 argparse help paths passed; positional `beam_sweep.py` printed usage and returned its existing status 1. |
| `bash -n scripts/<recipe>.sh` | All nine recipes passed syntax checks; none executed. |
| `uv build --offline --cache-dir /private/tmp/xmr4el-uv-cache --no-build-isolation --out-dir /private/tmp/xmr4el-cleanup-dist` | Source distribution and wheel built. |
| Installed wheel outside checkout, install marked read-only | All package modules imported; synthetic train/predict/save/load/kNN and `tests/smoke_installed.py` passed; wheel metadata has no pandas requirement. Used existing ML dependencies. |
| Before/after synthetic equivalence | 24 arrays bit-identical: fitted features, PIFA and ranked scores for dense TF-IDF, sparse TF-IDF and a mocked hybrid encoder; routing features on/off, kNN on/off. Baseline source `ebbf008f9ae47172fb488b17bb995183ebb64c60` extracted into a temporary directory; no corpus data or saved user trees used. Probe: `/private/tmp/xmr4el-cleanup-equivalence.py` (temporary validation artifact). |
| Type checking | No type checker or type-check configuration is present; none added. |

Synthetic equivalence does not establish corpus accuracy, GPU reproducibility, or external-container compatibility. Those remain unverified. Existing saved trees and callers of removed internal wrapper classes/signatures need retraining or updating; the documented `XModel` API and JSON backend names are retained.

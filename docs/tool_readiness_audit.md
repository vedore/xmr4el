# XMR4EL tool readiness audit

Audit date: 2026-10-09. Source of truth: the local fork at commit
`54803a7209772e57a67101dd1863f132dbaf3826`. No upstream comparison was performed.
This report records the preceding read-only audit; it does not implement its recommendations.

## A. Executive summary

**XMR4EL is PARTIALLY READY as an experimentation tool.** Its supported training,
prediction, evaluation, and persistence pipeline works in lightweight integration
tests. The existing abstractions are sufficient for a thin CLI without a major
rewrite. Reliable experiment management still needs stronger validation, complete
run outcomes, reproducibility metadata, and explicit execution isolation.

| Category | Verdict | Evidence and practical limit |
|---|---|---|
| Core framework | **READY** | Supported TF-IDF recipes train, predict, save, and reload in integration tests. This verdict concerns valid supported configurations, not every input or accelerator environment. |
| Research | **PARTIALLY READY** | Configurations, trees, logs, metrics, and Git metadata are saved. Dataset identity, dependency snapshots, complete seed handling, and pinned transformer identity are missing. |
| Experiment sandbox | **PARTIALLY READY** | Fresh model instances and separate processes can execute different configurations. There is no batch planner, scheduler, comparison layer, or durable run lifecycle. |
| CLI | **READY** | `XModel` and `evaluate_tree()` provide callable core APIs. Existing scripts already launch training and evaluation. A dependable management CLI needs the boundary fixes below. |
| Textual TUI | **PARTIALLY READY** | The model APIs can support it, but process supervision, cancellation, structured progress, and status tracking are not ready. |
| Production/package | **PARTIALLY READY** | A Python package is declared, but paths assume a checkout in places, persistence uses pickle, and clean installation and deployed execution were not verified. |

These judgments separate software readiness from predictive quality. Benchmark
accuracy changes, including abbreviation-expansion gains, do not establish API,
packaging, or orchestration reliability.

## B. Existing capabilities and verification

### Current execution flow

1. [Preprocessor](../xmr4el/data/readers.py) reads PubTator or grouped TSV inputs.
   Training data is grouped by label; texts in a group are positives or synonyms.
   PubTator input supplies annotated mention spans; the framework does not discover
   mentions in unannotated documents.
2. [`XModel._fit()`](../xmr4el/xmodel.py) flattens those groups, fits a
   [TextEncoder](../xmr4el/features/encoder.py), constructs the label matrix `Y`,
   and builds normalized PIFA label embeddings `Z` through
   [LabelEmbeddingFactory](../xmr4el/features/label_embeddings.py).
   `initial_labels` follows the sorted label columns of `Y`.
3. [`HierarchicalMLModel.train()`](../xmr4el/hierarchy/tree.py) builds and trains
   layers sequentially. [MLModel](../xmr4el/hierarchy/node.py) handles individual
   nodes, clustering, and matcher training. Child training uses the relevant
   cluster rows and augmented routing-score features.
4. [`XModel.predict()`](../xmr4el/xmodel.py) encodes queries and traverses the
   hierarchy. Leaf probabilities are multiplied by routing-path probabilities.
   Optional [mention-kNN scoring](../xmr4el/learning/scoring.py) adjusts candidate
   scores before final ranking.
5. [`evaluate_tree()`](../xmr4el/eval.py) loads evaluation mentions, filters to
   training-supported labels, computes ranking metrics, and returns a metrics
   dictionary plus intermediate state for optional downstream analysis.

### Supported components

| Area | Current implementation |
|---|---|
| Feature recipes | `tfidf`, `sapbert_char_context` |
| Sparse vectorizer | `Tfidf` |
| Reduction | `SklearnTruncatedSVD` |
| Transformer aliases | BioBERT, sentence BioBERT, SapBERT; custom checkpoint names through transformer configuration |
| Clustering registry | `BalancedKMeans` |
| Classifier registry | `JointLogisticRegression`, using `JointOvRLogistic` |
| Additional scoring | Candidate-restricted mention-kNN and optional cross-encoder reranking |

Registries exist in [vectorizers.py](../xmr4el/features/vectorizers.py),
[reduction.py](../xmr4el/features/reduction.py),
[clusterers.py](../xmr4el/hierarchy/clusterers.py), and
[classifiers.py](../xmr4el/learning/classifiers.py). They register imported classes
under lowercased class names. They are not a rich capability or parameter-schema
catalog. Screening scripts and separate PECOS baselines are not additional
interchangeable backends in the main pipeline.

Configuration can select the registered implementations and supported feature
recipes. The currently available algorithm inventory is small; modular structure
does not imply that arbitrary alternatives already work.

### Entry points and outputs

| Capability | Source | Status |
|---|---|---|
| Train through Python | `XModel.train()` in [xmodel.py](../xmr4el/xmodel.py) | Verified with small synthetic inputs |
| Predict through Python | `XModel.predict()` | Verified with small synthetic inputs |
| Save and reload | `XModel.save()/load()`, encoder and hierarchy persistence | Integration-tested; transformer checkpoint portability remains limited |
| Train from files | [scripts/train.py](../scripts/train.py) | Implemented for PubTator and grouped TSV; full corpus execution not repeated |
| Evaluate saved trees | [scripts/evaluate.py](../scripts/evaluate.py), `evaluate_tree()` | Metric and filtering behavior tested |
| Record one experiment | [scripts/run_experiment.py](../scripts/run_experiment.py) | Implemented and tested; lifecycle gaps remain |
| Rerank candidates | [rerank.py](../xmr4el/rerank.py) and evaluation script | Small CPU tests; not a general orchestration API |

The experiment wrapper accepts JSON with `name`, `model_config`, `train_path`,
and `evals`. It supports evaluation-only runs through `-xmodel_path`. It writes
`spec.json`, a copied model configuration for training runs, `git.json`, a tracked
Git patch when dirty, `run.log`, `tree.json`, and per-split metrics JSON.
`XModel.save()` returns the saved-tree path.

The encoder persists fitted vectorizers and reducers. Saved model state also
retains training texts and feature/label arrays needed by existing behavior.
Node persistence now omits transient fused scores and label embeddings. This is
useful, but complete saved trees can still be large.

### Checks actually performed

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/ -q -p no:cacheprovider
```

**Result: 45 tests passed in 32.07 seconds.** Coverage includes readers,
configuration/imports, label mapping, TF-IDF pipeline persistence, encoder
behavior, clustering, hierarchy traversal, classifier behavior, candidate scoring,
metrics, logging, the experiment wrapper, and small reranking checks.

Additional temporary probes verified:

- A tiny CPU TF-IDF hierarchy trained and predicted successfully.
- `depth=0` returned from training with an empty hierarchy.
- `beam_size=1.5` passed prediction configuration resolution, then failed during
  ranking with a partition-index type error.
- A NaN mention-kNN coefficient produced non-finite scores without early rejection.
- Empty query input failed in the feature pipeline with a low-level zero-sample error.
- Experiment validation accepted invalid identifiers, missing paths, and an invalid
  evaluation beam size.
- Evaluation with no supported gold labels returned metadata and `None` state,
  bypassing prediction-parameter validation.
- Repeated mocked wrapper calls retained file handlers; empty evaluations were
  still followed by a completion log.
- An injected failure during retraining left an existing model with its old tree
  and newly replaced encoder/label state.
- Git metadata reported a clean tracked tree while an untracked Markdown file
  existed. That file did not change executed code, but demonstrated the detection gap.

Transformer/OOM tests use mocks rather than downloaded checkpoints or real GPU
exhaustion. Existing corpus results recorded in [STATUS.md](../STATUS.md) provide
historical evidence, not experiments reproduced by this audit.

## C. Critical blockers for reliable experiments

### 1. Configuration validation is incomplete

**Evidence:** `check_spec()` in
[run_experiment.py](../scripts/run_experiment.py) checks required and unknown keys,
nonempty evaluations, and duplicate split names. It does not enforce value types,
valid identifiers, resource constraints, or file existence. `XModel.load_config()`
passes JSON directly into the constructor. `resolve_predict_config()` checks only
some numeric bounds. `HierarchicalMLModel.train()` accepts a zero-depth loop.

**Consequence:** malformed settings can fail after expensive work, produce an empty
model, create unintended output paths, or emit non-finite scores. Invalid numeric
parameters are not consistently rejected before execution.

**Smallest fix:** validate identifiers, paths, supported component names, positive
integer depth and beam size, nonnegative integer top-k, finite scoring coefficients,
and relevant component bounds before training or evaluation. Keep checks in shared
model/configuration boundaries so every interface benefits. No new configuration
dependency is required for an initial fix.

### 2. Empty evaluation can be recorded as successful

**Evidence:** `evaluate_tree()` in [eval.py](../xmr4el/eval.py) returns early with
metadata and `None` when filtering leaves zero evaluation rows.
[evaluate.py](../scripts/evaluate.py) exits normally, and the experiment wrapper
writes the metadata and logs run completion.

**Consequence:** a dataset or label mismatch can look like a successful benchmark
even though no predictive metrics were computed. Batch comparisons can silently
include incomparable or empty results.

**Smallest fix:** represent the outcome explicitly. A benchmark run expecting
metrics should fail clearly or record an explicit empty/skipped evaluation status.
Validate search settings even when evaluation has no eligible rows. Preserve
`rows_total`, eligible rows, and label coverage in comparisons.

### 3. Transformer identity and cache compatibility are not pinned

**Evidence:** [transformers.py](../xmr4el/features/transformers.py) builds cache keys
from checkpoint names, requested pooling/length settings, dtype, and unique texts.
Checkpoint revisions, tokenizer identity, and implementation changes are absent.
`sentence_model()` does not expose a pinned revision. The encoder saves transformer
configuration rather than a self-contained checkpoint.

**Consequence:** a cached vector may represent an older checkpoint or preprocessing
implementation. Reloading a tree on another machine can require network access or
resolve a different checkpoint. A saved tree alone is not a complete transformer
environment snapshot.

**Smallest fix:** pin checkpoint/tokenizer revisions, record effective encoder
settings, and include them plus a cache-format version in cache identity. Document
whether reload requires a preinstalled checkpoint or a bundled local checkpoint.
Keep the existing atomic cache-file replacement.

### 4. Reproducibility metadata is incomplete

**Evidence:** `git_state()` uses `git diff HEAD`, so untracked files do not affect
its dirty flag. Wrapper artifacts do not record dataset content/version identity,
dependency versions, hardware, complete random seeds, or all resolved defaults.
Reducer defaults can use `random_state=None`.

**Consequence:** runs can appear equivalent while using different local code,
datasets, environments, or randomness. A copied input JSON is useful but does not
fully describe effective execution.

**Smallest fix:** record Git status including untracked paths, resolved settings,
dataset identifiers/fingerprints, relevant dependency versions, hardware, and seeds.
Record enough to reproduce each stage; this does not require MLflow or a server.

### 5. Run and artifact completion are not durable

**Evidence:** [run_experiment.py](../scripts/run_experiment.py) writes JSON directly
and has no persistent status/failure record. Exceptions are logged and raised;
interruptions are not modeled as run states. Run directories use second-resolution
timestamps plus names. `XModel.save()` reserves a unique directory, but publishes
its contents progressively rather than marking a complete atomic artifact.

**Consequence:** a crash can leave partial JSON or a partially saved tree. A run
directory does not prove completion, and automatic retry cannot distinguish failed,
interrupted, and completed stages reliably. Two equally named starts in the same
second collide, although existing runs are not overwritten.

**Smallest fix:** write atomic metadata updates, persist running/completed/failed/
cancelled outcomes with failure reasons, use collision-resistant run IDs, and mark
saved trees complete only after persistence succeeds. Do not call ordinary retry
checkpoint resumption: durable training checkpoints are currently absent.

## D. Technical debt

### 6. Wrapper logging leaks across repeated in-process calls

`main()` in [run_experiment.py](../scripts/run_experiment.py) attaches a root
`FileHandler` without closing or removing it. Global verbosity and import-time
logging configuration elsewhere also affect the process. Repeated wrapper calls
can write later logs into earlier runs and accumulate handlers.

Close/remove handlers in `finally`, and extract a callable execution function from
argument parsing. Separate subprocesses are a practical initial isolation boundary.
Fresh model instances alone do not isolate process-global logging.

### 7. Failed retraining leaves a reused model inconsistent

`XModel.train()` replaces encoder, arrays, and labels before the new hierarchy has
finished training. The previous `self.model` remains until success.

The failure probe demonstrated an old tree paired with new encoder/label state.
Until corrected, construct a fresh `XModel` for every run. A small fix is to commit
new fitted state only after successful training, or explicitly invalidate the model
on failure. This is not a reason to rewrite the pipeline.

### 8. Resource selection is implicit and memory use is substantial

[torch_device()](../xmr4el/__init__.py) automatically selects CUDA, MPS, or CPU.
Transformer configuration discards `device` and selects automatically.
[JointOvRLogistic.fit()](../xmr4el/learning/classifiers.py) selects acceleration
using an `n * L` threshold, not an explicit run allocation or memory estimate.
Its training uses dense features, targets, scores, and class weights; LBFGS also
retains optimizer history. Classifier training limits BLAS threads to one, but this
does not constitute a project-wide worker policy.

[label_max_cos()](../xmr4el/learning/scoring.py) restricts score computation to
candidates but still densifies the training mention matrix. PIFA construction and
other transformations can also allocate dense intermediates. Transformer OOM
retries reduce batch size; they do not solve total host-memory pressure.

Sequential subprocess execution is the sensible default. Add explicit device and
thread controls before concurrent execution. GPU scheduling and memory budgeting
must come before launching several models on one accelerator.

### 9. Paths and packaging assume a source checkout

Transformer cache paths derive from the installed module location, which can land
under a read-only `site-packages` directory. Script outputs are relative to the
current working directory. The wrapper derives its Git checkout from its script
location, while [train.py](../scripts/train.py) contains environment-specific setup.
[pyproject.toml](../pyproject.toml) packages `xmr4el` but declares no console entry
point; scripts and configuration presets are not explicitly installed as package
resources by the current manifest.

Use an explicit writable output/cache root and test a clean installed package from
outside the checkout. Add a package entry point only when implementing the CLI.
Do not infer wheel usability from successful imports in the developer checkout.

### Other bounded issues

- Empty prediction input and entirely empty context blocks can fail through
  lower-level feature errors. Decide and validate the supported behavior.
- Grouped TSV labels rely on positional correspondence; surplus labels are
  truncated. Explicitly validate/document this contract.
- Cross-encoder `fit()` seeds after `from_pretrained()` has constructed the model;
  a randomly initialized head is not made reproducible by that later seed alone.
- Classifier training expects nested wrapper/model behavior. Adding a backend
  requires implementing that contract, not merely registering a class name.
- Hierarchy training uses temporary node artifacts, not durable resume checkpoints.
- Pickle persistence should load only trusted artifacts; saved state is not a safe
  interchange format for arbitrary external models.

## E. Missing experiment-management functionality

| Capability | Existing foundation | Remaining work |
|---|---|---|
| Programmatic training/evaluation | `XModel`, `evaluate_tree()` | Shared validated run function independent of script `main()` |
| Configuration inspection | JSON constructors and component registries | Parameter metadata, defaults, types, cross-component validation |
| Algorithm discovery | Imported-class registries | Public capability listing; distinguish main backends from standalone baselines |
| Run isolation | Fresh instances; independent processes possible | Formal per-run context and process ownership |
| Batch execution | One experiment wrapper | Sequential batch planner, bounded parameter expansion, failure policy |
| Progress | Logs and timings | Structured stage/progress events and durable status |
| Cancellation | Process interruption possible | Managed termination and consistent cancelled state |
| Retry/resume | Saved trees allow evaluation-only execution | Safe stage reuse; checkpoint resume only where genuinely supported |
| Artifact reuse | Transformer vector cache; fitted saved encoder/tree | Explicit compatibility identities and dependency-aware reuse |
| Results comparison | Metrics dictionaries and JSON | Run index, comparable-row/label checks, CSV/JSON export |
| Resource management | Automatic device selection and local OOM handling | Explicit device/thread limits, accelerator ownership, concurrency budgets |
| Reproducibility | Input snapshots, Git commit and tracked patch | Resolved defaults, seeds, dataset/environment/checkpoint identity |

The architecture can support batches without changing preprocessing, PIFA,
clustering, matcher training, or traversal semantics. Reuse must respect stage
dependencies: transformer embeddings, fitted TF-IDF/SVD, label embeddings, and
trained trees are different artifacts with different compatibility requirements.

A lightweight local runner and filesystem metadata are sufficient initially.
Neither a dedicated orchestration framework nor MLflow is required to fix the
current blockers. Introduce them only if multi-machine scheduling or shared
tracking becomes a concrete requirement.

## F. Prioritized action plan

1. **Validate all execution inputs before expensive work.** Modify
   `scripts/run_experiment.py`, `xmr4el/xmodel.py`, and hierarchy training boundaries.
   Add regression coverage in `tests/test_run_experiment.py`,
   `tests/test_audit_regressions.py`, and hierarchy/configuration tests for invalid
   types, non-finite values, zero depth, identifiers, and missing files.
2. **Make outcomes explicit and artifacts reliably complete.** Modify
   `xmr4el/eval.py`, `scripts/evaluate.py`, `scripts/run_experiment.py`, and
   `XModel.save()`. Cover empty evaluation, failed/interrupted runs, partial saves,
   atomic metadata writes, and run-ID collisions.
3. **Complete reproducibility metadata and transformer cache identity.** Update
   `scripts/run_experiment.py`, `xmr4el/features/transformers.py`, and encoder
   persistence as necessary. Record effective settings, revisions, datasets,
   seeds, environment, and untracked code status. Extend transformer/encoder tests.
4. **Expose one reusable run function and fix state cleanup.** Extract script
   execution into a small module such as `xmr4el/experiments.py`, retaining the
   existing wrapper as a caller. Close log handlers and use fresh models per run.
   Address failed refit state in `xmr4el/xmodel.py`; cover both failure and reuse.
5. **Make resources explicit.** Update `xmr4el/__init__.py`, transformer execution,
   and classifier device selection. Preserve current defaults where appropriate;
   add explicit CPU/device/thread control and avoid concurrent GPU work by default.
6. **Add sequential batches and result comparison.** Build on the shared run
   function rather than duplicating training logic. Validate the complete plan,
   cap grid expansion, persist individual outcomes, and compare coverage alongside
   ranking metrics. Restart failed stages only when their prerequisites are valid.
7. **Verify installation and external working directories.** Check
   `pyproject.toml`, `MANIFEST.in`, script entry points, writable cache/output paths,
   and Docker behavior. Perform a clean installation smoke test before claiming
   reusable package deployment.
8. **Build interfaces after those boundaries are stable.** A small CLI is the next
   useful interface. Add Textual only after process supervision, status events,
   cancellation, and result indexing exist. Keep all training/evaluation logic in
   the shared layer.

Essential work is items 1–4 and a safe sequential execution policy. Parallel
scheduling, rich tracking, durable training checkpoints, and a TUI are later
enhancements, not prerequisites for a useful local research tool.

## G. Final recommendation

**Fix the execution boundaries first, then build a small CLI. Defer the Textual
application until run supervision is available.**

Keep `XModel`, `TextEncoder`, the component registries, hierarchical models,
matchers, and metric helpers. They already provide the required pipeline
separation. The main gaps sit around validation, run state, reproducibility,
resources, and artifacts; a new interface cannot repair those implicitly.

The current framework is usable for supervised local experiments with valid
configurations, fresh model instances, separate processes, and inspected outputs.
It is not yet a robust unattended system for managing and comparing many runs.

### Verification limits

Not checked: fresh installation, wheel contents, Docker execution, real checkpoint
downloads, full-corpus retraining, or accelerator capacity. Historical corpus
results were not reproduced. Package portability and large-run resource behavior
remain unverified. No implementation files were modified during the audit.

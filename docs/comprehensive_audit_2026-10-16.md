# Comprehensive code quality audit

Date: 2026-10-16. Read-only audit; findings describe the implementation inspected on that date.

What this repo does: XMR4EL trains label-centric entity rankers using text features, PIFA embeddings, hierarchical clustering, and node classifiers. It was assessed as a research system handling large label spaces, including use as an installed Python library.

## Overall assessment

**6/10.** The pipeline boundaries are sensible, and regression tests cover important label-mapping, routing, classifier, and persistence behavior. The main weaknesses are unbounded memory allocations, fragile artifact identity, inconsistent input validation, and installation-related side effects.

## Checks completed

- **61 tests passed** in 39.68 seconds.
- **Five standalone self-checks passed:** routing, unseen mentions, feature screening, leaf scoring, PECOS comparison.
- **62 Python files** parsed; **25 JSON configs** parsed; **eight shell scripts** passed syntax checks.
- Wheel build and import from the extracted wheel passed using existing dependencies.
- `uv pip check`: all 50 installed packages compatible.
- Ruff: **10 diagnostics**. Mostly unused code; two closure-name diagnostics in the benchmark are not demonstrated runtime failures.
- `uv audit --frozen`: **two advisories** against locked PyTorch 2.10.0.
- No matching private-key/token signatures found in scanned tracked text files.

No repository files were edited during the audit. An unrelated `STATUS.md` modification appeared during inspection and was left untouched. This report was saved afterward at the user's request.

Representative commands:

```bash
.venv/bin/python -B -m pytest -p no:cacheprovider tests/
uv pip check --python .venv/bin/python
uv tool run --cache-dir /private/tmp/xmr4el-audit-cache ruff check --no-cache --select F,E9 xmr4el scripts tests --output-format concise
uv audit --frozen --cache-dir /private/tmp/xmr4el-audit-cache --output-format json
.venv/bin/python -B scripts/diagnose_routing.py -selfcheck
.venv/bin/python -B scripts/diagnose_unseen.py -selfcheck
.venv/bin/python -B scripts/experiments/screen_features.py -selfcheck
.venv/bin/python -B scripts/experiments/screen_leaf_scorer.py -selfcheck
.venv/bin/python -B scripts/baselines/pecos_compare.py selfcheck
```

## Critical

None confirmed under the assumed local, trusted-artifact workflow.

## High

1. **Classifier training loses sparse scalability — confirmed implementation limitation.**

   [`JointOvRLogistic.fit`](../xmr4el/learning/classifiers.py#L143) densifies targets and features, then creates several additional matrices proportional to training rows × labels. A node with 100,000 rows and 10,000 targets needs approximately **8 GB for the integer target matrix alone**; each additional float32 matrix needs 4 GB. CPU fallback does not address this allocation.

   **Recommendation:** Fit independent label blocks and bound temporary matrices. Preserve the current objective and verify probabilities against the existing liblinear comparison tests.

2. **Prediction memory grows with the complete evaluation batch — confirmed implementation limitation.**

   [`HierarchicalMLModel.predict`](../xmr4el/hierarchy/tree.py#L315) copies features across beam entries and computes complete leaf probability matrices before selecting candidates. [`evaluate_tree`](../xmr4el/eval.py#L101) submits all retained queries together. Reducing `topk` does not bound these intermediate allocations.

   **Recommendation:** Process queries in bounded batches, including within large visited leaves. Verify identical candidates and scores against unbatched prediction.

3. **Transformer inference requires a writable package installation — reproduced.**

   [`CACHE_DIR`](../xmr4el/features/transformers.py#L37) resolves relative to the installed package's parent directory. [`Transformer.transform`](../xmr4el/features/transformers.py#L86) requires a successful cache write after computing embeddings. A cache miss in a read-only installation raises `PermissionError`.

   **Recommendation:** Use a user cache directory. Cache-write failure should not discard successfully computed predictions. Add an installed-package/read-only-cache regression test.

4. **Loading an untrusted model can execute arbitrary code — confirmed, conditional security exposure.**

   [`XModel.load`](../xmr4el/xmodel.py#L246) and component loaders deserialize pickle before validating model state. This becomes a serious integration vulnerability if another application accepts uploaded or externally supplied trees. Pickle-based loading explicitly requires trusted artifacts. [scikit-learn persistence guidance](https://scikit-learn.org/stable/model_persistence.html).

   **Recommendation:** State the trust requirement explicitly. For external artifact exchange, use validated metadata and non-executable array/weight formats; until then, restrict loading to trusted artifacts.

## Medium

5. **PubTator parsing can silently attach incorrect context or drop annotations — reproduced.**

   [`load_pubtator_file`](../xmr4el/data/readers.py#L168) recognizes headers using substring membership. A mention containing literal `|t|` was treated as a title and disappeared. The same function accepted annotation document `2` under document `1`, with span `(-4, 999)`, and emitted context from document `1`.

   **Recommendation:** Recognize header structure explicitly; validate document IDs and `0 <= start <= end <= len(document)`. Include file and line information in errors.

6. **Out-of-fold candidate generation can introduce labels outside the saved model — reproduced.**

   [`out_of_fold_top_k`](../xmr4el/rerank.py#L65) trains fold trees on all supplied training labels, then assumes every prediction belongs to `xm.initial_labels`. A saved model restricted to label `A`, paired with training data containing `B`, raises `KeyError: 'B'`. The CLI accepts these inputs.

   **Recommendation:** Restrict fold training to the saved vocabulary before fitting. Validate fold count against available query documents before training any fold.

7. **Unknown search settings survive validation and fail during evaluation — reproduced.**

   [`XModel.__init__`](../xmr4el/xmodel.py#L123) accepts `predict_config={"beam_szie": 1}`. Direct prediction silently retains the default beam; [`evaluate_tree`](../xmr4el/eval.py#L101) later expands the settings and raises an unexpected-keyword `TypeError`.

   **Recommendation:** Reject keys outside `PREDICT_DEFAULTS` during configuration loading. This prevents expensive training followed by an avoidable evaluation failure.

8. **Output-writing guarantees differ between converters and split tools — reproduced/confirmed.**

   [`to_pubtator.main`](../scripts/to_pubtator.py#L93) creates its output before opening the input. A missing input leaves an empty output, and the next invocation refuses to overwrite it. [`split_pubtator.main`](../scripts/split_pubtator.py#L59) overwrites existing split files sequentially; interruption can leave mixed generations.

   **Recommendation:** Reuse the temporary-file approach already implemented in [`dict_to_pubtator.py`](../scripts/dict_to_pubtator.py#L58). Publish split outputs only after every file succeeds.

9. **Saved models do not freeze their transformer identity — potential reproducibility defect.**

   [`sentence_model`](../xmr4el/features/transformers.py#L42) supports a revision, but tracked model configs leave it unset. The encoder stores configuration rather than checkpoint weights. After a Hub update or cache eviction, a saved classifier can receive features from a different checkpoint.

   **Recommendation:** Resolve and persist an immutable checkpoint revision during training. Include it in cache keys and artifact metadata. This issue is acknowledged in the cache comment but remains operationally significant.

10. **Latency measurements depend on embedding-cache history — reproduced.**

    [`bench_latency.main`](../scripts/bench_latency.py#L76) times `text_encoder.predict`, which includes the persistent embedding cache. Three identical transforms produced one encoder forward and two cache loads. Repeating the benchmark can therefore measure a different operation. [`evaluate_tree`](../xmr4el/eval.py#L102) also rounds total time before calculating per-query latency.

    **Recommendation:** Report cache hits and encoder execution separately. Calculate derived timing metrics from the unrounded duration.

11. **Context diagnostics count every tied gold cluster as rank one — confirmed logic defect.**

    [`diagnose_context.main`](../scripts/diagnose_context.py#L76) calculates rank using only strictly greater scores. With four equal scores, every gold cluster receives rank one: reported recall is 100%, while the uniform top-one chance line is 25%.

    **Recommendation:** Apply the actual routing tie rule or report expected recall under random ties. Print the chance line alongside routing measurements.

12. **“Routed” recall includes losses caused by final truncation — confirmed diagnostic ambiguity.**

    [`evaluate_tree`](../xmr4el/eval.py#L110) computes candidate recall after `topk`. Its per-string breakdown calls this value `"routed"`. With `topk=1`, candidate recall becomes accuracy@1, including ranking losses after successful routing.

    **Recommendation:** Measure routing/candidate retention before final truncation, or rename the metric to make that limitation explicit.

13. **PECOS exports can pair saved features with different training text — confirmed validation gap.**

    [`pecos_compare.export`](../scripts/baselines/pecos_compare.py#L53) reloads training text from a supplied file while exporting `xm.X`. Existing assertions compare row counts and labels, not text identity. Editing mentions without changing their labels/counts passes validation and makes feature-based and text-based baselines inconsistent.

    **Recommendation:** Export text from `xm.training_set`, which already defines feature-row order. Validate any supplied corpus against that state.

14. **Inference routing unnecessarily promotes float32 features to float64 — reproduced.**

    [`augment_features`](../xmr4el/hierarchy/tree.py#L34) receives float32 arrays during training and Python float lists during prediction. The latter produces float64 output, doubling value storage before [`_dense`](../xmr4el/learning/classifiers.py#L132) converts it back to float32.

    **Recommendation:** Construct appended routing values using `X.dtype`. Test both dtype preservation and score equivalence.

15. **The locked dependency set has known advisories requiring triage — scan confirmed; exploitability unconfirmed.**

    [`uv.lock`](../uv.lock#L1031) selects PyTorch 2.10.0. OSV flagged **CVE-2025-3000** and **CVE-2026-4538**. Repository code does not directly call the affected TorchScript/export-loading APIs; transitive reachability was not established. Upstream evidence: [TorchScript crash](https://github.com/pytorch/pytorch/issues/149623), [export-loading report](https://github.com/pytorch/pytorch/pull/176791).

    **Recommendation:** Record reachability decisions and update affected packages where fixes are available. Run dependency auditing regularly; do not equate package presence with a confirmed exploitable application vulnerability.

16. **Inference artifacts retain substantial training-only state — confirmed design limitation.**

    [`XModel.save`](../xmr4el/xmodel.py#L225) retains training text, `X`, `Y`, and `Z`; loading restores them even with kNN disabled. Dense embeddings are frequently stored as CSR, adding index storage. [`HierarchicalMLModel.load`](../xmr4el/hierarchy/tree.py#L160) additionally densifies sparse cluster assignments for validation.

    **Recommendation:** Separate inference-required state from optional research state. Validate child label membership through sparse indices. Preserve existing diagnostic exports where needed.

17. **The advertised package logger override misses component logs — reproduced.**

    [`set_logger`](../xmr4el/__init__.py#L13) changes the logger used by `get_logger`, while components independently use `logging.getLogger(__name__)`. A custom logger received the XModel message but missed the encoder message.

    **Recommendation:** Use one consistent logger strategy throughout the package. Test that custom handlers receive warnings from downstream components.

18. **Tests protect important regressions but miss deployment and diagnostic contracts — confirmed coverage gaps.**

    The automatic cache redirection in [`tests/conftest.py`](../tests/conftest.py#L6) hides the installed-cache failure. GPU fallback tests mock `_lbfgs`, so they do not verify actual device allocation/release. No tracked CI configuration was found.

    **Recommendation:** Add focused regressions for the reproduced failures above, an installed-wheel smoke test, and CI running the existing suite plus Ruff. Keep hardware checks separate. No numerical coverage percentage was measured.

## Low

19. **Public validation sometimes disappears under optimized Python — confirmed.**

    [`load_pubtator_file`](../xmr4el/data/readers.py#L150), [`Transformer.transform`](../xmr4el/features/transformers.py#L70), and component factories use `assert` for supplied configuration. `python -O` removes these checks; invalid abbreviation modes can then take unintended branches.

    **Recommendation:** Use `ValueError` for external arguments. Retain assertions for internal invariants.

20. **Dependency declarations and unused code need small cleanup — confirmed.**

    [`pyproject.toml`](../pyproject.toml#L18) declares `joblib`, with no direct imports in package/scripts, while directly imported `threadpoolctl` is undeclared. Ruff also identified unused imports, an unused clustering variable, and redundant f-string prefixes.

    **Recommendation:** Declare dependencies actually imported, remove the redundant direct `joblib` declaration, and remove verified unused statements. Joblib remains a transitive scikit-learn dependency.

## Architecture and integration assessment

The preprocessing → encoding → label embeddings → hierarchy → matcher boundaries should remain. The greatest coupling comes from persistence restoring whole object state, inference requiring training artifacts, and global cache/device/logger ownership. Address those boundaries before replacing factories or reorganizing modules. A broad rewrite is unnecessary.

## Top five improvements

| Rank | Improvement | Impact | Effort |
|---|---|---|---|
| 1 | Bound training targets and prediction batches | Prevents memory exhaustion as scale grows | Large |
| 2 | Repair parser, vocabulary, and export identity validation | Prevents corrupted experiments and late crashes | Medium |
| 3 | Move transformer caching outside the installation | Enables reliable installed-library use | Small |
| 4 | Persist immutable encoder identity | Makes saved models reproducible | Small–medium |
| 5 | Correct timing, ties, and routing diagnostics | Makes experiment conclusions trustworthy | Small–medium |

## Prioritized refactoring roadmap

1. **Reliability:** Add regressions for findings 3, 5–8, and 13; repair those boundaries; run existing tests automatically.
2. **Measurement:** Fix benchmark cache accounting, tie handling, and candidate/routing terminology before further comparisons.
3. **Scalability:** Batch prediction, block classifier targets, preserve float32, and remove dense assignment validation. Measure peak memory alongside numerical equivalence.
4. **Integration:** Separate inference artifacts from research state, freeze checkpoint identities, unify logging, and define supported dependency versions.
5. **Cleanup:** Remove verified unused code and redundant dependency declarations. Preserve component extensibility.

## Limitations

Not checked: full-corpus training, real CUDA/MPS execution, container builds, Git history secrets, or fresh dependency resolution. Memory limits and conditional artifact-loading risks remain unverified on deployment hardware.

# Code quality audit — 2026-10-16

What this repo does: XMR4EL trains hierarchical entity-linking rankers from texts grouped by label. This audit assumes local research runs with large label vocabularies, plus reuse through its Python API.

**Overall: 6.5/10.** The pipeline has useful component boundaries, substantial regression coverage, and working packaging. The main weaknesses are dependency compatibility, memory guarantees, input validation, and artifact integrity.

Every runtime package module and the Python/shell entry points were inspected. Findings reflect the final observed checkout, `7f832b6`; external edits occurred during the audit. The audit itself made no repository changes. This document records the audit results; it does not implement fixes.

## Critical

None confirmed.

## High

### 1. Package requirements permit incompatible installations

**Classification:** confirmed compatibility defect.

**Locations:** [pyproject.toml:24](../../pyproject.toml#L24), [transformers.py:17](../../xmr4el/features/transformers.py#L17).

`sentence-transformers` has no minimum version, although the implementation requires newer module paths, constructor arguments, and methods. For example, the permitted 5.1.1 release uses different constructor arguments. An existing environment can satisfy package metadata while failing to import or construct the encoder. See the [upstream 5.1.1 implementation](https://raw.githubusercontent.com/huggingface/sentence-transformers/v5.1.1/sentence_transformers/models/Transformer.py).

**Recommendation:** declare tested minimum versions and test both the minimum environment and `uv.lock`. Repository locking does not protect consumers installing the wheel into another project.

### 2. Sparse features become an unrestricted dense allocation

**Classification:** confirmed behavior; projected scalability risk.

**Locations:** [classifiers.py:145](../../xmr4el/learning/classifiers.py#L145), [scoring.py:43](../../xmr4el/learning/scoring.py#L43).

`JointOvRLogistic._dense()` always allocates `rows × (features + 1)` float32 values. Label blocking limits target matrices, but does not bound this allocation. A supported configuration retaining 100,000 features across 100,000 rows requires approximately **40 GB for this array alone**.

Candidate kNN also materializes query-chunk × selected-training-row similarities; candidate unions can cover much of the training set.

**Recommendation:** reject oversized dense conversions before allocation with an actionable reduction requirement. Independently block kNN training rows and retain running per-label maxima.

## Medium

### 3. The classifier silently accepts NaN/Inf inputs

**Classification:** reproduced.

**Location:** [classifiers.py:156](../../xmr4el/learning/classifiers.py#L156).

Both NaN and infinity in `X` produced a fitted object with nonfinite weights instead of an error. Empty training input raises `ZeroDivisionError`. This matters for callers using the classifier or encoded-feature API directly.

**Recommendation:** validate nonempty, aligned, finite inputs before fitting, and reject nonfinite fitted weights. Check sparse `.data` without densifying it.

### 4. Saving an unfitted model creates an unreadable completed artifact

**Classification:** reproduced.

**Location:** [xmodel.py:226](../../xmr4el/xmodel.py#L226).

`XModel(features="tfidf").save(...)` succeeds and writes `xmodel.pkl`, but loading fails because `hml/hml.pkl` does not exist. The completion marker therefore does not guarantee a loadable model.

**Recommendation:** require a fitted hierarchy, encoder, and label mapping before creating the output directory. Add a regression asserting that rejected saves leave no artifact.

### 5. Entirely empty contexts break feature fitting

**Classification:** reproduced.

**Location:** [encoder.py:182](../../xmr4el/features/encoder.py#L182).

Two valid inputs such as `"aspirin [SEP] "` and `"fever [SEP] "` fail with `empty vocabulary`. This affects dictionary-only training, `context_window=0`, and contexts containing only filtered tokens. Existing tests cover individual empty contexts within a nonempty corpus.

**Recommendation:** define the all-empty-context behavior explicitly: retain a zero context block, or reject the configuration before transformer encoding with a specific error.

### 6. PIFA unnecessarily promotes features to float64

**Classification:** reproduced.

**Location:** [label_embeddings.py:29](../../xmr4el/features/label_embeddings.py#L29).

`MultiLabelBinarizer` produces int64 labels. Multiplying these by float32 features produces a float64 intermediate; conversion back happens after densification and normalization. At 180,000 labels × 2,304 features, one float64 intermediate occupies **3.1 GiB**, before additional copies.

**Recommendation:** preserve feature precision during multiplication:

```python
Z = Y.astype(X.dtype, copy=False).T @ X
```

Validate numerical tolerances and ranking stability because accumulation precision changes.

### 7. Reranker compatibility checks use directory names instead of model identity

**Classification:** confirmed defect.

**Locations:** [rerank.py:161](../../xmr4el/rerank.py#L161), [train_reranker.py:79](../../scripts/train_reranker.py#L79).

Different trees with the same basename pass the check. Renaming the correct tree fails it. Independent output roots and copied artifacts make both cases possible.

**Recommendation:** save and compare the existing `tree_id`, together with the ordered-label identity needed by the reranker.

### 8. PubTator validation misses malformed supervision

**Classification:** reproduced.

**Location:** [readers.py:190](../../xmr4el/data/readers.py#L190).

An annotation whose offsets identify `"Aspirin"` but whose mention text is `"WRONGTX"` is accepted. An annotation ending with an empty concept field is silently dropped after stripping removes its trailing tab. Consequently, malformed data can change supervision or evaluation denominators without a clear diagnostic.

**Recommendation:** validate the original annotation surface and required fields, with filename/line errors or explicit rejection counts. Preserve the documented composite-mention handling.

### 9. Experiment validation does not detect train/evaluation overlap

**Classification:** missing safeguard; actual dataset leakage was not checked.

**Location:** [run_experiment.py:40](../../scripts/run_experiment.py#L40).

Validation checks paths and settings, but permits training and evaluating on the same file or overlapping documents. File hashes establish provenance without establishing split independence.

**Recommendation:** reject identical train/dev/test inputs and report document overlap for ordinary evaluation runs. Keep intentional training diagnostics explicitly identifiable.

### 10. Model exchange requires fully trusted artifacts

**Classification:** documented security limitation.

**Locations:** [xmodel.py:276](../../xmr4el/xmodel.py#L276), [node.py:95](../../xmr4el/hierarchy/node.py#L95).

Loading executes unrestricted pickle deserialization before structural validation. A malicious artifact can execute code. The README correctly documents this restriction; it is not a newly discovered remote-service exploit.

**Recommendation:** retain the trusted-artifact boundary in integrations. If untrusted model exchange becomes required, introduce a safe export format; post-load checks cannot secure pickle.

## Low

### 11. Optional constructor settings fail late with implementation errors

**Classification:** reproduced.

**Locations:** [xmodel.py:114](../../xmr4el/xmodel.py#L114), [classifiers.py:83](../../xmr4el/learning/classifiers.py#L83).

`XModel(features="tfidf").train(...)` finishes encoding and then raises `AttributeError` because `matcher_config` is `None`. Several backend dispatchers similarly expose `KeyError` for unknown types.

**Recommendation:** validate required component configurations at construction and report supported types. Document whether omitted configurations have defaults.

### 12. Embedding-cache keys are ambiguous for strings containing NUL

**Classification:** reproduced.

**Location:** [transformers.py:107](../../xmr4el/features/transformers.py#L107).

Joining fields with `"\0"` makes `["a", "b\0c"]` and `["a\0b", "c"]` produce the same cache key. Both contain two strings, so the row-count assertion passes and the second request receives incorrect embeddings.

**Recommendation:** hash an unambiguous serialized sequence, such as JSON, or length-prefixed fields.

### 13. Candidate pruning lacks a consistent tie rule

**Classification:** reproduced reproducibility issue.

**Location:** [tree.py:376](../../xmr4el/hierarchy/tree.py#L376).

With four equal leaf scores, requesting one candidate returned label 3; requesting four candidates and then `topk=1` returned label 0. Stable sorting downstream cannot recover candidates discarded by `argpartition`.

**Recommendation:** define a secondary ordering by label ID at pruning boundaries if reproducible tied rankings are required.

## Architecture, maintainability, and reusability

- Preprocessing, encoding, label representation, clustering, matching, and traversal are meaningfully separated. Preserve those boundaries.
- Persistence and configuration dispatch repeat across component wrappers. Their inconsistent validation contributes to findings 1, 4, and 11. Address those contracts before introducing a generic serialization framework.
- `prepare_layer()` explicitly restricts children to their cluster's positive rows. Missing out-of-cluster negatives are an implemented modeling choice requiring an experiment, not a confirmed coding defect.
- The public CSR output and saved search configuration support integration. Dependency metadata, fitted-state checks, and artifact identity currently weaken that integration.
- Every declared runtime dependency has a direct implementation use. No technically justified dependency deletion was found.

## Testing and checks

- **80 tests passed** in 53 seconds on the final observed checkout.
- Configured Ruff checks passed.
- Additional Ruff security, bugbear, performance, and complexity checks were inspected. Flags such as MD5 for deterministic splitting and immediately executed loop closures were not treated as defects.
- Wheel build and installed-package smoke check passed from a temporary source copy.
- Four diagnostic/screening self-checks passed.
- Actual encoder construction and forward passes passed with a local synthetic BERT using CLS and mean pooling.
- Shell syntax checks passed.

The important coverage gaps are the reproduced cases above, dependency-minimum installations, and real accelerator memory behavior. CI tests one Ubuntu environment; its installed-wheel smoke mocks transformer computation. See [CI configuration:5](../../.github/workflows/ci.yml#L5).

No specific vulnerability in the locked dependency set was confirmed. The Torch floor excludes CVE-2025-3000's affected range, according to [OSV](https://osv.dev/vulnerability/GHSA-rrmf-rvhw-rf47). A complete transitive advisory scan was not available.

Commands used for the final suite and configured lint checks:

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  .venv/bin/python -m pytest -q -p no:cacheprovider \
  --basetemp=/private/tmp/xmr4el-audit-final-pytest

ruff check --no-cache xmr4el scripts tests
```

Ruff was executed using an existing cached binary. The wheel was built offline from a temporary source copy with `uv build --wheel --no-build-isolation`; the installed smoke ran outside the repository with `XMR4EL_TRANSFORMER_CACHE=0`.

Additional passing self-checks:

```bash
.venv/bin/python scripts/diagnose_routing.py -selfcheck
.venv/bin/python scripts/diagnose_unseen.py -selfcheck
.venv/bin/python scripts/experiments/screen_features.py -selfcheck
.venv/bin/python scripts/baselines/pecos_compare.py selfcheck
```

## Top 5 improvements

| Rank | Improvement | Impact | Effort |
|---|---|---|---|
| 1 | Reject invalid numeric inputs and unfitted saves | Prevent silent model corruption and invalid artifacts | Small |
| 2 | Declare and test supported dependency minima | Make installation reliable in other repositories | Small–medium |
| 3 | Preserve PIFA dtype and bound dense allocations | Reduce memory pressure and avoid predictable allocation failures | Medium |
| 4 | Bind rerankers to artifact and label identities | Prevent silent cross-model misuse | Small |
| 5 | Strengthen annotation and split validation | Protect training supervision and experiment validity | Medium |

## Prioritized roadmap

1. **Correctness:** add focused regressions for nonfinite inputs, unfitted saves, empty contexts, and malformed annotations; fix those shared boundaries.
2. **Portability:** establish dependency minima, test minimum/locked installations, and record relevant runtime versions with artifacts.
3. **Artifact integrity:** use `tree_id` and ordered-label identity for rerankers; make failed loads identify the missing component precisely.
4. **Scale:** remove PIFA promotion, add allocation preflight checks, and block kNN similarity computation across training rows.
5. **Cleanup:** unify error behavior in existing dispatchers and consolidate duplicated aggregation/persistence logic only where those fixes require it.

**Not checked:** corpus contents, full-scale retraining, CUDA/MPS execution, Docker builds, or complete dependency vulnerability scanning.

**Remaining risk:** passing synthetic tests does not establish memory bounds or artifact compatibility across environments.

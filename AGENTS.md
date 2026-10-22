# AGENTS.md

## Project purpose

This repository implements an eXtreme Multi-Label Ranking approach to solve Entity Linking in huge dataset spaces.

The main goal of the codebase is to:
- train and evaluate entity linking systems over very large label spaces
- support modular experimentation with featurization, clustering, candidate generation, and ranking
- make it easy to swap components without rewriting the whole training loop

This project is research-oriented. Prefer clarity, traceability, and small safe changes over broad refactors.

---

## What this repository is expected to contain

When analyzing this repo, assume the main logic is organized around these areas:

- `xmr4el/data/readers.py`, `data/splits.py`: local readers and PubTator splitting
- `xmr4el/features/`: encoder, vectorizers, reduction, transformers, label embeddings
- `xmr4el/hierarchy/clusterers.py`: clustering backends and hierarchy construction
- `xmr4el/learning/classifiers.py`, `learning/matcher.py`: classifier backends and the matcher
- `xmr4el/learning/scoring.py`: mention-kNN label scores (`label_max_cos`)
- `xmr4el/hierarchy/node.py`, `hierarchy/tree.py`: tree node and hierarchical model (training, traversal, persistence)
- `xmr4el/xmodel.py`: the `XModel` API; `xmr4el/eval.py`: evaluation metrics

Scripts live under `scripts/`, pytest checks under `tests/`, configs under `configs/`,
saved trees and exports under `outputs/` (gitignored).

---

## How to reason about the pipeline

Unless the code clearly shows otherwise, interpret the pipeline in this order:

1. **Preprocessing**
   - raw training data is grouped by label
   - texts are treated as synonyms or alternative mentions of the same concept/label

2. **Featurization**
   - texts may be converted into sparse features, dense features, or both
   - dimensionality reduction may optionally be applied

3. **Label embedding construction**
   - grouped training texts are transformed into a label matrix
   - label embeddings such as PIFA may be derived from instance features

4. **Hierarchy construction**
   - labels are clustered into a tree for efficient large-scale prediction

5. **Training**
   - one matcher per node is trained within the hierarchical framework (label-level at the leaves)

6. **Inference**
   - the hierarchy is traversed top-down
   - promising branches are expanded
   - final label scores = leaf matcher probability x routing path probability (optionally x mention-kNN)

When documenting or modifying the repo, preserve this mental model unless the code contradicts it.

---

## Input data assumptions

Assume the training data is label-centric.

Typical expectations:
- one file enumerates labels
- one or more files contain training texts associated with label IDs
- grouped texts for the same label act like synonym sets or multiple positives for that label
- local inputs live under `data/` and `datasets/`
- it reads PubTator and grouped TSV + label files

Do not assume full document-level entity linking with mention spans unless such functionality is explicitly present in the code.

---

## What to inspect first

When asked to explain, debug, extend, or document this repository, inspect in this order:

1. `README.md`
2. `pyproject.toml`
3. `xmr4el/xmodel.py`, `hierarchy/tree.py`, `hierarchy/node.py`
4. `xmr4el/features/encoder.py`, `features/vectorizers.py`, `features/reduction.py`, `features/transformers.py`, `features/label_embeddings.py`
5. `xmr4el/learning/classifiers.py`
6. `xmr4el/hierarchy/clusterers.py`
7. `scripts/`, `tests/`

Before making claims, verify them against the implementation.

---

## Rules for analysis

- Distinguish clearly between:
  - what is implemented
  - what is inferred
  - what is planned or missing
- Do not invent APIs, scripts, CLI commands, config files, or features
- If behavior is unclear, say exactly which file/function needs inspection
- Prefer pointing to exact files and functions over giving generic advice
- When proposing architectural changes, explain how they affect:
  - preprocessing
  - feature generation
  - label embeddings
  - hierarchy construction
  - matcher behavior

---

## Rules for code changes

- Prefer the smallest change that preserves current abstractions
- Do not rewrite the whole pipeline unless explicitly requested
- Keep component boundaries modular
- Avoid coupling featurization, clustering, and matcher logic unnecessarily
- Do not add compatibility code for old saved trees or configs; they are retrained instead
- Add or update tests when changing behavior

If editing model behavior, identify whether the change belongs in:
- preprocessing
- feature extraction
- label embedding creation
- clustering
- matcher
- hierarchical traversal / inference

---

## Rules for README or documentation work

When writing documentation:
- explain the repository in terms of the pipeline stages above
- describe the purpose of each major folder
- separate current behavior from future ideas
- call out data format expectations explicitly
- mention important dependencies only when relevant to actual code behavior

Do not claim support for a use case unless the repo already implements it.

---

## Rules for experiment suggestions

When suggesting improvements, organize them into one of these buckets:

1. **Featurization improvements**
   - better dense encoders
   - better sparse features
   - feature fusion
   - dimensionality reduction changes

2. **Label representation improvements**
   - better grouping of positives
   - alternative label embeddings
   - hard negative strategies

3. **Hierarchy / clustering improvements**
   - clustering algorithm changes
   - balance constraints
   - tree depth / branching tradeoffs

4. **Matcher improvements**
   - candidate generation quality
   - recall@k improvements
   - latency/throughput tradeoffs

5. **Leaf scoring improvements**
   - better leaf scoring
   - score fusion
   - calibration
   - reranking quality

6. **Data improvements**
   - better train/dev/test construction
   - label cleaning
   - synonym normalization
   - leakage prevention

When possible, recommend minimal, testable experiments first.

---

## Expectations for issue triage

If asked why a result is poor, check these first:
- data preprocessing assumptions
- label grouping quality
- train/dev/test leakage
- feature dimensionality
- matcher recall bottlenecks
- leaf score fusion (path probability, mention-kNN)
- hierarchy quality
- dependency/runtime environment issues

Do not jump directly to model replacement before checking pipeline quality.

Then check these known error origins, in pipeline order (state in `STATUS.md`, evidence in `docs/pipeline.md`):
- label index <-> label name mapping: model indices follow sorted `Y` columns; `XModel.initial_labels`
  must match them. A permuted mapping looks exactly like "routes at chance on dev, fine on train"
- `-ds_len` selects the first N label groups (first-seen order for PubTator, sorted group ID for
  grouped TSV), not a random sample
- feature block balance (`"sapbert_char_context"` normalises each block before the concat)
- PIFA label embeddings built from mention + context
- child matchers trained without out-of-cluster negatives
- leaf scores are only comparable across leaves through the routing path probability
  (`HierarchicalMLModel.predict` always multiplies it in)

Measure before fixing: prefer eval-only runs on saved trees and `scripts/diagnose_routing.py`
over retraining, and always report the chance line next to routing numbers.

---

## Dependency awareness

This project depends on a Python package setup and includes ML/NLP/vector dependencies.
Treat environment-sensitive issues carefully, especially:
- Python version compatibility
- CPU vs GPU paths
- Torch compatibility
- transformer model availability
- large-memory operations

If a bug might be environment-specific, say so explicitly.

---

## Good response style for this repository

Preferred response style:
- first summarize what part of the pipeline is relevant
- then point to the exact files/functions to inspect
- then suggest the smallest useful change
- then mention one or two next experiments

Avoid vague answers like “improve the model” without identifying the layer where the change belongs.

---

## If asked about improving XMR4EL for biomedical entity linking

Focus on:
- label construction quality
- synonym grouping quality
- candidate recall
- leaf matcher supervision
- fair train/dev/test design

Keep the scope limited to training and evaluating the model on local files.
# CLAUDE.md

Read `AGENTS.md` for repository conventions and `README.md` for setup and usage.

## Scope

This repository trains and evaluates XMR models on local inputs in `data/` and
`datasets/`. Keep configuration, model code, persistence, local preprocessing,
evaluation, and regression checks focused on that workflow.

## Commands

```bash
uv sync --locked
.venv/bin/python test/xmr4el/test_train_pipeline.py --help
.venv/bin/python test/xmr4el/test_evaluate_pipeline.py --help
```

Training accepts PubTator by default, or grouped TSV with `-labels_path`.
Plain TSV requires `emb_flag` 1 (`.models/xmr4el_flag1_config.json`); PubTator uses 6 (base config).
The base configuration is `.models/xmr4el_base_config.json`.
Saved trees are under `test/test_data/saved_trees/`.

## Working rules

- The user runs corpus training/evaluation; provide commands instead of launching them.
- Small synthetic checks listed in `README.md` are safe to run directly.
- Keep `data/`, `datasets/`, and saved model artifacts intact.
- Start every session by reading `STATUS.md` § "Resume here"; it says what is in flight and what to
  do with pasted output. Update that block at the end of each step and before a chat reset.
- `docs/results.md` holds valid results only; `docs/results_archive.md` holds invalid history.
- Verify claims against code; old notes and pre-fix measurements may be superseded.
- Preserve the loader ordering guard, the sorted label order, leaf label-level matching,
  ranker warm-starting, small-cluster reassignment, and local data formats.

## Current context (2026-09-30)

- Out of scope: UMLS, the KB/Postgres layer, the old SapBERT/KRISSBERT wrappers — all removed.
  (User 2026-10-01: SapBERT as a mention encoder is allowed; see `STATUS.md`.) PubTator is
  only a local file format here.
- Label index `j` = column `j` of `Y` = row `j` of `Z` = `XModel.initial_labels[j]`, which is the
  binarizer's `classes_` (sorted; labels with no training text have no column). `load` sorts legacy
  first-seen lists and refuses a label/`Z` count mismatch; regression in `test/xmr4el/test_data_loading.py`.
  Before this fix eval scored dev against a permuted label list, so **every dev number in
  `docs/results_archive.md` is invalid**.
- Transformer embedding batches are reassembled in numeric order (`featurization_wrapper/transformers.py`).
  Before, every flag 2-4 training run over 10 batches had permuted rows: the flag-4 tree `15-58-50`
  and the old 4.79x/3.81x train-side routing figures are invalid.
- Any new code that maps indices to label names must use `initial_labels`, never input order.

## Priorities

`STATUS.md` owns the work order, the pipeline observations, and the interpretation rules. Do not
duplicate them here. `fused_predict` (`xmr4el/xmr/base.py`) is not dead code: leaf training calls it.

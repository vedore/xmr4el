# CLAUDE.md

Read `AGENTS.md` for repository conventions and `README.md` for setup and usage.

## Scope

This repository trains and evaluates XMR models on local inputs in `data/` and
`datasets/`. Keep configuration, model code, persistence, local preprocessing,
evaluation, and regression checks focused on that workflow.

## Commands

```bash
uv sync --locked
.venv/bin/python scripts/train.py --help
.venv/bin/python scripts/evaluate.py --help
```

Local machine: `.venv/bin/python`. Inside the Docker containers (`xmr4el.dockerfile`, `pecos.dockerfile`): `python3`.

Training accepts PubTator by default, or grouped TSV with `-labels_path`.
Plain TSV requires `"features": "tfidf"` (copy of the base config); PubTator uses `"sapbert_char_context"`.
Configs: `configs/xmr4el_base_config.json` (MedMentions), `configs/xmr4el_bc5cdr_config.json`.
Search defaults are the config's `predict_config`, saved with the tree; bare `evaluate.py` uses them.
Saved trees are under `outputs/saved_trees/`.

## Working rules

- The user runs corpus training/evaluation; provide commands instead of launching them.
- Small synthetic checks listed in `README.md` are safe to run directly.
- Keep `data/`, `datasets/`, and `outputs/` (saved trees, exports) intact.
- Start every session by reading `STATUS.md` § "Resume here"; it says what is in flight and what to
  do with pasted output. Update that block at the end of each step and before a chat reset.
- `docs/results.md` holds valid results only; invalid results are deleted, not marked.
- Verify claims against code; old notes may be superseded.
- No compatibility code for old saved trees or old configs; retrain instead.
- Preserve the loader ordering guard, the sorted label order, leaf label-level matching,
  small-cluster reassignment, and local data formats.

## Current context

- Out of scope: UMLS, the KB/Postgres layer, the old SapBERT/KRISSBERT wrappers — all removed.
  (User 2026-10-01: SapBERT as a mention encoder is allowed; see `STATUS.md`.) PubTator is
  only a local file format here.
- Label index `j` = column `j` of `Y` = row `j` of `Z` = `XModel.initial_labels[j]`, which is the
  binarizer's `classes_` (sorted; labels with no training text have no column). `load` refuses a label/`Z`
  count mismatch; regression in `tests/integration/test_label_mapping.py`.
- Transformer embedding batches are kept in memory in row order (`xmr4el/features/transformers.py`).
- Any new code that maps indices to label names must use `initial_labels`, never input order.

## Priorities

`STATUS.md` owns the work order and the interpretation rules; `docs/pipeline.md` holds the
implemented-behavior table, synthetic checks and pipeline observations. Do not
duplicate them here. Internal nodes set `fused_scores` from the matcher's cluster scores in
`MLModel.train`, and `prepare_layer` consumes them; score fusion for that step
(`fused_predict`) is future work (`STATUS.md` § Future work).

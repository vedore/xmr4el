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
uv sync --locked --extra tui && .venv/bin/python -m xmr4el_tui   # TUI over outputs/runs/
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
  SapBERT as a mention encoder is allowed. PubTator is
  only a local file format here.
- Label index `j` = column `j` of `Y` = row `j` of `Z` = `XModel.initial_labels[j]`, which is the
  binarizer's `classes_` (sorted; labels with no training text have no column). `load` refuses a label/`Z`
  count mismatch; regression in `tests/integration/test_label_mapping.py`.
- Transformer embedding batches are kept in memory in row order (`xmr4el/features/transformers.py`).
- Any new code that maps indices to label names must use `initial_labels`, never input order.
- Results comparability: `xmr4el_tui/services/comparison.py` `assess` only; evidence table in
  `docs/design/tui_architecture.md` §3a. Bump `xmr4el.eval.EVAL_PROTOCOL` when a metric definition changes.
  Metrics written before 2026-10-28 lack `eval_protocol` / `strings_sha256`: CAUTION, no deltas, re-evaluate.
- The runner re-hashes inputs after reading them (`check_inputs`); a changed input fails the run.

## Priorities

`STATUS.md` owns the work order and the interpretation rules; `docs/pipeline.md` holds the
implemented-behavior table, synthetic checks and pipeline observations. Do not
duplicate them here. Internal nodes set `fused_scores` from the matcher's cluster scores in
`MLModel.train`, and `prepare_layer` consumes them; score fusion for that step
at predict time is future work (`docs/future_work.md` item 5).

## Development workflow

Claude Code uses `/model opusplan`: Opus for planning and Sonnet for execution. Use this configuration without unnecessary manual model switching.

- For complex or cross-cutting changes, use Plan Mode to resolve architecture, tradeoffs, integration risks, and important edge cases before implementation.
- For small, well-scoped changes, implement directly without unnecessary planning overhead.
- Inspect only the relevant code, dependencies, and documentation. Do not repeat completed architecture audits.
- Reuse established patterns and extend existing functionality. Avoid speculative refactoring, duplicate abstractions, and unrelated changes.
- Keep plans concise and actionable: affected files, implementation approach, risks, and focused tests.
- During execution, follow the approved plan. Reconsider architecture only when new evidence reveals a material issue.
- Prefer focused tests and relevant regression checks. Expand test coverage when shared infrastructure changes or failures indicate wider impact.
- Preserve scientific correctness, reproducibility, configuration semantics, and existing TUI behavior. Never sacrifice correctness to reduce model usage.
- Report changes, tests performed, failures, and unresolved limitations accurately.

`AGENTS.md` defines repository conventions. `STATUS.md` owns current progress and priorities. Do not duplicate their contents here.
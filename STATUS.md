# Status and next steps

This file holds only: what is in flight, what to do next, working rules, interpretation rules.
Anything else lives elsewhere: history `docs/archive/status_log.md` (TUI log 2026-10-22..30 at its end);
open research problems `docs/future_work.md`; valid results `docs/results.md`; implemented behavior,
synthetic checks, pipeline observations `docs/pipeline.md`; commands `docs/results.md` § Commands.

## Resume here

Updated 2026-10-30. Branch `tui` (HEAD `b89cf23` + this STATUS commit), pushed to `origin/tui`. Untracked
`CHANGES_TO_TUI.md` (user's request note, done). `b89cf23`: `docs/design/tui.md` rewritten to built behaviour only (old
design in git at `58cdffd`; screens §5.1 Dashboard … §5.7 Batch), `tui_architecture.md` updated, docstring refs
remapped; README badges, mermaid diagram, screenshots `docs/images/tui_*.svg` (Textual `save_screenshot`, root `.`).

**Product state**
- Backend (`xmr4el/`, `scripts/`): trains and evaluates XMR trees on local PubTator / grouped TSV. Unchanged by the TUI work
  except F1 (below).
- TUI (`xmr4el_tui/`, extra `tui`): Dashboard, Run Monitor, Results, New Experiment (Dataset Setup, Model Builder,
  Review Batch, detached batch coordinator). Design `docs/design/tui.md`, code map `docs/design/tui_architecture.md`.
- Verification: full suite 469 expected (468 passed + `tests/features/test_text_encoder.py`, whose context fixture had rank 3 for 2 SVD
  components and failed since the first commit; fixed 2026-10-30, `tests/features` 19 pass; full suite not rerun after the fix).
  No real-corpus training was run for the TUI work.

**Committed fixes (report `docs/audits/code_quality_architecture_review_2026-10-30.md`)**
- F1 `HierarchicalMLModel.train` resolves `BalancedKMeans.DEFAULTS` before halving: omitted `kwargs`, `{}` and explicit
  `n_clusters=8` train the same hierarchy. Trees trained earlier with omitted defaults differ: retrain.
- F2 Review `fix()` refused while a tree is carried. F3 `app.open_review` returns to the Review below instead of stacking.
- F4 `Component.closed`: unknown kwargs for matcher/clustering block readiness.
- F5 removed unused `process.check/CheckResult/Launches`, several `runs.*` helpers, `Scanned.reaped`.
- `CHANGES_TO_TUI.md`: a launch ends the experiment (`app.batch_launched`): New Experiment screens close (kept for `c`),
  Batch Monitor opens over the Dashboard, `n` starts a new draft (`draft.launched`). tests/tui 315 pass.

**NEXT (in order)**
1. User: manual terminal/SSH and Linux checks in `docs/audits/new_experiment_step6.md`; one real batch via the TUI
   (`configs/xmr4el_tiny_config.json`, `datasets/ncbi_tiny`; build recipe in `README.md`).
2. Release PR open: lasigeBioTM/XMR4EL#1, branch `release` (`21bd5e5` on `8d2e213`, pushed to origin). Built in a
   scratch worktree by copying kept files from `tui`. Dropped: AGENTS/CLAUDE/STATUS/CHANGES_TO_TUI, docs/audits,
   docs/archive, docs/references, docs/results.md, run_*.sh, eq_/i1_/lc_/ncbi_eq_ specs, tiny config,
   pecos/scispacy dockerfiles+scripts, screen_leaf_scorer.py (needs pecos export) and the 2 pecos export tests.
   README rewritten for users. 469 passed, ruff clean. Later fixes: commit on `release`, push.
   `f988702` (pushed): the same docs/README/docstring change; release drops the audits links and `58cdffd` refs
   (not in its history). Release worktree: old scratch path (`git worktree list`); move it with
   `git worktree remove <path> && git worktree add ../xmr4el-release release`.

**Known limits**
- Coordinator SIGKILL: surviving runners are not recovered or killed (V1).
- Drafts are session-only. Batches launched before 2026-10-29 lack strings evidence: re-evaluate.
- Metrics written before 2026-10-28 lack `eval_protocol` / `strings_sha256`: Results shows CAUTION, no deltas.
- Not done: pairwise-Review scaling, event-history eviction (both scale-triggered).

**Open, not started (ask before):** Plan H items 5-7 (explicit device/threads, batches/compare, install check;
`docs/audits/tool_readiness_audit.md` § F); tie rule at `argpartition` (eval-only check).

**On pasted output:** read it against `docs/results.md` § Commands; record only valid results in `docs/results.md`
(invalid ones are deleted, not marked); update this block before a chat reset.

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

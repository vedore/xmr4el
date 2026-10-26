# Status and next steps

History: `docs/archive/status_log.md`. Open research problems: `docs/future_work.md`. Valid results: `docs/results.md`.
Implemented behavior, synthetic checks, pipeline observations, PECOS notes: `docs/pipeline.md`.
Commands: `docs/results.md` § Commands.

## Resume here

**Branches (2026-10-22):** lasigeBioTM/XMR4EL `main` = vedore/xmr4el `main` = orphan release `8d2e213` (only upstream branch). `tui` = `8d2e213` + one commit whose tree = old `repeat` tip `f54ae3a` (STATUS, AGENTS, audits, diagnostics, PECOS/scispaCy artifacts, full results.md, all tests). Old `repeat`/`release`/old `main` deleted locally and on the fork; their history is only in `../xmr4el-old-history.bundle` (`git clone ../xmr4el-old-history.bundle` to inspect). NEXT: TUI work on `tui` (Plan H item 8: CLI/TUI); ship to `main` by copying kept files, not merging dev-only files.

**TUI (branch `tui`; design `docs/design/tui.md`, code map `docs/design/tui_architecture.md`; Plan T dropped 2026-10-23).** Built: increment 1 (Runs, Run Overview/Log/Inputs, Compare; `tui` extra textual==8.2.8, CI/Docker `--extra tui`); increment 2 steps 1-5 (runner `-check`/`-run_dir`/bindings/exit 3, process identity + `stopped` + heartbeat, `events.jsonl` stages, liveness RUNNING/LOST/UNVERIFIED, Setup/Review/spawn in `xmr4el_tui/launch.py`); Setup rework after the user tried `n` (`517810c`: pickers for config/dataset/training file/split, arrow keys, ctrl+o spec picker, Trees screen `t` with `e` evaluate). pytest 150, ruff clean. Details per step: `git log` on `tui`.
**NEXT (user 2026-10-23): plan how the TUI looks** — a visual/UX review session of every screen (Runs, Run, Compare, Trees, Setup, Review; 80x24 and wider), decide changes with the user before building more. Render screens with a Textual `run_test` + `export_screenshot()` script and `qlmanage -t` to PNG (scratchpad), real `outputs/runs`.
**Agreed plan after that (user 2026-10-23):** (1) increment 2 step 6 Stop modal §5.9 (`s`, focus Keep running, STOPPING, `k` force after 30 s; `process.stop` exists); (2) server check by the user: small real training via TUI in the container, watch stages, stop -> STOPPED, SSH drop + `docker exec` reattach, colour with/without `-e TERM -e COLORTERM`; (3) increment 3: `d` duplicate, previous-run timings (§5.2); (4) increment-1 gaps: Log search `/ n N`, older log on request, compact 60x18 / <60x16 (§5.10); (5) README TUI guide + ship to `main` by copying kept files (push/PR only on the user's OK).

**Open, not started (ask before):** Plan H items 5-7 (explicit device/threads, batches/compare, install check;
`docs/audits/tool_readiness_audit.md` § F); code-quality audit #13 tie rule at `argpartition` (eval-only check).

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

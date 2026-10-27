# Status and next steps

History: `docs/archive/status_log.md`. Open research problems: `docs/future_work.md`. Valid results: `docs/results.md`.
Implemented behavior, synthetic checks, pipeline observations, PECOS notes: `docs/pipeline.md`.
Commands: `docs/results.md` § Commands.

## Resume here

**Branches (2026-10-22):** lasigeBioTM/XMR4EL `main` = vedore/xmr4el `main` = orphan release `8d2e213` (only upstream branch). `tui` = `8d2e213` + one commit whose tree = old `repeat` tip `f54ae3a` (STATUS, AGENTS, audits, diagnostics, PECOS/scispaCy artifacts, full results.md, all tests). Old `repeat`/`release`/old `main` deleted locally and on the fork; their history is only in `../xmr4el-old-history.bundle` (`git clone ../xmr4el-old-history.bundle` to inspect). NEXT: TUI work on `tui` (Plan H item 8: CLI/TUI); ship to `main` by copying kept files, not merging dev-only files.

**TUI (branch `tui`; design `docs/design/tui.md`, code map `docs/design/tui_architecture.md`; Plan T dropped 2026-10-23).** Built: increment 1 (Runs, Run Overview/Log/Inputs, Compare; `tui` extra textual==8.2.8, CI/Docker `--extra tui`); increment 2 steps 1-5 (runner `-check`/`-run_dir`/bindings/exit 3, process identity + `stopped` + heartbeat, `events.jsonl` stages, liveness RUNNING/LOST/UNVERIFIED, Setup/Review/spawn in `xmr4el_tui/launch.py`); Setup rework after the user tried `n` (`517810c`: pickers for config/dataset/training file/split, arrow keys, ctrl+o spec picker, Trees screen `t` with `e` evaluate). pytest 150, ruff clean. Details per step: `git log` on `tui`.
**NEXT (user 2026-10-23): plan how the TUI looks** — a visual/UX review session of every screen (Runs, Run, Compare, Trees, Setup, Review; 80x24 and wider), decide changes with the user before building more. Render screens with a Textual `run_test` + `export_screenshot()` script and `qlmanage -t` to PNG (scratchpad), real `outputs/runs`.
**2026-10-27 look review, round 1 built and committed:** user chose full-width bars (Runs/Trees body = Horizontal, no `split`), compact Log lines, Review summary + `j` JSON (opens at top), thin scrollbars, short footers (no palette key; `/`, `y` hidden), short table states (DONE/RUN/UNVER/…), Setup placeholders = inherited search values, new draft takes newest training run's model_config. Recorded in `docs/design/tui.md` §4, §5.1, §5.5, §5.7. tests/tui 29 pass, ruff clean. Not touched yet: Overview Rows wrap / mention table past 24 rows; Trees "missing" said twice; help (Textual HelpPanel) still narrows the bars. Render script pattern: `run_test` + `export_screenshot` (fake `process.check` with all 4 search keys), `qlmanage -t -s 1600`.
**Agreed plan after that (user 2026-10-23):** (1) DONE 2026-10-27 (committed; tests/tui 30 pass): Stop modal §5.9 + `Stop run` button on Run screen; tiny dataset `datasets/ncbi_tiny` (awk recipe in README) + `configs/xmr4el_tiny_config.json`, `-check` ok, not trained yet. A full `bc5cdr_chemical` run (2026-10-27_11-16-56, started by mistake from Setup) was stopped by SIGTERM to its group → recorded `stopped`; the user's TUI predated the Stop button. Tiny run + Stop not tried yet. 2026-10-27 Dashboard built from the user's pasted prompt (committed `9db7a30`): `xmr4el_tui/dashboard.py` (Explorer: Models > dataset dir of the training file > tree > evaluation, "Active, no saved tree yet", "Runs without a tree"; read-only Description; Active runs table), model in `explorer.py`; moved trees matched by tree_id (`runs.trees`); sync = existing 5 s thread scan. Then (user) the old frontend was deleted (`screens.py`, `launch.py`, `widgets.py`, their tests; old code at `58cdffd`): the Dashboard is the only screen, backend modules kept; left/right/active enter only notify the Results/Monitor target until those screens are rebuilt. tests/tui 25 pass, full suite 146, ruff clean. 2026-10-27 Dashboard refinement (user's pasted prompt, committed): one `xmr4el` theme (no picker/settings file), centred XMR4EL header with the four areas, full-row cursor (explorer, active runs) and row-selectable read-only Description (OptionList), explorer 22 % resizable (`<` `>`, drag border, `ctrl+b` hide), short per-panel footer, `n` = New Experiment notice; round 2: explorer horizontal scrollbar back, cursor's branch guides stay accent, Description cursor row one colour, quit = `ctrl+c` twice (ctrl+q still works); round 3: mouse drag selects text in the three panels and copies it on release (OSC 52, no OS integration; works through docker exec); sync stays the 5 s scan (no watcher dep). Design: `docs/design/tui.md` §4 Header/Style, §7. tests/tui 30, ruff clean, all committed. NEXT: user tries it (`python -m xmr4el_tui`; if drag-copy pastes nothing, the terminal blocks OSC 52 clipboard writes), then the plan below from (2); (2) server check by the user: small real training via TUI in the container, watch stages, stop -> STOPPED, SSH drop + `docker exec` reattach, colour with/without `-e TERM -e COLORTERM`; (3) increment 3: `d` duplicate, previous-run timings (§5.2); (4) increment-1 gaps: Log search `/ n N`, older log on request, compact 60x18 / <60x16 (§5.10); (5) README TUI guide + ship to `main` by copying kept files (push/PR only on the user's OK).

**2026-10-27 architecture audit (`/private/tmp/xmr4el-architecture-audit.md`) implemented, NOT committed yet:** S2 `dashboard.py` split into `screens/dashboard_screen.py`, `screens/dashboard_details.py` (pure renderers), `widgets/{explorer,description,row_rendering,top_bar}.py`, `messages.py`; S3 tests/tui no longer import the backend (conftest string-target patch), process tests need no Textual, `python -S` import check; A1 one scan thread at a time, requests coalesced, cancelled scan publishes nothing, first scan after mount ("Scanning…"); A2 wrong-typed JSON fields read as absent + `Run.problems`, offset times converted to local naive, vanished run dirs skipped, unreadable outputs/runs keeps the last Snapshot + `Snapshot.problem`; A3 cache key = (name, mtime, size) of status/spec/tree/metrics, `ctrl+r` rereads all, tree locations canonical (abs/rel/`..`), moved tree joins its one existing location (copies stay missing); A4 `services/experiment_repository.py` (Snapshot, Activity, Details read in a worker, stale results dropped by seq); A6 events.jsonl tails read from offset (reset on truncate/replace); A5 stop needs live pid==pgid==sid, console log opened `x`, `process.Launches` owns handles/reap; O2 `forget_lines` on content replacement. Skipped: S4 (setup enumerators kept for New Experiment), O1/O3/O4. Full suite 170 pass, ruff clean, headless smoke on real outputs OK. NEXT: user reviews/commits (one commit or per step), then tries `python -m xmr4el_tui`.

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

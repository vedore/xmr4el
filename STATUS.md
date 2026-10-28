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

**2026-10-27 architecture audit (`/private/tmp/xmr4el-architecture-audit.md`) implemented, committed on `tui` (766e495..eaa1644, one commit per step, not pushed):** S2 `dashboard.py` split into `screens/dashboard_screen.py`, `screens/dashboard_details.py` (pure renderers), `widgets/{explorer,description,row_rendering,top_bar}.py`, `messages.py`; S3 tests/tui no longer import the backend (conftest string-target patch), process tests need no Textual, `python -S` import check; A1 one scan thread at a time, requests coalesced, cancelled scan publishes nothing, first scan after mount ("Scanning…"); A2 wrong-typed JSON fields read as absent + `Run.problems`, offset times converted to local naive, vanished run dirs skipped, unreadable outputs/runs keeps the last Snapshot + `Snapshot.problem`; A3 cache key = (name, mtime, size) of status/spec/tree/metrics, `ctrl+r` rereads all, tree locations canonical (abs/rel/`..`), moved tree joins its one existing location (copies stay missing); A4 `services/experiment_repository.py` (Snapshot, Activity, Details read in a worker, stale results dropped by seq); A6 events.jsonl tails read from offset (reset on truncate/replace); A5 stop needs live pid==pgid==sid, console log opened `x`, `process.Launches` owns handles/reap; O2 `forget_lines` on content replacement. Skipped: S4 (setup enumerators kept for New Experiment), O1/O3/O4. Full suite 170 pass, ruff clean, headless smoke on real outputs OK. NEXT: user tries `python -m xmr4el_tui` (first scan now lands after mount: brief "Scanning…"); push `tui` only on the user's OK; then the agreed plan from (2).

**2026-10-27 execution telemetry + View Logs (user's pasted prompt; committed on `tui`, not pushed):** runner-side
`xmr4el/telemetry/{events,resources}.py` (EventLog replaces the runner's EventFile: events schema v1 = `v`,
offset times, monotonic `duration_seconds`, `run` start/complete/failed/stopped, open stages closed failed/stopped;
eval parts data/predict/metrics/write via `evaluate_tree(events=...)`; `resources.json` per-stage CPU/RSS/GPU,
`-resource_interval` default 5, 0 = off; `logging.captureWarnings`); runner ignores a repeated SIGTERM or one after
the final status (`ENDING`); `process.stop` refuses runs whose status.json is already final. TUI: `logs.py` LogTail,
`widgets/log_view.py`, `screens/log_screen.py`; Description row `View Logs [Open]` (eval: Run/State; run: Type/State;
tree: Training run/Run state), enter opens, esc returns with the row kept. Docs tui.md §7, §8 (schema, resources,
logs, measured overhead), §9 (cancellation, state mapping), tui_architecture.md. Full suite 186 pass, ruff clean.
Follow-up (same day, same commit): optional extra `nvml` (nvidia-ml-py, in uv.lock and xmr4el.dockerfile);
NVML queried by device index (a bare `cuda` device initialised CUDA); `vram_peak_bytes` renamed
`torch_gpu_allocated_peak_bytes` (PyTorch allocator tensors, not process/device memory); `stop(force=True)` records
`stopped`/SIGKILL/`forced` after the process is verified gone; procinfo treats zombies as gone. Full suite 192 pass, ruff clean.
NEXT: push `tui` only on the user's OK; user tries View Logs (`python -m xmr4el_tui`) and a tiny run to see
resources.json on the server (`gpu_util` stays null without nvidia-ml-py).

**2026-10-28 Run monitor (user's pasted prompt; committed on `tui`, not pushed):** `screens/monitor_screen.py`
(RunMonitorScreen, by run id from Dashboard Active runs `enter` / Description `right`), `monitor_details.py`
(pure texts), `stop_dialog.py`; widgets `pipeline.py`, `monitor_panels.py` (RunOverview + Cancel Run,
ResourcePanel), `log_panel.py` (LogScreen's reader/search moved here; LogScreen and Live logs both use it),
`resize.py` (column clamp + border drag, Explorer too). `progress.EventTail` (one incremental events reader: the
repository's Activity and the monitor), hierarchy layers as stage parts. Fullscreen log = the Live logs panel
maximized (`f`), not a second reader. Cancel Run = `process.stop(run)` (graceful only) after StopDialog. Footer
scrollbar fix (`Footer { scrollbar-size: 0 0 }`). Docs: tui.md §5.11, §7; tui_architecture.md. Full suite 201,
ruff clean, headless read-only smoke on real outputs/runs OK. NEXT: user tries it (`python -m xmr4el_tui`, then a
run from a second terminal: tiny spec on stdin, or `-spec configs/experiments/ncbi_disease.json -resource_interval 1`);
push `tui` only on the user's OK. Cancel Run is offered only for runs leading their own group/session (TUI launches,
or shell launches via `setsid`; macOS: `python -c "import os,sys; os.setsid(); os.execvp(...)"` wrapper). Offered,
not decided: a pid-only SIGTERM fallback for shell runs (its workers would not get the signal); ask before building.

**2026-10-28 Run monitor audit (`/private/tmp/XMR4EL-run-monitor-audit.md`) fixed (not committed yet):** RM-01
Cancel Run needs a TUI launch record (`process.spawn` writes `.console/<id>.launch.json`; `process.owned`; `stop`
refuses a status.json identity changed since read) — shell runs, even under `setsid`, are now monitor only; RM-02
`fmt.clean` strips CSI/OSC and escapes control chars (log lines, run-file strings, event text); RM-03 runner writes
final status last, monitor/log reads coalesce ticks and stop only after a read started post-end; RM-04 non-list
`stages`, non-string scopes, non-finite numbers, impossible run-id dates read as missing; RM-05 zero-row eval ends
`failed`; RM-06 resource sample dropped if its stage ended while measuring; RM-07 log retention keeps a scrolled-back
view and search match; RM-08 overview max 50 % + scrolls; RM-09 LOST has no elapsed; RM-10 Pipeline prunes stale
rows; RM-11 README; R-01 reap in the scan thread, log path chosen in the log worker. Docs tui.md §9,
tui_architecture.md. Committed on `tui`, not pushed. NEXT: push only on the user's OK; try a TUI launch once New Experiment exists (Cancel Run is never
offered before that).

**2026-10-28 Results V1 + theme tokens (user's pasted prompt; committed on `tui`, not pushed):** `theme.py` = the
only hex values (semantic tokens, beam palette, `$xmr-*` CSS variables via `App.get_theme_variable_defaults`;
fmt/monitor/log/dashboard colours migrated; test forbids hex elsewhere). Results: `screens/results_screen.py`,
`results_details.py`, `widgets/results_panels.py`, `ResultsExplorer` in `widgets/explorer.py` (FullRow got a
`paint_row` hook, Explorer a `label` hook), `services/comparison.py` (VERIFIED / CAUTION / INCOMPATIBLE from test
sha256 + rows_total; population hash or §11 derivation for subset deltas; reuses compare.py helpers). Dashboard `r`
/ Description `left` open it (tree with several evaluations: no baseline chosen, tree expands). Docs tui.md §4,
§5.12, §7; tui_architecture.md. Full suite 220 pass, ruff clean; renders checked at 140x40 and 80x24 on real
outputs/runs. NEXT: user reviews (`python -m xmr4el_tui`, `r` on an evaluation); the user has changes to Results planned for the next session.

**2026-10-28 Results V1 refinement + shared Explorer (user's pasted prompt; committed on `tui`, not pushed):**
shared `Explorer` widget (Dashboard and Results): trees on two rows (`name · SapBERT D2` from the training run's
model_config `features`/`depth`, read by the scan, `model_config.json` in the cache key; then `Oct 28 · <short id>`,
`explorer.short_ids` grows past 8 chars on collisions), gap rows above later trees/datasets/groups, whole item is
the click target, `/` search (cross-level words, temporary opens, esc clears first). Results: Summary compact
(metrics by the name, aligned, `2 ⚠` role, rule + Caution notes, no instructions); `Δ —` for withheld deltas;
Specs = 5th group (search keys + features, out of Efficiency); esc: right panels/filter → Explorer → Dashboard;
Show incompatible filter (off per session; VERIFIED/CAUTION shown, INCOMPATIBLE/unknown sha hidden, per evaluation).
Tokens `delta-unavailable`, `spec-differs`, `divider`. Docs tui.md §5.12, §7; tui_architecture.md. tests/tui 99
(new `test_explorer.py`), full suite 229, ruff clean; renders 140x40 / 80x24 on real outputs/runs checked.
Δ legend shortened, one blank row above and below it. NEXT: push `tui` only on the user's OK.

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

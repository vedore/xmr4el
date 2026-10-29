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
Δ legend shortened, one blank row above and below it. Then (committed): Inspection label column = longest group label + 3 (`LABEL_WIDTH`, was 26: "seen, >1 label · dictionary" pushed its value right); test `test_inspection_values_share_one_column`. NEXT: push `tui` only on the user's OK.

**2026-10-28 Navbar, notifications, contextual footer (user's pasted prompt; committed on `tui`, not
pushed):** `widgets/top_bar.py` = navbar (crumbs from the screen stack via each screen's `crumbs`, Results adds
Inspect; click an earlier crumb = `App.go_to`; title centred; bell + unseen); `services/notifications.py` (session
history 50, first scan silent, ACTIVE -> COMPLETED/FAILED/LOST/STOPPED, key (run id, state), reaped launches share
the LOST key); `screens/notification_center.py` (ModalScreen box, `ctrl+n`, `c` clear all, enter/click opens:
one evaluation with metrics -> Results with it as baseline, else Run Monitor; opened from the Dashboard);
`widgets/context_footer.py` (Footer + note; state on App: `say` 3.5 s per screen, quit confirmation ctrl+c / esc).
Every screen-level `notify` moved to `say`; toasts only for lifecycle (top right, one per scan, 4 s). Theme tokens
`overlay`, `tint-*`, `confirm-row`, `theme.chip`. Docs tui.md §4 Navbar / Three message layers, §7; tui_architecture.md.
tests/tui/test_notifications.py (16); full suite 245 pass, ruff clean. Follow-up (user): no emoji (bell = text
`Notifications`), blank row under the navbar, runs started this session and first seen ended (tiny run, < 5 s) are
notified too. Then symbols to plain text: Caution `!`, inspect `i`, incompatible `x`, stage markers `+` `>` `x`.
NEXT: push `tui` only on the user's OK.

**2026-10-28 Results V1 audit (R01-R11 + P2; audit file deleted by the user after the fixes) fixed (committed):** producer
`xmr4el/eval.py`: `eval_protocol` (EVAL_PROTOCOL = 1), `strings_sha256` (per kept row: position, train labels of
its string, dictionary answer), acc@1_all = 0 with no row in vocabulary, empty string groups null rates; runner
re-hashes inputs after loading / each eval (`check_inputs`: changed = run fails, no metrics). TUI: deltas need equal
recorded protocol (missing = CAUTION), Generalization needs equal strings hash, metrics/spec path or split conflict =
INCOMPATIBLE, non-string hashes/commits = not recorded; provenance.json/git.json in the scan signature (`Run.files`);
search mounts only matches; vanished baseline marked, vanished inspection says so; Summary max 40 %, analysis
table min-height 6; same-minute evaluation labels fall back to the run id; member rows show their role number.
Follow-up: hashes must be 64 lowercase hex (commits 40/64) or count as not recorded (fixtures hash labels via
`fake_runs.sha`); Results `ctrl+r` rereads facts too; legacy `xmr4el_tui/compare.py` + `tests/tui/test_compare.py`
deleted (helpers moved into `services/comparison.py`); README Results line; tui_architecture.md updated.
Every existing metrics file lacks `eval_protocol` / `strings_sha256`: Results shows CAUTION, no deltas, until
re-evaluated (no compatibility code). Docs tui.md §5.12, §11; evidence reference tui_architecture.md §3a; rules in
AGENTS.md (bump EVAL_PROTOCOL) and CLAUDE.md. Full suite 251 pass, `uvx --offline ruff check xmr4el xmr4el_tui scripts
tests` clean. Left on purpose: caching repeated metadata reads in read_facts (0.1 s / 1,000 evals). All of this is
committed on `tui`. NEXT: user re-evaluates the trees they compare.

**2026-10-28 TUI application UX audit (`docs/audits/tui_application_ux_integration_audit_2026-10-28.md`,
F01-F11 + R01) fixed (committed on `tui`, not pushed):** notices keyed by run
incarnation (`notifications.incarnation`: id + recorded start + pid/start token), targets carry it and `open_notice`
rejects a replaced run; LOST provisional (repository re-reads a run found gone: a final status written meanwhile
wins; the manager keeps watching LOST/UNKNOWN and a recorded outcome replaces the interruption notice); STOPPED =
"Run cancelled" only for this TUI's Cancel Run (`notices.requested`), else "Run stopped · Stopped by <signal>";
first-scan cutoff = scan start (`Snapshot.at`) to the second; `sent`/`watched` pruned to existing run dirs;
center rows keyed by notice identity (selection and scroll kept on arrivals), list height fits the terminal;
notification destinations pushed over the current screen (comparison kept); one toast at a time, lines ≤ 88 chars,
max 6 rows, opening the center clears it; HelpPanel docked (not split) under the navbar (`App.fit_help`), bell
falls back to `Notif`; `n` hidden from the footer and the empty Dashboard says New Experiment is planned. Docs tui.md
§4 + key table, tui_architecture.md. Full suite 269 pass, ruff clean. NEXT: push `tui` only on the user's OK; user may
delete the audit file.

**2026-10-28 New Experiment step 2, Dataset Setup (user's pasted prompt; committed on `tui`, not pushed):** `n` (Dashboard footer)
opens `screens/dataset_setup.py` (Datasets 30 % | Dataset Configuration); `datasets.py` (stdlib: discovery from
`configs/experiments/*.json` merged per directory + `runs.datasets/split_files/training_files`, registry
`outputs/tui/datasets.json`, 64 KiB PubTator sniff, validation, statuses); `draft.py` (`app.draft`: selected, per-dataset
edits, `confirmed` Selection for the next steps, `trees` placeholder); `screens/file_picker.py` (path field + entries,
filter, typed/pasted path, esc cancels); footer confirmations generalized (`App.ask` / `asked`, used by `u` discard);
`widgets/description.Rows` shared OptionList base. Docs tui.md §5.13 + keys, tui_architecture.md, README.
tests/tui/test_dataset_setup.py (20); full suite 288 pass, ruff clean; renders 140x40 / 80x24 checked. Not built:
Model Builder, Review Batch, launching. NEXT: user tries `python -m xmr4el_tui`, `n`; then Step 3 (Model Builder) reads `app.draft.confirmed`.

**2026-10-29 New Experiment step 3, Model Builder + full-width footer feedback (user's pasted prompt; committed on `tui`, not pushed):**
Dataset Setup `m` (or its `Model Builder` row) pushes `screens/model_builder.py` (tree tabs + `widgets/pipeline_grid.py`
grid 35 % | Library or Editor = Attributes 40 % + JSON preview 60 %). `components.py` (stdlib registry: 8 slots,
components the backend loads, fields with backend defaults/ranges, parse/check/assign, `json_lines` by top-level key,
`template` from the spec of the confirmed training file else base config, tfidf refused); `draft.py` `Tree` (id, name,
config) + add/duplicate/remove/move/rename/apply. Editor = copy of the slot's keys; Apply validates and assigns once;
esc/slot/tree/breadcrumb with unapplied changes ask in the footer (`u` discards; `App.go_to` `leave_guard`,
`App.action_escape` `confirm_cancelled`). Footer: feedback now replaces all shortcuts over the whole bar on new token
`feedback-row` (confirm > feedback > shortcuts). Tokens `block`, `section`. Docs tui.md §4, §5.14, keys;
tui_architecture.md; README. Tests: tests/tui/test_model_builder.py (17), test_footer.py (3), tests/test_tui_components.py
(20, backend cross-check + load_config round trip). Full suite 328 (one run had the timing-sensitive
`test_footer_feedback_replaces_toasts_and_expires` fail under load; passes alone and in two tests/tui runs), ruff clean.
NEXT: user tries `python -m xmr4el_tui`, `n`, confirm, `m`; push only on the user's OK; then Step 4 Review Batch (runner
`-check` per tree = model-level readiness; materialize model_config files there, not in the builder).

**2026-10-29 Model Builder UX refinement (user's pasted prompt; not committed):** Model Builder 45 % / lower 55 %
(`app.tcss`); Tab/Shift+Tab stay inside the focused panel (`action_local`, panels top = tabs + grid, library, attrs
panel = fields + help + Apply/Cancel, json), ctrl+tab / ctrl+shift+tab and F6 / shift+F6 cycle visible panels at each
panel's last focus (`] [` removed); up/down move between parameters (`action_field_step`, `AttrSelect` collapsed),
help box under the list from `Field.help` + `components.json_path` / `constraints` (all fields have help, test);
Tree Actions menu (`a`, `Actions` button; Rename/Duplicate/Move/Add/Remove; remove confirms with `y`, `n` adds, Delete
optional); `pipeline_grid.py` `Geo` per zoom level (compact/normal/expanded/fit via `+ - 0` and `ZoomBar`), model
centred by margins, `reveal` offset by them. tests/tui 184 pass, ruff clean; renders at 140x40 / 100x30 checked.
Limit: ctrl+tab needs a terminal with the kitty keyboard protocol (F6 otherwise). NEXT: user tries it; commit/push on OK.

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

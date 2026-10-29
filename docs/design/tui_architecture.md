# XMR4EL TUI code architecture

Status: implemented, Dashboard, Run monitor and Results (2026-10-28). Implements [tui.md](tui.md); section
numbers `§n` refer to it. §1-§6 describe the current code; §7-§10 are the build history. New Experiment is
planned: its key only says what it would open.

## 1. Rules

- `xmr4el_tui` never imports `xmr4el` (torch at import, §10). It reads run files and
  spawns `scripts/run_experiment.py`.
- Textual is imported only by `app.py`, `messages.py`, `screens/` and `widgets/` (the old
  Runs/Run/Trees/Compare/Setup/Review screens were removed 2026-10-27; old code in git at `58cdffd`).
  Everything else is stdlib-only (`tests/tui/test_runs.py` imports it under `python -S`), pure where
  possible, and tested without a terminal.
- Every file read goes through `runs` / `progress` / `logs` / `services/experiment_repository.py`; renderers in
  `screens/dashboard_details.py` and `screens/monitor_details.py` read no file. The UI thread reads no file either: scans and the selected
  item's Details run in thread workers.
- Run directories are the only durable state. The app holds the last Snapshot and the Launches
  (Popen handles of children it started); nothing else.
- Every file read tolerates missing or corrupt input, and fields of the wrong JSON type, and reports it as
  data (`Run.problems`, `Details.problems`, `Snapshot.problem`), never as an exception into the UI.
- Imports at the top of each module.

## 2. Layout

```text
xmr4el_tui/
  __init__.py      empty (no Textual import: the runner imports procinfo without the extra)
  __main__.py      argparse (--root), App(root).run()
  procinfo.py      process identity, stdlib only; shared by the runner and the TUI
  runs.py          Run/Eval/Tree records; tolerant reads; scan cache by file signature; canonical tree
                   locations; state_word, elapsed; spec normalization
  progress.py      events.jsonl -> stage list; current_stage(run, events), last_stage_event(events)
  logs.py          LogTail: incremental, bounded run.log reads (levels, torn lines, truncation); log_path
  fmt.py           durations, timestamps, metrics, counts, stage markers, state colours
  theme.py         the only colour values: semantic tokens (DARK), beam_color, $xmr-* CSS variables
  components.py    Model Builder registry (stdlib): slots, components, attribute fields (backend defaults and
                   ranges), parse / check / assign, slot problems, summaries, JSON lines by top-level key, template
  process.py       -check, spawn, derived state, stop; Launches (owner of started children)
  explorer.py      explorer model: trees by dataset -> evaluations, runs without a tree; keys, labels, tree
                   descriptor / detail / collision-safe short id, prune (search, Results filter)
  messages.py      RunsChanged (the app rescanned)
  app.py           App: serialized, coalesced scan worker every 5 s; Snapshot; Launches reaping; global keys, theme
  services/
    experiment_repository.py  Snapshot (runs, trees, Activity per recorded-running run) with the read cache
                   and incremental events tails; Details of the selected item
    notifications.py  session notifications: run-state transitions -> Notice (by incarnation; LOST provisional; dedup, bounded, unseen)
    comparison.py  Results: Facts per evaluation (hashes, preprocessing, commits), assess -> Compat
                   (VERIFIED / CAUTION / INCOMPATIBLE + reason), metric groups and allowed deltas
  screens/
    dashboard_screen.py   DashboardScreen + ActiveRuns; Details loaded in a worker; open_run -> Run monitor
    dashboard_details.py  pure Rich texts: banner, metrics, tree / evaluation / run / group descriptions
    log_screen.py  LogScreen: fullscreen read-only log (View Logs): a LogPanel
    monitor_screen.py     RunMonitorScreen (§5.11): telemetry worker (EventTail + resources.json when changed),
                   stage selection -> resources, Cancel Run -> process.stop in a worker, layout / panel keys
    monitor_details.py    pure Rich texts: overview, stage rows, resource summaries (resources.json scopes)
    stop_dialog.py StopDialog: Cancel Run confirmation (names the run id)
    notification_center.py  NotificationCenter: ModalScreen box under the bell (history, open, clear all)
    results_screen.py  ResultsScreen (§5.12): baseline + challengers, facts worker, inspection, group keys
    results_details.py pure Rich texts: baseline line, summary, analysis cells, inspection
    dataset_setup.py   DatasetSetupScreen (§5.13); file_picker.py FilePicker
    batch_monitor.py   BatchMonitorScreen, BatchesScreen (§5.16): queue + selected tree's log, Stop Batch
    model_builder.py   ModelBuilderScreen (§5.14): tree tabs + grid, Library, editor (Editing copy), preview()
  widgets/
    explorer.py    Explorer tree (resizable; items of 1-2 rows + gap rows over Tree lines; `/` search) + node
                   labels; ResultsExplorer (row states, beam chip, `i` click); one widget for both screens
    results_panels.py  BaselinePanel, ComparisonSummary, DetailedAnalysis (DataTable + notes), InspectionView
    description.py Description (read-only OptionList rows)
    row_rendering.py  FullRow cursor and Selectable (mouse text selection) mixins
    top_bar.py     TopBar: navbar (breadcrumbs from the screen stack, centred title, bell)
    context_footer.py  ContextFooter: Textual Footer, or feedback / confirmation over the whole bar (state on the App)
    pipeline_grid.py  PipelineGrid (Model Builder slots, branch navigation, click regions), TreeTabs
    log_view.py    LogView: virtual-rendered log lines, level filter, search highlight
    log_panel.py   LogPanel: one LogTail polled in a worker while active, header, search box (LogScreen and the
                   Run monitor's Live logs)
    pipeline.py    Pipeline: stages / parts / layers as a Tree updated in place; follows the current stage
    monitor_panels.py  RunOverview (+ Cancel Run button), ResourcePanel
    resize.py      column_width clamp + DragRightBorder (Explorer, Run monitor left column)
  app.tcss
tests/tui/
  fake_runs.py     builder for run directories (shapes copied from real runs)
  test_runs.py  test_progress.py  test_logs.py  test_procinfo.py  test_process.py
  test_dashboard.py  test_monitor.py  test_results.py  test_notifications.py  test_dataset_setup.py
  test_model_builder.py  test_footer.py  (Textual run_test + Pilot; importorskip("textual"))
tests/test_tui_components.py  the builder registry against the backend (defaults, rejections, load_config round trip)
```

Runner side (imports xmr4el, never imported by the TUI): `xmr4el/telemetry/events.py` (EventLog: the
events.jsonl handler, durations, open stages closed on failure / stop) and `xmr4el/telemetry/resources.py`
(ResourceMonitor: per-stage resources.json), tui.md §8; tests in `tests/test_telemetry.py`.

Split by responsibility, not size: a widget used by a second screen moves to `widgets/`, a message handled
by more than one screen to `messages.py`, file reads to the repository. No base screen or generic layer
before a second consumer exists.

## 3. Data model (`runs.py`)

```python
@dataclass(frozen=True)
class Eval:
    split: str
    test_path: str
    train_path: str | None
    overrides: dict           # search keys given in the spec
    metrics: dict | None      # metrics_<split>.json, None = not produced yet

@dataclass(frozen=True)
class Run:
    id: str                   # directory name, e.g. 2026-10-20_13-36-20_ncbi_disease
    path: Path
    name: str                 # spec name
    kind: str                 # "train" | "eval"  (spec.json xmodel_path null -> train)
    recorded: str             # status.json status, or "unknown"
    started: datetime         # status.json started, else parsed from id
    finished: datetime | None
    error: str | None
    tree_id: str | None       # tree.json
    tree_path: str | None
    tree_here: bool           # <tree_path>/xmodel.pkl exists
    evals: tuple[Eval, ...]   # spec order
    identity: dict | None     # pid/host/uid/start token/... (§9), None for old runs
    problems: tuple[str, ...] # "metrics_dev.json: invalid JSON", ...
    live: object = None       # process.Live, set by the app's scan for recorded-running runs
    train_path, features, depth  # train run: spec train_path, model_config features/depth (explorer descriptor)
    files: tuple = ()         # scan signature; part of equality, so a rewritten provenance/git makes the Run differ
```

`spec`, `provenance`, `git` and `model_config` are read lazily by `Run` helpers (`run.spec()`, ...; a
JSON object or None) for Results (`services/comparison.read_facts`); the Dashboard reads them once per
selection through `ExperimentRepository.details`.

Functions:

| Function | Purpose |
| --- | --- |
| `scan(root, cache) -> list[Run]` | Read every non-dot dir under `outputs/runs/`; reuse a cached `Run` while the (name, mtime, size) of its `RUN_FILES` (status, spec, tree, model_config, provenance, git) and metrics files are unchanged (`signature`, stored in `Run.files`); a dir vanishing mid-read is left out. Newest first. |
| `trees(root, runs) -> list[Tree]` | Group runs by `canonical(root, tree_path)`; a missing location joins the one existing location of its tree_id (several = copies: stays missing). |
| `same_tree(runs, run) -> list[Run]` | Other runs with the same `tree_id` (details `also:`). |
| `training_run(runs, tree_id) -> Run \| None` | The `kind == "train"` run that saved this tree (§11 derivation, eval-only defaults). |
| `previous_completed(runs, run) -> Run \| None` | Newest COMPLETED run with the same name, older than `run` (§5.2 timings). |
| `normalize_spec(recorded) -> (spec, xmodel_path)` | §10 round trip: drop `xmodel_path` from the copy, return it separately. |
| `command_line(spec, xmodel_path) -> str` | Equivalent CLI for Inputs (`-spec -` with the normalized JSON). |

Scan cost: about 5 small JSON reads per changed run; unchanged runs cost one directory listing.
Rescan every 5 s and on `ctrl+r` (rereads every file), in one Textual thread at a time: a request during a
scan makes that scan go once more. `events.jsonl` of recorded-running runs is read from the last offset
(again from 0 when truncated or replaced).

## 3a. Comparison evidence (`services/comparison.py`)

The only place that decides comparability is `assess(base, other)`; `read_facts` gathers what it needs off the UI
thread. Each piece of evidence proves one thing and gates specific deltas; none implies another.

| Evidence | Recorded in | Proves | Missing | Different |
| --- | --- | --- | --- | --- |
| metrics `test_path` / `train_path` / `split` vs spec eval | metrics, spec.json | the metrics belong to that eval (`Facts.conflict`) | — | INCOMPATIBLE |
| test-file sha256 | provenance.json `inputs` (re-checked after evaluation by the runner) | same dataset bytes | INCOMPATIBLE | INCOMPATIBLE |
| `rows_total` | metrics | same loaded rows | INCOMPATIBLE | INCOMPATIBLE |
| `eval_protocol` | metrics (`xmr4el.eval.EVAL_PROTOCOL`) | same metric definitions | CAUTION, no deltas | note, no deltas |
| `population_sha256` (`[row position, gold]` kept) | metrics | same in-vocabulary rows | §11 derivation: tree training-file sha256, preprocessing, one clean commit; else CAUTION | note, no acc@1 / MRR / R@k / R@cand / hybrid deltas |
| `strings_sha256` (`[row position, #train labels of the string, dictionary answer]` per kept row) | metrics (only with an eval `train_path`) | same seen/unseen groups and dictionary (vocabulary, abbreviation expansion and reference file all included) | CAUTION, no Generalization deltas | note, no Generalization deltas |

- Hashes count only as 64 lowercase hex characters, commits as 40/64 (`evidence`); anything else is "not recorded".
- `allowed(flag, compat)` maps a row's flag (`all`, `population`, `strings`, `hybrid`, `hybrid_all`) to these
  results; every delta also needs `compat.protocol`. A withheld delta is `Δ —`; a missing or `null` value is `N/A`
  and gets no delta (so an empty string group never gets one).
- Metrics files written before `eval_protocol` / `strings_sha256` existed (every run before 2026-10-28) give
  CAUTION without deltas. Re-evaluate the tree; there is no fallback (no compatibility code).
- Change a metric definition in `evaluate_tree` (formula, denominator, filtering, tie rule) → bump
  `EVAL_PROTOCOL`, so old and new values are never subtracted.
- Facts reload when the scan's `Run` tuple changes (provenance/git are in the signature) and on Results `ctrl+r`.

## 4. Other pure modules

**`progress.py`**

```python
@dataclass(frozen=True)
class Stage:
    key: str                  # load, encode, labels, hierarchy, save, eval:<split>
    state: str                # done | current | pending | failed
    seconds: float | None
    detail: str               # "layer 1/2 · node 1/1 · 474,854 rows", "cache hit", ...
    parts: tuple["Stage", ...] = ()   # encode blocks (§5.2)
    bar: tuple[int, int] | None = None  # (done, total) distinct mentions

def read_events(path, offset=0) -> tuple[list[dict], int]   # incremental tail, skips a torn last line
def stages(events, kind, splits, recorded) -> list[Stage]
def last_event_age(events, now) -> float | None
```

Without `events.jsonl`, `stages` returns `[]` and the Overview says the stage list is
unknown (§8).

**`fmt.py`**: `duration(s)`, `stamp(dt, long=False)`, `metric(x)`, `count(n)`,
`pct(a, b)`, `marker(state, ascii)`. ASCII when `NO_COLOR` is set or the locale is not
UTF-8 (§4).

**`procinfo.py`** (stdlib only; increment 2):

```python
def identity(pid) -> dict | None     # uid, start token, runtime, pgid, sid; None if gone
def own_identity() -> dict           # the runner calls this for status.json
def same(recorded, live) -> bool
```

Start token: Linux `/proc/<pid>/stat` field 22; macOS `ps -o lstart= -p <pid>`.
Runtime: Linux `/proc/sys/kernel/random/boot_id` + `readlink /proc/self/ns/pid`;
macOS `sysctl -n kern.boottime`. PGID/SID via `os.getpgid` / `os.getsid`.

**`datasets.py`** (stdlib; Dataset Setup, tui.md §5.13): `discover(root)` -> `Dataset` (key = directory or
`local:<id>`, name, variant, `Candidate`s per role train / dev / test with their sources); `registered` / `register`
(`outputs/tui/datasets.json`); `inspect(root, path)` (stat + 64 KiB PubTator sniff, cached by mtime/size);
`validate(root, dataset, selection, taken)` -> `[(field, kind, message)]`; `status(problems)`. `Selection` (frozen):
key, name, variant, train, evals (`EvalInput` split / path / enabled), format.

**`draft.py`** (stdlib): `ExperimentDraft` on the app: `selected`, `edits` (per dataset key), `confirmed`
(`Selection`); Model Builder: `trees` (`Tree`: stable `id`, `name`, applied `config`, in execution order), `tree`
(shown), `slot` / `feature_slot` (grid cursor); `add_tree`, `duplicate` (deep copy), `remove` (last stays), `move`,
`rename` (`name_problem`), `apply` (one assignment of a validated config). Editor drafts never live here.

**`components.py`** (stdlib; Model Builder, tui.md §5.14): `SLOTS` (id, grid name, config key, components, extra
top-level fields), `Component` (type, fields, base config), `Field` (path, kind, backend default, range, empty =
omit / null / required). `installed`, `part` / `merged` (slot keys out of / into a config), `assign` (left-out
defaults stay out), `parse` / `check` / `coerce`, `slot_problems` / `problems`, `summary`, `json_lines` (each
`json.dumps(indent=2)` line with its top-level key), `template(root, selection)`. Mirrors the backend without
importing it; `tests/test_tui_components.py` (outside tests/tui) holds the two together.

## 5. UI modules

| Class | Shows | Reads |
| --- | --- | --- |
| `App` | help panel, theme; notifications (`Notifications.observe` per scan with the scan's start time, one toast at a time, `open_notice` pushes over the current screen after an incarnation check); footer state (`say`, quit confirmation); `go_to` (breadcrumb click) | `ExperimentRepository.snapshot` every 5 s in a worker -> `App.snapshot`, posts `RunsChanged` |
| `RunMonitorScreen` | one run by id: overview, pipeline, resources of the selected stage, live log | `App.snapshot` (state); `progress.EventTail` + resources.json (stat, reread on change) every 1 s in a worker while active, one read after it ends; provenance.json once; the log via `LogPanel` |
| `DashboardScreen` | explorer (Models > dataset > tree > evaluation; Active, no saved tree yet; Runs without a tree), description of the selected item, active runs | `App.snapshot`; `explorer.build`; `ExperimentRepository.details` of the selected item in a worker (xmodel.json, no pickle) |

| `ResultsScreen` | baseline + up to 3 challengers: summary, one metric group for all, inspection of one | `App.snapshot` (metrics); `comparison.read_facts` for every evaluation in a worker when the scan's runs change |

| `DatasetSetupScreen` | New Experiment step 1: Datasets (`DatasetList`) and Dataset Configuration (`ConfigRows`, a `Rows` OptionList like Description); `FilePicker` pushed per file row; `m` pushes the Model Builder | `datasets.discover` + registry on mount / `ctrl+r`; list statuses in a worker; `app.draft` |

| `ModelBuilderScreen` | New Experiment step 2: `TreeTabs`, `PipelineGrid`, `Library` (a `Rows`), editor (`Editing`: slot, component, origin, working copy, parse errors) with Attributes and `JsonPreview`; `leave_guard` / `confirm_cancelled` hooks used by `App.go_to` / `App.action_escape` | `app.draft` (confirmed dataset, trees); `components.template` reads specs / base config on a new tree |

`DashboardScreen.open_results(key)` pushes Results with that evaluation as baseline. `open_run(run_id, split)`
pushes the Run monitor. Footer confirmations other than quit: `App.ask(question, key, verb)` / `App.asked(key)`
(the same key again confirms, `esc` cancels, `ctrl+c` replaces it with the quit question). Used by Dataset Setup:
`runs.datasets/split_files/training_files` (via `datasets.discover`). Backend kept for the launch steps:
`process.check/Launches/rejection`, `runs.normalize_spec/command_line/model_configs`.

Screens receive `RunsChanged` and re-render from the new `Run`, keeping selection,
scroll and marks by run `id` (§7).

## 6. Process side (`process.py`, increment 2)

```python
def check(root, spec, xmodel_path, revision) -> CheckResult   # spawns -check, parses one JSON object
def spawn(root, spec, run_dir, xmodel_path, config_sha, tree_id) -> Popen  # + .console/<id>.launch.json
def owned(run, rec) -> bool                   # the launch record matches run id, run dir and identity
def derived_state(run, here) -> Live          # RUNNING | LOST | UNVERIFIED; stoppable only when the live
                                              # process leads its group and session (the recorded flag is no
                                              # proof) and owned() (a TUI launched it)
def stop(run, force=False) -> str             # re-verifies identity first (§9); returns outcome
class Launches                                # start() = spawn + keep the handle; reap() -> [(run id, what
                                              # happened)] for exits without a final status (exit 3: rejection)
```

- `check` runs in a thread worker; the app ignores results whose `revision` is not
  the draft's current revision (§5.5).
- `spawn`: argument list, `cwd=root`, `start_new_session=True`, stdin = spec JSON,
  stdout/stderr = `outputs/runs/.console/<run-id>.log`, created exclusively (a reused run id raises
  `FileExistsError` before anything starts). The Popen goes into `App.launches` only so `reap()` collects it
  and reports a launch that wrote no final status; the run's state always comes from files.
- Exit code 3 + the JSON line in the console log -> `BindingRejected` -> Setup (§10).

## 7. Changes outside the package

| Increment | File | Change |
| --- | --- | --- |
| 1 | `xmr4el/eval.py` `evaluate_tree` | `population_sha256` (sha256 of the JSON list of `[row position, gold label]` kept by the vocabulary filter) and `preprocessing` (`context_window`, `abbrev_expansion`) in the metrics. No metric value changes. |
| Results audit | `xmr4el/eval.py` `evaluate_tree` | `eval_protocol` (`EVAL_PROTOCOL`), `strings_sha256`; `acc@1_all = 0` when no row is in vocabulary; empty string groups store `null` rates (§3a). |
| Results audit | `scripts/run_experiment.py` | `check_inputs`: re-hash the training file after loading and each eval's files after evaluating; a change fails the run before metrics are written. |
| 1 | `pyproject.toml` | package discovery `xmr4el_tui*`, `[project.optional-dependencies] tui = ["textual==<pinned>"]`, `[project.scripts] xmr4el-tui`, package data `*.tcss`; `uv.lock`. |
| 1 | `xmr4el.dockerfile` | `--extra tui` in the dependency sync. |
| 1 | `.github/workflows/ci.yml` | `uv sync --locked --extra tui`; ruff over `xmr4el_tui`. |
| 2 | `scripts/run_experiment.py` | `-check`, `-spec -`, `-run_dir`, `-model_config_sha256`, `-tree_id`, exit 3; config captured once; identity in `status.json` (`procinfo.own_identity`); `stopped`; heartbeat thread; `events.jsonl` handler. |
| 2 | `xmr4el/xmodel.py`, `hierarchy/tree.py`, `features/encoder.py`, `features/transformers.py` | `extra={"event": ...}` on the stage log calls (wording unchanged); encoder part events; throttled transformer ticks on `xmr4el.progress`. |

Increment 2 library edits touch training code paths only through logging `extra=`;
no numeric change, no retrain.

## 8. Tests

- `fake_runs.make_run(root, name, *, kind, status, metrics, tree_id, tree_here,
  events, identity, corrupt=())` writes a run directory with the real file shapes.
- Pure modules: every state in §4, the §11 matrix (hash / derived / unknown /
  different), `normalize_spec` round trip, events with a torn last line.
- Screens: Textual `run_test(size=(80, 24))` + Pilot for §7 keys, focus restore,
  column dropping at 60 columns, `NO_COLOR`.
- `evaluate_tree`: population hash equal for the same file and tree, different when
  the vocabulary drops a different row; existing metric values unchanged; `strings_sha256` follows the
  effective dictionary; zero-sample metrics (`tests/test_eval.py`).
- Comparison evidence (§3a): `tests/tui/test_results.py` (protocol, strings, conflicts, malformed evidence,
  provenance-only reload, vanished baseline / inspection, 80x24 Generalization rows); search mounting and
  same-minute labels in `tests/tui/test_explorer.py`; input re-check in `tests/test_run_experiment.py`.
- Runner (increment 2): extends `tests/test_run_experiment.py` (synthetic TF-IDF spec,
  subprocess, `cwd=tmp`).

## 9. Increment 1 build order

One commit each on `tui`:

1. `evaluate_tree` population hash + preprocessing, test.
2. Packaging: `pyproject.toml`, `uv.lock`, empty `xmr4el_tui` with `__main__`,
   Dockerfile, CI. Check: wheel install outside the checkout imports it.
3. `fmt.py`, `runs.py`, `progress.py` (old runs: empty) + tests (the two-run `compare.py` was later replaced by `services/comparison.py`).
4. `RunsScreen` + `RunScreen` (Overview from metrics, Log, Inputs) + screen tests.
5. `CompareScreen` + tests.

Check per commit: `.venv/bin/python -m pytest -q && uvx ruff check xmr4el xmr4el_tui scripts tests`.
Manual check after 4: `.venv/bin/python -m xmr4el_tui` on the real `outputs/runs/`.

### Increment 2 build order

One commit each on `tui`:

1. Runner: `-check`, `-run_dir`, `-model_config_sha256`, `-tree_id`, exit 3; model config read once. Done.
2. `procinfo.py`; identity in `status.json`, `stopped` on SIGTERM/Ctrl+C, `heartbeat`. Done (`error` = `SIGTERM` / `SIGINT`).
3. `events.jsonl` handler; `extra={"event": ...}` on the stage log calls, encoder parts, transformer ticks. Done (`xmr4el.progress(**event)` for part/tick/runner events; the runner emits load, save, eval).
4. `process.py` (`check`, `spawn`, `derived_state`, `stop`); Runs/Run show RUNNING/LOST from it. Done (`Run.live` = `process.Live`, set by the app's scan for recorded-running runs; `rejection` reads the exit-3 line).
5. Setup + Review screens, spawn (§5.5-5.7). Done (`launch.py`; Runs `n` new, Runs/Run `e` evaluate tree; `d` duplicate left to increment 3; a superseded `-check` runs to its end, its result dropped by revision).
6. Stop modal (§5.9).

## 10. Decisions

- User 2026-10-23: Textual is the `tui` extra, pinned to the version `uv` resolves.
- User 2026-10-23: CI installs the extra (`uv sync --locked --extra tui`), so screen
  tests run on every push. Change `ci.yml` in step 2 with the extra itself; earlier
  it would fail on an unknown extra.

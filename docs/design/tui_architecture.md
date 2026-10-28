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
  compare.py       comparability verdict and deltas for two selected evaluations (no screen yet)
  fmt.py           durations, timestamps, metrics, counts, stage markers, state colours
  theme.py         the only colour values: semantic tokens (DARK), beam_color, $xmr-* CSS variables
  process.py       -check, spawn, derived state, stop; Launches (owner of started children)
  explorer.py      Dashboard explorer model: trees by dataset -> evaluations, runs without a tree; keys, labels
  messages.py      RunsChanged (the app rescanned)
  app.py           App: serialized, coalesced scan worker every 5 s; Snapshot; Launches reaping; global keys, theme
  services/
    experiment_repository.py  Snapshot (runs, trees, Activity per recorded-running run) with the read cache
                   and incremental events tails; Details of the selected item
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
    results_screen.py  ResultsScreen (§5.12): baseline + challengers, facts worker, inspection, group keys
    results_details.py pure Rich texts: baseline line, summary, analysis cells, inspection
  widgets/
    explorer.py    Explorer tree (resizable) + node labels; ResultsExplorer (row states, beam chip, ⓘ click)
    results_panels.py  BaselinePanel, ComparisonSummary, DetailedAnalysis (DataTable + notes), InspectionView
    description.py Description (read-only OptionList rows)
    row_rendering.py  FullRow cursor and Selectable (mouse text selection) mixins
    top_bar.py     TopBar (shared header)
    log_view.py    LogView: virtual-rendered log lines, level filter, search highlight
    log_panel.py   LogPanel: one LogTail polled in a worker while active, header, search box (LogScreen and the
                   Run monitor's Live logs)
    pipeline.py    Pipeline: stages / parts / layers as a Tree updated in place; follows the current stage
    monitor_panels.py  RunOverview (+ Cancel Run button), ResourcePanel
    resize.py      column_width clamp + DragRightBorder (Explorer, Run monitor left column)
  app.tcss
tests/tui/
  fake_runs.py     builder for run directories (shapes copied from real runs)
  test_runs.py  test_progress.py  test_logs.py  test_compare.py  test_procinfo.py  test_process.py
  test_dashboard.py  test_monitor.py  test_results.py  (Textual run_test + Pilot; importorskip("textual"))
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
```

`spec`, `provenance`, `git` and `model_config` are read lazily by `Run` helpers (`run.spec()`, ...; a
JSON object or None) for Compare; the Dashboard reads them once per selection through
`ExperimentRepository.details`.

Functions:

| Function | Purpose |
| --- | --- |
| `scan(root, cache) -> list[Run]` | Read every non-dot dir under `outputs/runs/`; reuse a cached `Run` while the (name, mtime, size) of its status/spec/tree/metrics files are unchanged (`signature`); a dir vanishing mid-read is left out. Newest first. |
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

**`compare.py`**

```python
@dataclass(frozen=True)
class Verdict:
    level: str                # "hash" | "derived" | "unknown" | "different"
    reasons: tuple[str, ...]  # header lines of §5.8
    differs: tuple[str, ...]  # tree, beam_size 10 -> 5, packages, ...
    subset_deltas: bool       # acc@1, MRR, R@k, R@cand
    all_rows_delta: bool      # acc@1_all
    timing_delta: bool

def verdict(a: Run, a_split, b: Run, b_split, runs) -> Verdict
def default_splits(a: Run, b: Run) -> tuple[str, str]   # first shared split in A's order
def rows(a_metrics, b_metrics, verdict) -> list[tuple[str, str, str, str]]  # metric, A, B, delta|""
```

`verdict` implements §11 in order: eval-file sha (`provenance.json`) and
preprocessing; population hash (`metrics.population_sha256`) or, for older runs, the
derivation through `training_run` (train sha, preprocessing, one clean commit);
device for timing.

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

## 5. UI modules

| Class | Shows | Reads |
| --- | --- | --- |
| `App` | footer, help panel, theme | `ExperimentRepository.snapshot` every 5 s in a worker -> `App.snapshot`, posts `RunsChanged` |
| `RunMonitorScreen` | one run by id: overview, pipeline, resources of the selected stage, live log | `App.snapshot` (state); `progress.EventTail` + resources.json (stat, reread on change) every 1 s in a worker while active, one read after it ends; provenance.json once; the log via `LogPanel` |
| `DashboardScreen` | explorer (Models > dataset > tree > evaluation; Active, no saved tree yet; Runs without a tree), description of the selected item, active runs | `App.snapshot`; `explorer.build`; `ExperimentRepository.details` of the selected item in a worker (xmodel.json, no pickle) |

| `ResultsScreen` | baseline + up to 3 challengers: summary, one metric group for all, inspection of one | `App.snapshot` (metrics); `comparison.read_facts` for every evaluation in a worker when the scan's runs change |

`DashboardScreen.open_results(key)` pushes Results with that evaluation as baseline. `open_run(run_id, split)`
pushes the Run monitor. Backend kept for Setup: `process.check/Launches/rejection`, `compare.verdict/rows`
(Results reuses compare's hash/preprocessing helpers),
`runs.normalize_spec/command_line/datasets/split_files/training_files/model_configs`.

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
  the vocabulary drops a different row; existing metric values unchanged.
- Runner (increment 2): extends `tests/test_run_experiment.py` (synthetic TF-IDF spec,
  subprocess, `cwd=tmp`).

## 9. Increment 1 build order

One commit each on `tui`:

1. `evaluate_tree` population hash + preprocessing, test.
2. Packaging: `pyproject.toml`, `uv.lock`, empty `xmr4el_tui` with `__main__`,
   Dockerfile, CI. Check: wheel install outside the checkout imports it.
3. `fmt.py`, `runs.py`, `progress.py` (old runs: empty), `compare.py` + tests.
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

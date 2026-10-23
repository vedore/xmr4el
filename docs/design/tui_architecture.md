# XMR4EL TUI code architecture

Status: proposed (2026-10-23). Implements [tui.md](tui.md); section numbers `§n` refer
to it. Nothing here is implemented yet.

## 1. Rules

- `xmr4el_tui` never imports `xmr4el` (torch at import, §10). It reads run files and
  spawns `scripts/run_experiment.py`.
- Only `app.py` / `screens.py` / `widgets.py` import Textual. Everything else is
  stdlib-only, pure where possible, and tested without a terminal.
- Run directories are the only durable state. The app holds a scan result, the
  current draft, and Popen handles of children it started (to reap them); nothing else.
- Every file read tolerates missing or corrupt input and reports it as data
  (`Run.problems`), never as an exception into the UI.
- Imports at the top of each module.

## 2. Layout

```text
xmr4el_tui/
  __init__.py      empty (no Textual import: the runner imports procinfo without the extra)
  __main__.py      argparse (--root), App(root).run()
  procinfo.py      process identity, stdlib only; shared by the runner and the TUI
  runs.py          scan outputs/runs/ -> Run records; spec normalization; tree links
  progress.py      events.jsonl -> stage list
  compare.py       comparability verdict and deltas for two selected evaluations
  fmt.py           durations, timestamps, metrics, counts, stage markers
  process.py       spawn, -check, derived state, stop            (increment 2)
  app.py           App: scan timer, global keys, command palette, theme
  screens.py       RunsScreen, RunScreen, CompareScreen; Setup/Review/StopModal (inc. 2)
  widgets.py       RunTable, Details, StageList, MetricsBlock, Banner, LogView
  app.tcss
tests/tui/
  fake_runs.py     builder for run directories (shapes copied from real runs)
  test_runs.py  test_progress.py  test_compare.py  test_fmt.py  test_procinfo.py
  test_screens.py  (Textual run_test + Pilot; importorskip("textual"))
```

Split `screens.py` when it passes ~600 lines, not before.

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

`spec`, `provenance`, `git` and `model_config` are read lazily by `Run` helpers
(`run.spec()`, `run.provenance()`, ...) for Inputs and Compare, not at scan time.

Functions:

| Function | Purpose |
| --- | --- |
| `scan(root, cache) -> list[Run]` | Read every non-dot dir under `outputs/runs/`; reuse a cached `Run` when the dir's and `status.json`'s mtimes are unchanged. Newest first. |
| `same_tree(runs, run) -> list[Run]` | Other runs with the same `tree_id` (details `also:`). |
| `training_run(runs, tree_id) -> Run \| None` | The `kind == "train"` run that saved this tree (§11 derivation, eval-only defaults). |
| `previous_completed(runs, run) -> Run \| None` | Newest COMPLETED run with the same name, older than `run` (§5.2 timings). |
| `normalize_spec(recorded) -> (spec, xmodel_path)` | §10 round trip: drop `xmodel_path` from the copy, return it separately. |
| `command_line(spec, xmodel_path) -> str` | Equivalent CLI for Inputs (`-spec -` with the normalized JSON). |

Scan cost: about 5 small JSON reads per changed run; unchanged runs cost two
`stat` calls. Rescan every 5 s and on `r`, in a Textual thread worker.

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
| `App` | header right side (host, running count), footer, command palette, theme | `scan` every 5 s -> posts `RunsChanged` |
| `RunsScreen` | §5.1 table + details | `App.runs`; `same_tree`, `previous_completed` |
| `RunScreen` | §5.2-5.4 tabs Overview / Log / Inputs | the `Run`, `progress`, `LogView` tail |
| `CompareScreen` | §5.8 | `compare.verdict`, `compare.rows` |
| `RunTable` | DataTable, column priority by width (§5.1) | |
| `StageList`, `MetricsBlock`, `Banner` | §5.2-5.4 blocks | |
| `LogView` | bounded tail of `run.log` (last 256 KB, older on request), follow, search, `w` filter | polls size every 1 s |

Screens receive `RunsChanged` and re-render from the new `Run`, keeping selection,
scroll and marks by run `id` (§7).

## 6. Process side (`process.py`, increment 2)

```python
def check(root, spec, xmodel_path, revision) -> CheckResult   # spawns -check, parses one JSON object
def spawn(root, spec, run_dir, xmodel_path, config_sha, tree_id) -> Popen
def derived_state(run, here: dict) -> str     # RUNNING | LOST | UNVERIFIED | recorded
def stop(run, force=False) -> str             # re-verifies identity first (§9); returns outcome
```

- `check` runs in a thread worker; the app ignores results whose `revision` is not
  the draft's current revision (§5.5).
- `spawn`: argument list, `cwd=root`, `start_new_session=True`, stdin = spec JSON,
  stdout/stderr = `outputs/runs/.console/<run-id>.log`. The Popen goes into
  `App.children` only so `poll()` reaps it; the run's state always comes from files.
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

## 10. Decisions

- User 2026-10-23: Textual is the `tui` extra, pinned to the version `uv` resolves.
- User 2026-10-23: CI installs the extra (`uv sync --locked --extra tui`), so screen
  tests run on every push. Change `ci.yml` in step 2 with the extra itself; earlier
  it would fail on an unknown extra.

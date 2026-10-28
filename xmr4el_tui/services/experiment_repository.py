"""Every file the TUI reads (tui.md §10): scans of outputs/runs/ and outputs/saved_trees/ into one Snapshot, and the
Details of the selected explorer item. Stdlib only, no UI.

The repository owns the read cache, so one scan at a time may call `snapshot` (the app serializes its scans).
`details` touches no cache: it may run in another thread beside a scan.
"""
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path

from xmr4el_tui import procinfo, process, runs
from xmr4el_tui.progress import EventTail, current_stage, last_stage_event


@dataclass(frozen=True)
class Activity:
    """What a recorded-running run is doing, from its events."""
    stage: str | None = None    # `hierarchy  layer 1/2 · …` of its current stage
    device: str | None = None
    last: tuple | None = None   # (stage, time) of its last event


@dataclass(frozen=True)
class Snapshot:
    runs: tuple = ()          # newest first, recorded-running runs carry their live state (§9)
    found: bool = False       # outputs/runs exists
    trees: tuple = ()
    active: dict = field(default_factory=dict)  # recorded-running run id -> Activity
    scanned: bool = False     # False: the first scan has not landed yet
    problem: str | None = None  # why this scan failed (the other fields are then the last successful scan's)
    at: datetime | None = None  # when this scan started (before anything was read)


@dataclass(frozen=True)
class Details:
    """Files one Description shows beyond the Snapshot, each read once. Empty = not loaded (yet)."""
    spec: dict = field(default_factory=dict)        # spec.json of the item's run (a tree: its training run)
    git: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)
    view: dict = field(default_factory=dict)        # xmodel.json of a tree
    config_copy: bool = False                       # the training run's model_config.json exists
    saved: datetime | None = None                   # when the training run wrote tree.json
    heartbeat: float | None = None                  # seconds since the run's heartbeat
    problems: tuple = ()                            # "provenance.json: invalid JSON", ...


class ExperimentRepository:
    def __init__(self, root):
        self.root = Path(root)
        self.cache = {}  # runs.scan cache
        # recorded-running run id -> (EventTail, state word and Activity derived from it: LOST fails the current
        # stage). ponytail: every parsed event of the run kept; a streaming reducer if files get huge
        self.tails = {}
        self.last = Snapshot()  # last successful scan

    def snapshot(self, fresh=False):
        """Read everything the Dashboard lists. Only reads files; `fresh` rereads every one (no cache)."""
        at = datetime.now()
        if fresh:
            self.cache.clear()
            self.tails.clear()
        try:
            found = runs.scan(self.root, self.cache)
        except FileNotFoundError:
            self.last = Snapshot(trees=tuple(runs.trees(self.root, [])), scanned=True, at=at)
            return self.last
        except OSError as err:  # outputs/runs cannot be listed: keep the last good scan, say why
            return replace(self.last, scanned=True, at=at, problem=f"outputs/runs/ cannot be read ({type(err).__name__}); "
                                                            "showing the last successful scan")
        here = procinfo.runtime()
        found = [self.live(r, here) if r.recorded == "running" else r for r in found]
        active = {r.id: self.activity(r) for r in found if r.recorded == "running"}
        self.tails = {k: v for k, v in self.tails.items() if k in active}
        self.last = Snapshot(tuple(found), True, tuple(runs.trees(self.root, found)), active, scanned=True, at=at)
        return self.last

    def live(self, run, here):
        """`run` (recorded running) with its live state. A process found gone may have written its final status
        after status.json was read: then the run is read again and shows that outcome, not LOST."""
        live = process.derived_state(run, here)
        if live.state == "LOST":
            fresh = runs.read_run(run.path, self.root)
            if fresh.recorded in process.FINAL:  # its files changed: the cache rereads it next scan
                return fresh
        return replace(run, live=live)

    def activity(self, run):
        """Activity of a recorded-running run. Reads only the events appended since the last scan; a replaced or
        truncated events.jsonl is read again from the start; an unchanged one is not touched."""
        word = runs.state_word(run)
        tail, old_word, activity = self.tails.get(run.id) or (EventTail(run.path / "events.jsonl"), None, None)
        if not tail.read() and old_word == word:
            return activity
        activity = Activity(current_stage(run, tail.events), runs.text(runs.read_dict(run.path / "provenance.json"),
                                                                       "torch_device"), last_stage_event(tail.events))
        self.tails[run.id] = (tail, word, activity)
        return activity

    def details(self, node):
        """Details of explorer node `node` (explorer.Node): a tree, an evaluation or a run; others have none."""
        problems = []

        def read(path):
            obj, problem = runs.read_json(path)
            if problem or (obj is not None and not isinstance(obj, dict)):
                problems.append(problem or f"{path.name}: not a JSON object")
            return obj if isinstance(obj, dict) else {}

        kind = node.kind if node is not None else None
        if kind not in ("tree", "eval", "run"):
            return Details()
        run = node.ref.trained_by if kind == "tree" else node.ref[0] if kind == "eval" else node.ref
        view = read(self.root / node.ref.path / "xmodel.json") if kind == "tree" else {}
        if run is None:
            return Details(view=view, problems=tuple(problems))
        try:
            saved = datetime.fromtimestamp((run.path / "tree.json").stat().st_mtime) if kind == "tree" else None
        except OSError:
            saved = None
        return Details(spec=read(run.path / "spec.json"), git=read(run.path / "git.json"),
                       provenance=read(run.path / "provenance.json"), view=view,
                       config_copy=(run.path / "model_config.json").is_file(), saved=saved,
                       heartbeat=runs.heartbeat_age(run), problems=tuple(problems))

"""Every file the TUI reads (tui.md §10): scans of outputs/runs/ and outputs/saved_trees/ into one Snapshot, and the
Details of the selected explorer item. Stdlib only, no UI.

The repository owns the read cache, so one scan at a time may call `snapshot` (the app serializes its scans).
`details` touches no cache: it may run in another thread beside a scan.
"""
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path

from xmr4el_tui import procinfo, process, runs
from xmr4el_tui.progress import current_stage, last_stage_event, read_events


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
        self.last = Snapshot()  # last successful scan

    def snapshot(self, fresh=False):
        """Read everything the Dashboard lists. Only reads files; `fresh` rereads every one (no cache)."""
        if fresh:
            self.cache.clear()
        try:
            found = runs.scan(self.root, self.cache)
        except FileNotFoundError:
            self.last = Snapshot(trees=tuple(runs.trees(self.root, [])), scanned=True)
            return self.last
        except OSError as err:  # outputs/runs cannot be listed: keep the last good scan, say why
            return replace(self.last, scanned=True, problem=f"outputs/runs/ cannot be read ({type(err).__name__}); "
                                                            "showing the last successful scan")
        here = procinfo.runtime()
        found = [replace(r, live=process.derived_state(r, here)) if r.recorded == "running" else r for r in found]
        active = {r.id: self.activity(r) for r in found if r.recorded == "running"}
        self.last = Snapshot(tuple(found), True, tuple(runs.trees(self.root, found)), active, scanned=True)
        return self.last

    def activity(self, run):
        events, _ = read_events(run.path / "events.jsonl")
        return Activity(current_stage(run, events), runs.text(runs.read_dict(run.path / "provenance.json"),
                                                              "torch_device"), last_stage_event(events))

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

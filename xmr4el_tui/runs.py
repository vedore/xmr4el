"""Read outputs/runs/<run-id>/ into Run records (tui.md §10 "Run browser data"). Stdlib only, no UI.

Every read tolerates missing or corrupt files: problems are recorded in `Run.problems`, never raised.
"""
import json
import re
import shlex
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

RUNS_DIR = Path("outputs", "runs")
SEARCH_KEYS = ("beam_size", "topk", "knn_beta", "leaf_candidates")
RUN_ID = re.compile(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_(.+)")
RECORDED_STATES = {"running", "completed", "failed", "stopped"}
# status.json keys that are not process identity (§9)
STATUS_KEYS = {"status", "started", "finished", "error"}


@dataclass(frozen=True)
class Eval:
    split: str
    test_path: str | None
    train_path: str | None
    overrides: dict           # search keys given in the spec
    metrics: dict | None      # metrics_<split>.json, None = not produced yet


@dataclass(frozen=True)
class Run:
    id: str                   # directory name, e.g. 2026-10-20_13-36-20_ncbi_disease
    path: Path
    name: str                 # spec name
    kind: str                 # "train" | "eval" | "unknown" (no readable spec.json or tree.json)
    recorded: str             # status.json status, or "unknown"
    started: datetime | None  # status.json started, else parsed from id
    finished: datetime | None
    error: str | None
    tree_id: str | None       # tree.json
    tree_path: str | None
    tree_here: bool           # <tree_path>/xmodel.pkl exists
    evals: tuple[Eval, ...]   # spec order
    identity: dict | None     # pid/host/... from status.json (§9), None for old runs
    problems: tuple[str, ...] # "metrics_dev.json: invalid JSON", ...
    live: object = None       # process.Live of a recorded-running run (set by the app's scan), else None

    def spec(self):
        return read_json(self.path / "spec.json")[0]

    def provenance(self):
        return read_json(self.path / "provenance.json")[0]

    def git(self):
        return read_json(self.path / "git.json")[0]

    def model_config(self):
        return read_json(self.path / "model_config.json")[0]

    def eval(self, split):
        return next((e for e in self.evals if e.split == split), None)


def read_json(path):
    """(object, problem): (None, None) when missing, (None, "<file>: <reason>") when unreadable."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f), None
    except FileNotFoundError:
        return None, None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as err:
        reason = "invalid JSON" if isinstance(err, json.JSONDecodeError) else type(err).__name__
        return None, f"{Path(path).name}: {reason}"


def _time(value):
    try:
        return datetime.fromisoformat(value) if isinstance(value, str) else None
    except ValueError:
        return None


def read_run(path, root):
    """One run directory -> Run."""
    problems = []

    def load(name, kind=dict):
        obj, problem = read_json(path / name)
        if problem:
            problems.append(problem)
        elif obj is not None and not isinstance(obj, kind):
            problems.append(f"{name}: not a JSON {kind.__name__}")
            obj = None
        return obj

    status, spec, tree = load("status.json"), load("spec.json"), load("tree.json")
    if status is None and not any(p.startswith("status.json") for p in problems):
        problems.append("status.json: missing")
    if spec is None and not any(p.startswith("spec.json") for p in problems):
        problems.append("spec.json: missing")
    status, tree = status or {}, tree or {}

    m = RUN_ID.fullmatch(path.name)
    recorded = status.get("status") if status.get("status") in RECORDED_STATES else "unknown"
    if spec is not None:
        kind = "train" if spec.get("xmodel_path") is None else "eval"
    elif "trained" in tree:
        kind = "train" if tree["trained"] else "eval"
    else:
        kind = "unknown"

    spec_evals = (spec or {}).get("evals")
    if isinstance(spec_evals, list) and all(isinstance(e, dict) and isinstance(e.get("split"), str) for e in spec_evals):
        splits = [(e["split"], e) for e in spec_evals]
    else:  # no usable spec: whatever metrics exist, in name order
        splits = [(p.name[len("metrics_"):-len(".json")], {}) for p in sorted(path.glob("metrics_*.json"))]
    evals = []
    for split, e in splits:
        metrics = load(f"metrics_{split}.json")
        evals.append(Eval(split=split, test_path=e.get("test_path") or (metrics or {}).get("test_path"),
                          train_path=e.get("train_path") or (metrics or {}).get("train_path"),
                          overrides={k: e[k] for k in SEARCH_KEYS if k in e}, metrics=metrics))

    tree_path = tree.get("xmodel_path") if isinstance(tree.get("xmodel_path"), str) else None
    identity = {k: v for k, v in status.items() if k not in STATUS_KEYS} or None
    return Run(id=path.name, path=path, name=(spec or {}).get("name") or (m.group(2) if m else path.name),
               kind=kind, recorded=recorded,
               started=_time(status.get("started")) or (datetime.strptime(m.group(1), "%Y-%m-%d_%H-%M-%S") if m else None),
               finished=_time(status.get("finished")), error=status.get("error"),
               tree_id=tree.get("tree_id"), tree_path=tree_path, tree_here=tree_on_disk(root, tree_path),
               evals=tuple(evals), identity=identity, problems=tuple(problems))


def tree_on_disk(root, tree_path):
    return bool(tree_path) and (Path(root) / tree_path / "xmodel.pkl").is_file()


def scan(root, cache=None):
    """Every non-dot directory under <root>/outputs/runs, newest first. `cache` (dict, updated in place) reuses a
    Run whose directory and status.json mtimes are unchanged. Raises FileNotFoundError without outputs/runs."""
    cache = {} if cache is None else cache
    runs = []
    for path in sorted(Path(root, RUNS_DIR).iterdir()):
        if path.name.startswith(".") or not path.is_dir():
            continue
        try:
            status_mtime = (path / "status.json").stat().st_mtime_ns
        except OSError:
            status_mtime = None
        key = (path.stat().st_mtime_ns, status_mtime)
        hit = cache.get(path.name)
        if hit and hit[0] == key:  # trees are synced separately from runs: recheck that one file
            run = replace(hit[1], tree_here=tree_on_disk(root, hit[1].tree_path))
        else:
            run = read_run(path, root)
        cache[path.name] = (key, run)
        runs.append(run)
    for gone in set(cache) - {r.id for r in runs}:
        del cache[gone]
    return sorted(runs, key=lambda r: (r.started or datetime.min, r.id), reverse=True)


def same_tree(runs, run):
    """Other runs on the same tree (details `also:`)."""
    return [r for r in runs if run.tree_id and r.tree_id == run.tree_id and r.id != run.id]


def training_run(runs, tree_id):
    """The train run that saved this tree, or None (not synced, or trained outside run_experiment)."""
    return next((r for r in runs if tree_id and r.kind == "train" and r.tree_id == tree_id), None)


def previous_completed(runs, run):
    """Newest COMPLETED run with the same name, older than `run`."""
    older = [r for r in runs if r.name == run.name and r.recorded == "completed" and r.id != run.id
             and r.started and run.started and r.started < run.started]
    return max(older, key=lambda r: r.started, default=None)


def normalize_spec(recorded):
    """Recorded spec.json -> (input spec, xmodel_path). xmodel_path is invocation metadata: dropped from the spec,
    returned separately (None = training). Other keys are kept for the runner to accept or reject."""
    spec = dict(recorded)
    return spec, spec.pop("xmodel_path", None)


def command_line(spec, xmodel_path):
    """Shell command equivalent to the run: the normalized spec on stdin."""
    args = ["python", "scripts/run_experiment.py", "-spec", "-"]
    if xmodel_path is not None:
        args += ["-xmodel_path", xmodel_path]
    return f"{shlex.join(args)} <<'EOF'\n{json.dumps(spec, indent=2)}\nEOF"

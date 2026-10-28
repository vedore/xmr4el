"""Read outputs/runs/<run-id>/ into Run records (tui.md §10 "Run browser data"). Stdlib only, no UI.

Every read tolerates missing or corrupt files and fields of the wrong JSON type: problems are recorded in
`Run.problems`, never raised; a field of the wrong type reads as absent.
"""
import json
import os
import re
import shlex
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

from xmr4el_tui.fmt import clean

RUNS_DIR = Path("outputs", "runs")
SEARCH_KEYS = ("beam_size", "topk", "knn_beta", "leaf_candidates")
RUN_ID = re.compile(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_(.+)")
RECORDED_STATES = {"running", "completed", "failed", "stopped"}
TREES_DIR = Path("outputs", "saved_trees")
# <dataset>/{train,dev,test}.pubtator, MedMentions <dataset>/corpus_pubtator_{train,dev,test}.txt
SPLIT_FILE = re.compile(r"(?:.*[_-])?(train|dev|test)\.pubtator|.*pubtator[_-](train|dev|test)\.txt")
SPLITS = ("train", "dev", "test")
# status.json keys that are not process identity (§9)
STATUS_KEYS = {"status", "started", "finished", "error", "forced"}  # forced: process.record_forced
# with metrics_*.json: every file read_run reads, and the provenance / git evidence Results compares (services.comparison)
RUN_FILES = ("status.json", "spec.json", "tree.json", "model_config.json", "provenance.json", "git.json")


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
    train_path: str | None = None  # spec train_path of a train run
    features: str | None = None    # model_config.json of a train run (explorer descriptor)
    depth: int | None = None
    files: tuple = ()              # scan signature: a rewritten provenance.json / git.json makes the Run differ

    def spec(self):
        return read_dict(self.path / "spec.json")

    def provenance(self):
        return read_dict(self.path / "provenance.json")

    def git(self):
        return read_dict(self.path / "git.json")

    def model_config(self):
        return read_dict(self.path / "model_config.json")

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


def read_dict(path):
    """The JSON object in `path`, None when missing, unreadable or not an object."""
    obj = read_json(path)[0]
    return obj if isinstance(obj, dict) else None


def as_dict(obj):
    return obj if isinstance(obj, dict) else {}


def dig(obj, *keys):
    """obj[k1][k2]..., None when a level is missing or not a JSON object."""
    for k in keys:
        if not isinstance(obj, dict):
            return None
        obj = obj.get(k)
    return obj


def text(obj, key):
    """obj[key] when it is a string (control sequences removed, fmt.clean), else None."""
    value = obj.get(key) if isinstance(obj, dict) else None
    return clean(value) if isinstance(value, str) else None


def parse_time(value):
    """ISO time -> naive local datetime (one with an offset is converted, so all times compare), None when absent or
    invalid."""
    try:
        t = datetime.fromisoformat(value) if isinstance(value, str) else None
    except ValueError:
        return None
    return t.astimezone().replace(tzinfo=None) if t and t.tzinfo else t


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

    def string(obj, name, key):
        """obj[key] when a string (fmt.clean); present with another type: a problem, read as absent."""
        value = (obj or {}).get(key)
        if value is not None and not isinstance(value, str):
            problems.append(f"{name}: {key} is not a string")
            return None
        return value and clean(value)

    def time(name, key):
        value = string(status, name, key)
        if value is not None and parse_time(value) is None:
            problems.append(f"{name}: {key} is not an ISO time")
        return parse_time(value)

    status, spec, tree = load("status.json"), load("spec.json"), load("tree.json")
    model = load("model_config.json") or {}
    if status is None and not any(p.startswith("status.json") for p in problems):
        problems.append("status.json: missing")
    if spec is None and not any(p.startswith("spec.json") for p in problems):
        problems.append("spec.json: missing")
    status, tree = status or {}, tree or {}

    m = RUN_ID.fullmatch(path.name)
    recorded = string(status, "status.json", "status")
    if recorded is not None and recorded not in RECORDED_STATES:
        problems.append(f"status.json: unknown status {recorded!r}")
    recorded = recorded if recorded in RECORDED_STATES else "unknown"
    if spec is not None:
        kind = "train" if spec.get("xmodel_path") is None else "eval"
    elif "trained" in tree:
        kind = "train" if tree["trained"] else "eval"
    else:
        kind = "unknown"

    spec_evals = (spec or {}).get("evals")
    if isinstance(spec_evals, list) and all(isinstance(e, dict) and isinstance(e.get("split"), str) for e in spec_evals):
        splits = [(clean(e["split"]), e) for e in spec_evals]
    else:  # no usable spec: whatever metrics exist, in name order
        splits = [(p.name[len("metrics_"):-len(".json")], {}) for p in sorted(path.glob("metrics_*.json"))]
    evals = []
    for split, e in splits:
        metrics = load(f"metrics_{split}.json")
        evals.append(Eval(split=split, test_path=text(e, "test_path") or text(metrics, "test_path"),
                          train_path=text(e, "train_path") or text(metrics, "train_path"),
                          overrides={k: e[k] for k in SEARCH_KEYS if k in e}, metrics=metrics))

    tree_path = string(tree, "tree.json", "xmodel_path")
    identity = {k: v for k, v in status.items() if k not in STATUS_KEYS} or None
    return Run(id=path.name, path=path, name=string(spec, "spec.json", "name") or (m.group(2) if m else path.name),
               kind=kind, recorded=recorded,
               started=time("status.json", "started") or id_time(m),
               finished=time("status.json", "finished"), error=string(status, "status.json", "error"),
               tree_id=string(tree, "tree.json", "tree_id"), tree_path=tree_path,
               tree_here=tree_on_disk(root, tree_path),
               evals=tuple(evals), identity=identity, problems=tuple(problems),
               train_path=string(spec, "spec.json", "train_path"), features=text(model, "features"),
               depth=model.get("depth") if type(model.get("depth")) is int else None)


def id_time(m):
    """The start time in a RUN_ID match, None without one or when it is no date (2026-99-99_…)."""
    try:
        return datetime.strptime(m.group(1), "%Y-%m-%d_%H-%M-%S") if m else None
    except ValueError:
        return None


def state_word(run):
    """Shown state: the live check for recorded-running runs (UNVERIFIED without one), else the recorded state."""
    if run.recorded == "running":
        return run.live.state if run.live else "UNVERIFIED"
    return run.recorded.upper()


def elapsed(run, now=None):
    """Finish - start; a recorded-running run counts to `now` unless it is LOST (when it died is unknown: None)."""
    end = run.finished or (now or datetime.now() if run.recorded == "running" and state_word(run) != "LOST" else None)
    return (end - run.started).total_seconds() if end and run.started else None


def heartbeat_age(run, now=None):
    """Seconds since the runner last touched `heartbeat`, None without one."""
    try:
        return (now or datetime.now()).timestamp() - (run.path / "heartbeat").stat().st_mtime
    except OSError:
        return None


def tree_on_disk(root, tree_path):
    return bool(tree_path) and (Path(root) / tree_path / "xmodel.pkl").is_file()


def signature(path):
    """(name, mtime, size) of every file read_run reads: adding, removing or rewriting one in place changes it."""
    with os.scandir(path) as entries:
        return tuple(sorted((e.name, st.st_mtime_ns, st.st_size) for e in entries
                            if e.name in RUN_FILES or (e.name.startswith("metrics_") and e.name.endswith(".json"))
                            for st in (e.stat(),)))


def scan(root, cache=None):
    """Every non-dot directory under <root>/outputs/runs, newest first. `cache` (dict, updated in place) reuses a
    Run whose files are unchanged (`signature`). Raises FileNotFoundError without outputs/runs (OSError when it
    cannot be listed); a run directory that vanishes while it is read is left out."""
    cache = {} if cache is None else cache
    runs = []
    for path in sorted(Path(root, RUNS_DIR).iterdir()):
        if path.name.startswith(".") or not path.is_dir():
            continue
        try:
            key = signature(path)
        except FileNotFoundError:  # gone since iterdir (or a file of it: read it again next scan)
            if not path.is_dir():
                continue
            key = None
        except OSError:  # unreadable: read_run records why, never cached
            key = None
        hit = cache.get(path.name)
        if key is not None and hit and hit[0] == key:  # trees are synced separately from runs: recheck that one file
            run = replace(hit[1], tree_here=tree_on_disk(root, hit[1].tree_path))
        else:
            run = replace(read_run(path, root), files=key or ())
            if not path.is_dir():  # gone while read: its files read as missing, not as a broken run
                continue
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


@dataclass(frozen=True)
class Tree:
    id: str | None
    path: str                 # outputs/saved_trees/xmodel_<stamp>
    here: bool                # xmodel.pkl on this machine
    trained_by: Run | None    # None: training run not synced, or trained outside run_experiment
    used_by: tuple[Run, ...]  # every run on this tree, newest first (trained_by included)
    labels: int | None
    recorded_path: str | None = None  # a missing tree.json path of runs whose tree was found moved to `path`

    def train_path(self):
        """Training file of the tree (its training run's spec), or None."""
        return self.trained_by.train_path if self.trained_by else None


def canonical(root, path):
    """The one spelling of a tree location (relative to `root` or absolute, as recorded): relative to the real root
    when inside it, `..` and symlinks resolved. Two runs on the same directory always agree on it."""
    full, base = Path(os.path.realpath(Path(root, path))), Path(os.path.realpath(root))
    return str(full.relative_to(base)) if full.is_relative_to(base) else str(full)


def trees(root, all_runs):
    """Saved trees: those runs point at (tree.json), then tree directories no run mentions (xmodel.json), newest
    use first. Locations are compared canonically. A recorded tree missing at its path joins the one other location
    of its tree_id that exists (moved by hand; several = copies, ambiguous: it stays missing). Paths in the run
    files are never rewritten. No pickle is read."""
    found = {}
    for r in all_runs:  # newest first
        if r.tree_path:
            found.setdefault(canonical(root, r.tree_path), []).append(r)
    here = {p: tree_on_disk(root, p) for p in found}
    unmentioned = {}  # path -> (tree_id, labels) of directories no run mentions
    on_disk = sorted(Path(root, TREES_DIR).glob("*/xmodel.json")) if Path(root, TREES_DIR).is_dir() else []
    for view_path in reversed(on_disk):
        path = canonical(root, view_path.parent)
        if path not in found:
            view = read_dict(view_path)
            n = dig(view, "initial_labels", "len")
            unmentioned[path] = (text(view, "tree_id"), n if isinstance(n, int) else None)
    merged = {}  # location -> (runs on it, a recorded location of them that is missing)
    for path, used in found.items():
        tree_id, at = used[0].tree_id, path
        if not here[path] and tree_id:
            places = [q for q in found if q != path and here[q] and found[q][0].tree_id == tree_id]
            places += [q for q, (i, _) in unmentioned.items() if i == tree_id]
            at = places[0] if len(places) == 1 else path
        runs_at, recorded = merged.get(at, ([], None))
        merged[at] = (runs_at + used, recorded or (path if at != path else None))
    out = []
    for path, (used, recorded) in merged.items():
        unmentioned.pop(path, None)
        used = sorted(used, key=lambda r: (r.started or datetime.min, r.id), reverse=True)
        labels = next((e.metrics.get("labels") for r in used for e in r.evals if e.metrics), None)
        out.append(Tree(id=used[0].tree_id, path=path, here=tree_on_disk(root, path),
                        trained_by=next((r for r in used if r.kind == "train"), None), used_by=tuple(used),
                        labels=labels if isinstance(labels, int) else None, recorded_path=recorded))
    out += [Tree(id=i, path=p, here=tree_on_disk(root, p), trained_by=None, used_by=(), labels=n)
            for p, (i, n) in unmentioned.items()]
    return out


def model_configs(root):
    """configs/*.json (experiment specs live in configs/experiments/)."""
    return [str(p.relative_to(root)) for p in sorted(Path(root, "configs").glob("*.json"))]


def experiment_specs(root):
    return [str(p.relative_to(root)) for p in sorted(Path(root, "configs", "experiments").glob("*.json"))]


def split_files(root, dataset):
    """{split: path} of a dataset directory, in SPLITS order."""
    try:
        names = sorted(p.name for p in Path(root, dataset).iterdir() if p.is_file())
    except OSError:
        return {}
    found = {m[m.lastindex]: str(Path(dataset, n)) for n in names if (m := SPLIT_FILE.fullmatch(n))}
    return {s: found[s] for s in SPLITS if s in found}


def training_files(root, dataset):
    """Files of a dataset usable for training: names with "train" (train, train_plus_ctd)."""
    try:
        return [str(Path(dataset, p.name)) for p in sorted(Path(root, dataset).iterdir())
                if p.is_file() and "train" in p.stem and (p.suffix == ".pubtator" or "pubtator" in p.stem)]
    except OSError:
        return []


def datasets(root):
    """Directories under datasets/ with a dev or test file."""
    found = {str(p.parent.relative_to(root)) for p in Path(root, "datasets").rglob("*")
             if (m := SPLIT_FILE.fullmatch(p.name)) and m[m.lastindex] != "train"} if Path(root, "datasets").is_dir() else ()
    return sorted(found, key=str.lower)


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

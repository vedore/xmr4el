"""One tracked run: train a tree from a spec (or take a saved one) and evaluate it on each listed split.

Spec (JSON): {"name", "model_config", "train_path", "evals": [{"split", "test_path", "train_path"?, "beam_size"?,
"topk"?, "knn_beta"?, "leaf_candidates"?}]}; an eval's train_path adds the seen/unseen mention-string breakdown, search settings left out
come from the tree's predict_config. Unknown keys are errors. PubTator input only.

Writes outputs/runs/<timestamp>_<name>/ (an existing directory is refused): spec.json (as given + xmodel_path),
model_config.json, git.json (commit, dirty incl. untracked files, untracked paths) + git_diff.patch if tracked files
changed, provenance.json (sha256 of every input file, python/platform/package versions, torch device), run.log,
tree.json (saved tree path), metrics_<split>.json (scripts/evaluate.py's numbers), status.json (running ->
completed | failed with the error | stopped by SIGTERM/SIGINT; process identity from xmr4el_tui.procinfo kept
throughout), heartbeat (touched every HEARTBEAT seconds while running). An eval with no row in the tree's vocabulary fails the run. With -xmodel_path:
no training, evals only. `run(spec, xmodel_path)` is the same run without argument parsing.

The model config is read once: validation, model_config.json, its provenance sha256 and the model all use those bytes.
-run_dir PATH: run there instead (must not exist). -model_config_sha256 / -tree_id (TUI launches): the config bytes /
the tree's xmodel.json tree_id must match, else exit 3 before any output with one JSON line on stdout
{"error": "model_config_changed" | "tree_changed", "expected", "actual"}. -check: validate only (no run dir, no model
work, no pickle loaded); stdout = one JSON object {"ok", "error", "model_config_sha256" (training), "tree_id"
(eval-only), "search": {split: effective search}}, exit 0 ok / 1 invalid, logs on stderr.
"""
import hashlib
import contextlib
import json
import logging
import os
import platform
import re
import signal
import subprocess
import sys
import threading
import time
from argparse import ArgumentParser
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version

from xmr4el import set_verbosity, torch_device
from xmr4el.data.readers import Preprocessor
from xmr4el.eval import evaluate_tree, format_metrics
from xmr4el.xmodel import PREDICT_DEFAULTS, XModel, check_search
from xmr4el_tui.procinfo import own_identity

SPEC_KEYS = {"name", "model_config", "train_path", "evals"}
EVAL_KEYS = {"split", "test_path", "train_path", "beam_size", "topk", "knn_beta", "leaf_candidates"}
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGES = ("numpy", "scipy", "scikit-learn", "torch", "transformers")
HEARTBEAT = 30  # seconds


class Stopped(BaseException):
    """SIGTERM (TUI stop, kill, docker stop): the run is recorded as stopped."""


class BindingRejected(Exception):
    """The reviewed model config / tree changed since -check (exit 3)."""


def check_spec(spec, train, xmodel_path=None):
    """Raise ValueError on unknown/missing keys, a name that is not [A-Za-z0-9_.-]+, duplicate splits, bad search
    settings, a model config XModel refuses, or a missing input file, before any work. Returns the model config
    bytes (training) or None: callers use these, never the path again."""
    need = {"name", "evals"} | ({"model_config", "train_path"} if train else set())
    if set(spec) - SPEC_KEYS or need - set(spec):
        raise ValueError(f"spec keys: unknown {sorted(set(spec) - SPEC_KEYS)}, missing {sorted(need - set(spec))}")
    if not spec["evals"]:
        raise ValueError("spec has no evals")
    for e in spec["evals"]:
        if set(e) - EVAL_KEYS or {"split", "test_path"} - set(e):
            raise ValueError(f"eval {e}: allowed keys {sorted(EVAL_KEYS)}, split and test_path required")
    splits = [e["split"] for e in spec["evals"]]
    if len(set(splits)) != len(splits):
        raise ValueError(f"duplicate eval splits {splits}")
    for name in [spec["name"], *splits]:  # both become file names
        if not (isinstance(name, str) and re.fullmatch(r"[A-Za-z0-9_.-]+", name)):
            raise ValueError(f"name/split {name!r}: use [A-Za-z0-9_.-]+")
    for e in spec["evals"]:
        check_search(e)
    paths = [p for e in spec["evals"] for p in (e["test_path"], e.get("train_path")) if p]
    paths += [spec["model_config"], spec["train_path"]] if train else []
    missing = [p for p in paths if not os.path.isfile(p)]
    missing += [] if train or os.path.isfile(os.path.join(xmodel_path, "xmodel.pkl")) else [xmodel_path]
    if missing:
        raise ValueError(f"missing input files/trees: {missing}")
    same = [e["split"] for e in spec["evals"] if train and os.path.samefile(e["test_path"], spec["train_path"])]
    if same:
        raise ValueError(f"evals {same}: test_path is the training file")
    if not train:
        return None
    with open(spec["model_config"], "rb") as f:
        raw = f.read()
    XModel(**json.loads(raw))  # unknown keys, bad depth (= XModel.load_config on the captured bytes)
    return raw


def tree_view(xmodel_path):
    """tree_id and predict_config from the tree's readable xmodel.json (no pickle is loaded)."""
    path = os.path.join(xmodel_path, "xmodel.json")
    try:
        with open(path) as f:
            view = json.load(f)
    except (OSError, ValueError) as err:
        raise ValueError(f"{path}: {err}; effective search cannot be resolved") from err
    if not isinstance(view.get("predict_config"), dict) or not view.get("tree_id"):
        raise ValueError(f"{path}: no predict_config/tree_id; effective search cannot be resolved")
    return view["tree_id"], view["predict_config"]


def effective_search(spec, predict_config):
    """split -> the search each eval runs: the tree's predict_config over PREDICT_DEFAULTS, then the eval's keys."""
    return {e["split"]: {**PREDICT_DEFAULTS, **predict_config,
                         **{k: e[k] for k in PREDICT_DEFAULTS if e.get(k) is not None}} for e in spec["evals"]}


def check(spec, xmodel_path=None):
    """-check response (dict) for `spec`; raises on an invalid spec."""
    raw = check_spec(spec, train=xmodel_path is None, xmodel_path=xmodel_path)
    if raw is None:
        tree_id, predict_config = tree_view(xmodel_path)
        return {"ok": True, "error": None, "model_config_sha256": None, "tree_id": tree_id,
                "search": effective_search(spec, predict_config)}
    return {"ok": True, "error": None, "model_config_sha256": hashlib.sha256(raw).hexdigest(), "tree_id": None,
            "search": effective_search(spec, XModel(**json.loads(raw)).predict_config)}


def git_state(run_dir):
    """Commit, dirty flag (tracked changes or untracked files) and untracked paths of this checkout (the code that
    runs, whatever the working directory). Tracked changes go to git_diff.patch."""
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, cwd=REPO).stdout
    try:
        commit, diff = git("rev-parse", "HEAD").strip(), git("diff", "HEAD")
        untracked = git("ls-files", "--others", "--exclude-standard").splitlines()
    except OSError:  # no git binary (e.g. a container): recorded as unknown, the run goes on
        commit, diff, untracked = "", "", []
    if not commit:
        logging.getLogger("xmr4el.run_experiment").warning("git commit unknown (no git or not a repository)")
        return {"commit": None, "dirty": None, "untracked": None}
    if diff:
        with open(os.path.join(run_dir, "git_diff.patch"), "w") as f:
            f.write(diff)
    return {"commit": commit, "dirty": bool(diff or untracked), "untracked": untracked}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def provenance(spec, raw):
    """sha256 of every input file (the model config's = of its captured bytes `raw`), interpreter/platform, package
    versions and the torch device of this run."""
    paths = [p for e in spec["evals"] for p in (e["test_path"], e.get("train_path")) if p]
    paths += [spec["train_path"]] if raw is not None else []
    inputs = {p: sha256(p) for p in set(paths)}
    if raw is not None:
        inputs[spec["model_config"]] = hashlib.sha256(raw).hexdigest()

    def pkg(name):
        try:
            return version(name)
        except PackageNotFoundError:
            return None
    return {"inputs": dict(sorted(inputs.items())),
            "python": platform.python_version(), "platform": platform.platform(), "machine": platform.machine(),
            "packages": {name: pkg(name) for name in PACKAGES}, "torch_device": str(torch_device())}


def write_json(path, obj):
    """Write-then-rename: a crash never leaves a partial JSON file."""
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(path + ".tmp", path)


def run(spec, xmodel_path=None, run_dir=None, model_config_sha256=None, tree_id=None):
    """One run of `spec` (dict) in a new run dir (`run_dir`, default outputs/runs/<timestamp>_<name>); returns the run
    dir. A given model_config_sha256 / tree_id must match (BindingRejected before any output). Raises after recording
    a failure."""
    train = xmodel_path is None
    raw = check_spec(spec, train=train, xmodel_path=xmodel_path)
    if train and model_config_sha256 and hashlib.sha256(raw).hexdigest() != model_config_sha256:
        raise BindingRejected({"error": "model_config_changed", "expected": model_config_sha256,
                               "actual": hashlib.sha256(raw).hexdigest()})
    if not train and tree_id and tree_view(xmodel_path)[0] != tree_id:
        raise BindingRejected({"error": "tree_changed", "expected": tree_id, "actual": tree_view(xmodel_path)[0]})

    run_dir = run_dir or os.path.join("outputs", "runs", f"{datetime.now():%Y-%m-%d_%H-%M-%S}_{spec['name']}")
    os.makedirs(run_dir)  # exists -> FileExistsError: runs are never overwritten
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(level=logging.WARNING, format=fmt)
    log_file = logging.FileHandler(os.path.join(run_dir, "run.log"))
    log_file.setFormatter(logging.Formatter(fmt))
    logging.getLogger().addHandler(log_file)
    set_verbosity(1)
    logger = logging.getLogger("xmr4el.run_experiment")
    status = os.path.join(run_dir, "status.json")
    record = {"started": datetime.now().isoformat(timespec="seconds"), **own_identity()}
    beat = threading.Event()
    threading.Thread(target=heartbeat, args=(os.path.join(run_dir, "heartbeat"), beat), daemon=True).start()
    try:
        write_json(status, {"status": "running", **record})
        write_json(os.path.join(run_dir, "spec.json"), {**spec, "xmodel_path": xmodel_path})
        write_json(os.path.join(run_dir, "git.json"), git_state(run_dir))
        write_json(os.path.join(run_dir, "provenance.json"), provenance(spec, raw))
        logger.info("Run started: dir=%s argv=%s", run_dir, " ".join(sys.argv))
        if not train:
            xm, tree_path = XModel.load(xmodel_path), xmodel_path
        else:
            with open(os.path.join(run_dir, "model_config.json"), "wb") as f:
                f.write(raw)
            start = time.perf_counter()
            xm = XModel(**json.loads(raw))
            data = Preprocessor.load_pubtator_file(spec["train_path"], window=xm.context_window,
                                                   abbrev=xm.abbrev_expansion)
            xm.train(*Preprocessor.organize_pubtator_output(data))
            tree_path = xm.save(os.path.join("outputs", "saved_trees"))
            logger.info("Training completed: tree=%s elapsed=%.1fs", tree_path, time.perf_counter() - start)
        write_json(os.path.join(run_dir, "tree.json"), {"xmodel_path": tree_path, "tree_id": xm.tree_id,
                                                        "trained": train})
        tree = os.path.basename(os.path.normpath(tree_path))
        for e in spec["evals"]:
            start = time.perf_counter()
            m, _ = evaluate_tree(xm, tree, e["test_path"], train_path=e.get("train_path"),
                                 beam_size=e.get("beam_size"), topk=e.get("topk"), knn_beta=e.get("knn_beta"),
                                 leaf_candidates=e.get("leaf_candidates"))
            m.update(split=e["split"], eval_seconds=round(time.perf_counter() - start, 1))
            write_json(os.path.join(run_dir, f"metrics_{e['split']}.json"), m)
            logger.info("Eval %s:\n%s", e["split"], format_metrics(m))
            if not m["rows"]:
                raise ValueError(f"eval {e['split']}: no row has a gold label in the tree's vocabulary")
        write_json(status, {"status": "completed", **record, "finished": datetime.now().isoformat(timespec="seconds")})
        logger.info("Run completed: dir=%s", run_dir)
        return run_dir
    except (Stopped, KeyboardInterrupt) as err:
        signame = "SIGTERM" if isinstance(err, Stopped) else "SIGINT"
        write_json(status, {"status": "stopped", **record, "finished": datetime.now().isoformat(timespec="seconds"),
                            "error": signame})
        logger.warning("Run stopped by %s", signame)
        raise
    except BaseException as err:
        write_json(status, {"status": "failed", **record, "finished": datetime.now().isoformat(timespec="seconds"),
                            "error": f"{type(err).__name__}: {err}"})
        logger.exception("Run failed")
        raise
    finally:
        beat.set()
        logging.getLogger().removeHandler(log_file)
        log_file.close()


def heartbeat(path, done):
    """Touch `path` now and every HEARTBEAT seconds until `done` is set (a file of its own: never races status.json)."""
    while True:
        with open(path, "a"):
            os.utime(path)
        if done.wait(HEARTBEAT):
            return


def stop(signum, frame):
    raise Stopped(signum)


def main():
    parser = ArgumentParser()
    parser.add_argument("-spec", required=True, help="configs/experiments/<name>.json, or - for stdin")
    parser.add_argument("-xmodel_path", help="evaluate this saved tree instead of training")
    parser.add_argument("-run_dir", help="run directory to create (must not exist)")
    parser.add_argument("-model_config_sha256", help="refuse (exit 3) unless the model config has this sha256")
    parser.add_argument("-tree_id", help="refuse (exit 3) unless the tree's xmodel.json has this tree_id")
    parser.add_argument("-check", action="store_true", help="validate only; one JSON object on stdout")
    args = parser.parse_args()
    if args.check:
        try:
            with contextlib.redirect_stdout(sys.stderr):  # stdout carries only the response
                response = check(load_spec(args.spec), args.xmodel_path)
        except Exception as err:
            response = {"ok": False, "error": f"{type(err).__name__}: {err}", "model_config_sha256": None,
                        "tree_id": None, "search": {}}
        print(json.dumps(response))
        sys.exit(0 if response["ok"] else 1)
    spec = load_spec(args.spec)
    # SIGTERM (kill, docker stop) -> Stopped, so run() marks the run stopped; SIGKILL still leaves "running"
    signal.signal(signal.SIGTERM, stop)
    try:
        run(spec, args.xmodel_path, args.run_dir, args.model_config_sha256, args.tree_id)
    except BindingRejected as err:
        print(json.dumps(err.args[0]))
        sys.exit(3)
    except Stopped as err:
        sys.exit(128 + err.args[0])


def load_spec(path):
    if path == "-":
        return json.load(sys.stdin)
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    main()

"""One tracked run: train a tree from a spec (or take a saved one) and evaluate it on each listed split.

Spec (JSON): {"name", "model_config", "train_path", "evals": [{"split", "test_path", "train_path"?, "beam_size"?,
"topk"?, "knn_beta"?}]}; an eval's train_path adds the seen/unseen mention-string breakdown, search settings left out
come from the tree's predict_config. Unknown keys are errors. PubTator input only.

Writes outputs/runs/<timestamp>_<name>/ (an existing directory is refused): spec.json (as given + xmodel_path),
model_config.json, git.json (commit, dirty incl. untracked files, untracked paths) + git_diff.patch if tracked files
changed, provenance.json (sha256 of every input file, python/platform/package versions, torch device), run.log,
tree.json (saved tree path), metrics_<split>.json (scripts/evaluate.py's numbers), status.json (running ->
completed | failed with the error). An eval with no row in the tree's vocabulary fails the run. With -xmodel_path:
no training, evals only. `run(spec, xmodel_path)` is the same run without argument parsing.
"""
import hashlib
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from argparse import ArgumentParser
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version

from xmr4el import set_verbosity, torch_device
from xmr4el.data.readers import Preprocessor
from xmr4el.eval import evaluate_tree, format_metrics
from xmr4el.xmodel import XModel, check_search

SPEC_KEYS = {"name", "model_config", "train_path", "evals"}
EVAL_KEYS = {"split", "test_path", "train_path", "beam_size", "topk", "knn_beta"}
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGES = ("numpy", "scipy", "scikit-learn", "torch", "transformers", "sentence-transformers")


def check_spec(spec, train, xmodel_path=None):
    """Raise ValueError on unknown/missing keys, a name that is not [A-Za-z0-9_.-]+, duplicate splits, bad search
    settings, a model config XModel refuses, or a missing input file, before any work."""
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
    if train:
        XModel.load_config(spec["model_config"])  # unknown keys, bad depth


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


def provenance(spec, train):
    """sha256 of every input file, interpreter/platform, package versions and the torch device of this run."""
    paths = [p for e in spec["evals"] for p in (e["test_path"], e.get("train_path")) if p]
    paths += [spec["model_config"], spec["train_path"]] if train else []

    def pkg(name):
        try:
            return version(name)
        except PackageNotFoundError:
            return None
    return {"inputs": {p: sha256(p) for p in sorted(set(paths))},
            "python": platform.python_version(), "platform": platform.platform(), "machine": platform.machine(),
            "packages": {name: pkg(name) for name in PACKAGES}, "torch_device": str(torch_device())}


def write_json(path, obj):
    """Write-then-rename: a crash never leaves a partial JSON file."""
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(path + ".tmp", path)


def run(spec, xmodel_path=None):
    """One run of `spec` (dict) in a new run dir; returns the run dir. Raises after recording a failure."""
    train = xmodel_path is None
    check_spec(spec, train=train, xmodel_path=xmodel_path)

    run_dir = os.path.join("outputs", "runs", f"{datetime.now():%Y-%m-%d_%H-%M-%S}_{spec['name']}")
    os.makedirs(run_dir)  # exists -> FileExistsError: runs are never overwritten
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(level=logging.WARNING, format=fmt)
    log_file = logging.FileHandler(os.path.join(run_dir, "run.log"))
    log_file.setFormatter(logging.Formatter(fmt))
    logging.getLogger().addHandler(log_file)
    set_verbosity(1)
    logger = logging.getLogger("xmr4el.run_experiment")
    status = os.path.join(run_dir, "status.json")
    try:
        write_json(status, {"status": "running", "started": datetime.now().isoformat(timespec="seconds")})
        write_json(os.path.join(run_dir, "spec.json"), {**spec, "xmodel_path": xmodel_path})
        write_json(os.path.join(run_dir, "git.json"), git_state(run_dir))
        write_json(os.path.join(run_dir, "provenance.json"), provenance(spec, train))
        logger.info("Run started: dir=%s argv=%s", run_dir, " ".join(sys.argv))
        if not train:
            xm, tree_path = XModel.load(xmodel_path), xmodel_path
        else:
            shutil.copy(spec["model_config"], os.path.join(run_dir, "model_config.json"))
            start = time.perf_counter()
            xm = XModel.load_config(spec["model_config"])
            data = Preprocessor.load_pubtator_file(spec["train_path"], window=xm.context_window,
                                                   abbrev=xm.abbrev_expansion)
            xm.train(*Preprocessor.organize_pubtator_output(data))
            tree_path = xm.save(os.path.join("outputs", "saved_trees"))
            logger.info("Training completed: tree=%s elapsed=%.1fs", tree_path, time.perf_counter() - start)
        write_json(os.path.join(run_dir, "tree.json"), {"xmodel_path": tree_path, "trained": train})
        tree = os.path.basename(os.path.normpath(tree_path))
        for e in spec["evals"]:
            start = time.perf_counter()
            m, _ = evaluate_tree(xm, tree, e["test_path"], train_path=e.get("train_path"),
                                 beam_size=e.get("beam_size"), topk=e.get("topk"), knn_beta=e.get("knn_beta"))
            m.update(split=e["split"], eval_seconds=round(time.perf_counter() - start, 1))
            write_json(os.path.join(run_dir, f"metrics_{e['split']}.json"), m)
            logger.info("Eval %s:\n%s", e["split"], format_metrics(m))
            if not m["rows"]:
                raise ValueError(f"eval {e['split']}: no row has a gold label in the tree's vocabulary")
        write_json(status, {"status": "completed", "finished": datetime.now().isoformat(timespec="seconds")})
        logger.info("Run completed: dir=%s", run_dir)
        return run_dir
    except BaseException as err:  # KeyboardInterrupt too: the run dir says it did not complete
        write_json(status, {"status": "failed", "finished": datetime.now().isoformat(timespec="seconds"),
                            "error": f"{type(err).__name__}: {err}"})
        logger.exception("Run failed")
        raise
    finally:
        logging.getLogger().removeHandler(log_file)
        log_file.close()


def main():
    parser = ArgumentParser()
    parser.add_argument("-spec", required=True, help="configs/experiments/<name>.json")
    parser.add_argument("-xmodel_path", help="evaluate this saved tree instead of training")
    args = parser.parse_args()
    with open(args.spec) as f:
        spec = json.load(f)
    run(spec, args.xmodel_path)


if __name__ == "__main__":
    main()

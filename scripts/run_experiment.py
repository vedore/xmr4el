"""One tracked run: train a tree from a spec (or take a saved one) and evaluate it on each listed split.

Spec (JSON): {"name", "model_config", "train_path", "evals": [{"split", "test_path", "train_path"?, "beam_size"?,
"topk"?, "knn_beta"?}]}; an eval's train_path adds the seen/unseen mention-string breakdown, search settings left out
come from the tree's predict_config. Unknown keys are errors. PubTator input only.

Writes outputs/runs/<timestamp>_<name>/ (an existing directory is refused): spec.json (as given + xmodel_path),
model_config.json, git.json (commit, dirty) + git_diff.patch if dirty, run.log, tree.json (saved tree path),
metrics_<split>.json (scripts/evaluate.py's numbers). With -xmodel_path: no training, evals only.
"""
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from argparse import ArgumentParser
from datetime import datetime

from xmr4el import set_verbosity
from xmr4el.data.readers import Preprocessor
from xmr4el.eval import evaluate_tree, format_metrics
from xmr4el.xmodel import XModel

SPEC_KEYS = {"name", "model_config", "train_path", "evals"}
EVAL_KEYS = {"split", "test_path", "train_path", "beam_size", "topk", "knn_beta"}
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def check_spec(spec, train):
    """Raise ValueError on unknown/missing keys or duplicate splits, before any work."""
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


def git_state(run_dir):
    """Commit and dirty flag of this checkout (the code that runs, whatever the working directory)."""
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, cwd=REPO).stdout
    try:
        commit, diff = git("rev-parse", "HEAD").strip(), git("diff", "HEAD")
    except OSError:  # no git binary (e.g. a container): recorded as unknown, the run goes on
        commit, diff = "", ""
    if not commit:
        logging.getLogger("xmr4el.run_experiment").warning("git commit unknown (no git or not a repository)")
        return {"commit": None, "dirty": None}
    if diff:
        with open(os.path.join(run_dir, "git_diff.patch"), "w") as f:
            f.write(diff)
    return {"commit": commit, "dirty": bool(diff)}


def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def main():
    parser = ArgumentParser()
    parser.add_argument("-spec", required=True, help="configs/experiments/<name>.json")
    parser.add_argument("-xmodel_path", help="evaluate this saved tree instead of training")
    args = parser.parse_args()
    with open(args.spec) as f:
        spec = json.load(f)
    check_spec(spec, train=args.xmodel_path is None)

    run_dir = os.path.join("outputs", "runs", f"{datetime.now():%Y-%m-%d_%H-%M-%S}_{spec['name']}")
    os.makedirs(run_dir)  # exists -> FileExistsError: runs are never overwritten
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(level=logging.WARNING, format=fmt)
    log_file = logging.FileHandler(os.path.join(run_dir, "run.log"))
    log_file.setFormatter(logging.Formatter(fmt))
    logging.getLogger().addHandler(log_file)
    set_verbosity(1)
    logger = logging.getLogger("xmr4el.run_experiment")
    write_json(os.path.join(run_dir, "spec.json"), {**spec, "xmodel_path": args.xmodel_path})
    write_json(os.path.join(run_dir, "git.json"), git_state(run_dir))
    logger.info("Run started: dir=%s argv=%s", run_dir, " ".join(sys.argv))

    try:
        if args.xmodel_path:
            xm, tree_path = XModel.load(args.xmodel_path), args.xmodel_path
        else:
            shutil.copy(spec["model_config"], os.path.join(run_dir, "model_config.json"))
            start = time.perf_counter()
            xm = XModel.load_config(spec["model_config"])
            data = Preprocessor.load_pubtator_file(spec["train_path"], window=xm.context_window,
                                                   abbrev=xm.abbrev_expansion)
            xm.train(*Preprocessor.organize_pubtator_output(data))
            tree_path = xm.save(os.path.join("outputs", "saved_trees"))
            logger.info("Training completed: tree=%s elapsed=%.1fs", tree_path, time.perf_counter() - start)
        write_json(os.path.join(run_dir, "tree.json"), {"xmodel_path": tree_path, "trained": not args.xmodel_path})
        tree = os.path.basename(os.path.normpath(tree_path))
        for e in spec["evals"]:
            start = time.perf_counter()
            m, _ = evaluate_tree(xm, tree, e["test_path"], train_path=e.get("train_path"),
                                 beam_size=e.get("beam_size"), topk=e.get("topk"), knn_beta=e.get("knn_beta"))
            m.update(split=e["split"], eval_seconds=round(time.perf_counter() - start, 1))
            write_json(os.path.join(run_dir, f"metrics_{e['split']}.json"), m)
            logger.info("Eval %s:\n%s", e["split"], format_metrics(m))
    except Exception:
        logger.exception("Run failed")
        raise
    logger.info("Run completed: dir=%s", run_dir)


if __name__ == "__main__":
    main()

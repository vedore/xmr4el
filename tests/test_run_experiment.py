"""scripts/run_experiment.py end to end on a synthetic tfidf spec: run dir contents, metrics = evaluate.py's,
eval-only reuse of the saved tree, spec checks, no overwrite, empty evals fail the run, log handler cleanup."""
from datetime import datetime
import importlib.util
import hashlib
import json
import logging
from pathlib import Path
import random
import subprocess
import sys

import pytest

from xmr4el.eval import evaluate_tree
from xmr4el.xmodel import XModel

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_experiment.py"
CONFIG = dict(vectorizer_config={"type": "tfidf", "kwargs": {"analyzer": "char", "ngram_range": [2, 4]}},
              dimension_config={"type": "sklearntruncatedsvd", "kwargs": {"n_components": 6, "random_state": 42}},
              clustering_config={"type": "balancedkmeans", "kwargs": {"n_clusters": 2}},
              matcher_config={"type": "jointlogisticregression", "kwargs": {"max_iter": 100}},
              features="tfidf", depth=2, min_leaf_size=2)


def write_pubtator(path, n_docs, seed):
    r = random.Random(seed)
    with open(path, "w") as f:
        for d in range(n_docs):
            j = r.randrange(8)
            m = f"concept{j} syn{r.randrange(6)}"
            title = f"Study of {m} in group{j // 4}"
            s = title.index(m)
            f.write(f"{d}|t|{title}\n{d}|a|Results were clear.\n{d}\t{s}\t{s + len(m)}\t{m}\tDisease\tL{j}\n\n")


def run(tmp, *args):
    return subprocess.run([sys.executable, str(SCRIPT), *args], cwd=tmp, capture_output=True, text=True)


def load_script():
    mod_spec = importlib.util.spec_from_file_location("run_experiment", SCRIPT)
    mod = importlib.util.module_from_spec(mod_spec)
    mod_spec.loader.exec_module(mod)
    return mod


def test_run_experiment(tmp_path):
    write_pubtator(tmp_path / "train.pubtator", 120, 0)
    write_pubtator(tmp_path / "dev.pubtator", 40, 1)
    (tmp_path / "cfg.json").write_text(json.dumps(CONFIG))
    spec = {"name": "syn", "model_config": "cfg.json", "train_path": "train.pubtator",
            "evals": [{"split": "dev", "test_path": "dev.pubtator", "train_path": "train.pubtator"},
                      {"split": "dev_beam1", "test_path": "dev.pubtator", "beam_size": 1, "knn_beta": 2}]}
    (tmp_path / "spec.json").write_text(json.dumps(spec))
    r = run(tmp_path, "-spec", "spec.json")
    assert r.returncode == 0, r.stderr
    (run_dir,) = (tmp_path / "outputs/runs").iterdir()
    assert run_dir.name.endswith("_syn")
    assert {p.name for p in run_dir.iterdir()} >= {"spec.json", "model_config.json", "git.json", "run.log",
                                                   "tree.json", "metrics_dev.json", "metrics_dev_beam1.json",
                                                   "provenance.json", "status.json"}
    assert not list(run_dir.glob("*.tmp"))
    git = json.loads((run_dir / "git.json").read_text())
    assert len(git["commit"]) == 40 and isinstance(git["untracked"], list)
    assert git["dirty"] == bool(git["untracked"] or (run_dir / "git_diff.patch").exists())
    assert json.loads((run_dir / "status.json").read_text())["status"] == "completed"
    prov = json.loads((run_dir / "provenance.json").read_text())
    want_sha = hashlib.sha256((tmp_path / "train.pubtator").read_bytes()).hexdigest()
    assert prov["inputs"]["train.pubtator"] == want_sha and set(prov["inputs"]) == {"train.pubtator",
                                                                                  "dev.pubtator", "cfg.json"}
    assert prov["packages"]["numpy"] and prov["torch_device"]
    tree_path = json.loads((run_dir / "tree.json").read_text())["xmodel_path"]
    dev = json.loads((run_dir / "metrics_dev.json").read_text())
    beam1 = json.loads((run_dir / "metrics_dev_beam1.json").read_text())
    assert beam1["search"]["beam_size"] == 1 and beam1["search"]["knn_beta"] == 2 and "strings" not in beam1
    assert dev["rows"] == dev["rows_total"] == 40 and dev["strings"]["all"]["n"] == 40
    assert "acc@1" in (run_dir / "run.log").read_text(), "eval report in the log"

    # metrics = evaluate.py's function on the saved tree, minus timings
    want, _ = evaluate_tree(XModel.load(str(tmp_path / tree_path)), Path(tree_path).name,
                            str(tmp_path / "dev.pubtator"),
                            train_path=str(tmp_path / "train.pubtator"))
    for k in ("acc@1", "MRR", "R@5", "R@cand", "hybrid", "strings", "search", "rows"):
        assert dev[k] == want[k], k

    # eval only: the saved tree is reused, nothing trained
    r = run(tmp_path, "-spec", "spec.json", "-xmodel_path", tree_path)
    assert r.returncode == 0, r.stderr
    assert len(list((tmp_path / "outputs/saved_trees").iterdir())) == 1
    again = max((tmp_path / "outputs/runs").iterdir())
    assert again != run_dir and not (again / "model_config.json").exists()
    assert json.loads((again / "metrics_dev.json").read_text())["acc@1"] == dev["acc@1"]

    # spec errors before any work
    (tmp_path / "bad.json").write_text(json.dumps({**spec, "extra": 1}))
    r = run(tmp_path, "-spec", "bad.json")
    assert r.returncode != 0 and "unknown ['extra']" in r.stderr
    assert len(list((tmp_path / "outputs/runs").iterdir())) == 2


def test_run_dir_never_overwritten(tmp_path, monkeypatch):
    mod = load_script()
    monkeypatch.setattr(mod, "datetime", type("Fixed", (), {"now": staticmethod(lambda: datetime(2026, 1, 1))}))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "spec.json").write_text(json.dumps({"name": "x", "evals": [{"split": "d", "test_path": "t"}]}))
    (tmp_path / "t").write_text("")
    (tmp_path / "tree").mkdir()
    (tmp_path / "tree/xmodel.pkl").write_text("")
    (tmp_path / "outputs/runs/2026-01-01_00-00-00_x").mkdir(parents=True)
    monkeypatch.setattr(sys, "argv", ["run_experiment.py", "-spec", "spec.json", "-xmodel_path", "tree"])
    with pytest.raises(FileExistsError):
        mod.main()
    assert not list((tmp_path / "outputs/runs/2026-01-01_00-00-00_x").iterdir())


def test_spec_checks(tmp_path, monkeypatch):
    """Bad names, search settings, model configs and missing inputs fail before a run dir exists."""
    mod = load_script()
    monkeypatch.chdir(tmp_path)
    for f in ("cfg.json", "train.pubtator", "dev.pubtator"):
        (tmp_path / f).write_text(json.dumps(CONFIG) if f == "cfg.json" else "")
    good = {"name": "ok", "model_config": "cfg.json", "train_path": "train.pubtator",
            "evals": [{"split": "dev", "test_path": "dev.pubtator"}]}
    mod.check_spec(good, train=True)
    (tmp_path / "zero.json").write_text(json.dumps({**CONFIG, "depth": 0}))
    (tmp_path / "typo.json").write_text(json.dumps({**CONFIG, "dpeth": 2}))
    bad = [({**good, "name": "../x"}, "name/split"),
           ({**good, "evals": [{"split": "a b", "test_path": "dev.pubtator"}]}, "name/split"),
           ({**good, "evals": [{"split": "dev", "test_path": "dev.pubtator", "beam_size": 1.5}]}, "beam_size"),
           ({**good, "evals": [{"split": "dev", "test_path": "dev.pubtator", "topk": -1}]}, "topk"),
           ({**good, "evals": [{"split": "dev", "test_path": "dev.pubtator", "knn_beta": float("nan")}]}, "knn_beta"),
           ({**good, "evals": [{"split": "dev", "test_path": "nope.pubtator"}]}, "nope.pubtator"),
           ({**good, "train_path": "nope.pubtator"}, "nope.pubtator"),
           ({**good, "model_config": "zero.json"}, "depth")]
    for spec, msg in bad:
        with pytest.raises(ValueError, match=msg):
            mod.check_spec(spec, train=True)
    with pytest.raises(TypeError, match="dpeth"):
        mod.check_spec({**good, "model_config": "typo.json"}, train=True)
    with pytest.raises(ValueError, match="no_tree"):
        mod.check_spec({"name": "ok", "evals": good["evals"]}, train=False, xmodel_path="no_tree")
    assert not (tmp_path / "outputs").exists()


def test_empty_eval_fails_run(tmp_path, monkeypatch):
    """An eval with no in-vocabulary row fails the run (status failed, metrics kept); the run log handler is
    removed so a later in-process run does not write into this run's log."""
    mod = load_script()
    monkeypatch.chdir(tmp_path)
    write_pubtator(tmp_path / "train.pubtator", 120, 0)
    (tmp_path / "oov.pubtator").write_text("0|t|Study of zz in x\n0|a|ok.\n0\t9\t11\tzz\tDisease\tNOT_A_LABEL\n\n")
    xm = XModel(**CONFIG)
    data = __import__("xmr4el.data.readers", fromlist=["Preprocessor"]).Preprocessor
    xm.train(*data.organize_pubtator_output(data.load_pubtator_file("train.pubtator")))
    tree = xm.save("trees")
    spec = {"name": "oov", "evals": [{"split": "oov", "test_path": "oov.pubtator"}]}
    with pytest.raises(ValueError, match="no row"):
        mod.run(spec, tree)
    (run_dir,) = (tmp_path / "outputs/runs").iterdir()
    status = json.loads((run_dir / "status.json").read_text())
    assert status["status"] == "failed" and "no row" in status["error"]
    assert json.loads((run_dir / "metrics_oov.json").read_text())["rows"] == 0
    assert not [h for h in logging.getLogger().handlers if isinstance(h, logging.FileHandler)
                and str(run_dir) in h.baseFilename]

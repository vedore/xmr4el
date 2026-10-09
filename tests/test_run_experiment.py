"""scripts/run_experiment.py end to end on a synthetic tfidf spec: run dir contents, metrics = evaluate.py's,
eval-only reuse of the saved tree, spec checks, no overwrite."""
from datetime import datetime
import importlib.util
import json
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
                                                   "tree.json", "metrics_dev.json", "metrics_dev_beam1.json"}
    assert len(json.loads((run_dir / "git.json").read_text())["commit"]) == 40
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
    mod_spec = importlib.util.spec_from_file_location("run_experiment", SCRIPT)
    mod = importlib.util.module_from_spec(mod_spec)
    mod_spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "datetime", type("Fixed", (), {"now": staticmethod(lambda: datetime(2026, 1, 1))}))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "spec.json").write_text(json.dumps({"name": "x", "evals": [{"split": "d", "test_path": "t"}]}))
    (tmp_path / "outputs/runs/2026-01-01_00-00-00_x").mkdir(parents=True)
    monkeypatch.setattr(sys, "argv", ["run_experiment.py", "-spec", "spec.json", "-xmodel_path", "tree"])
    with pytest.raises(FileExistsError):
        mod.main()
    assert not list((tmp_path / "outputs/runs/2026-01-01_00-00-00_x").iterdir())

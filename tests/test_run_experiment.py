"""scripts/run_experiment.py end to end on a synthetic tfidf spec: run dir contents, metrics = evaluate.py's,
eval-only reuse of the saved tree, spec checks, no overwrite, empty evals fail the run, log handler cleanup,
-check responses, -run_dir and the reviewed config/tree bindings (exit 3)."""
from datetime import datetime
import importlib.util
import hashlib
import json
import logging
from pathlib import Path
import os
import random
import signal
import subprocess
import sys
from types import SimpleNamespace

import pytest

from xmr4el.eval import evaluate_tree
from xmr4el.xmodel import XModel
from xmr4el_tui import progress, runs

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


def run(tmp, *args, stdin=None):
    return subprocess.run([sys.executable, str(SCRIPT), *args], cwd=tmp, capture_output=True, text=True,
                          input=stdin)


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
    r = run(tmp_path, "-spec", "spec.json", "-check")
    assert r.returncode == 0 and not (tmp_path / "outputs").exists(), r.stderr
    checked = json.loads(r.stdout)
    config_sha = hashlib.sha256((tmp_path / "cfg.json").read_bytes()).hexdigest()
    assert checked["ok"] and checked["model_config_sha256"] == config_sha and checked["tree_id"] is None
    r = run(tmp_path, "-spec", "spec.json", "-model_config_sha256", "0" * 64)
    assert r.returncode == 3 and not (tmp_path / "outputs").exists(), r.stderr
    assert json.loads(r.stdout) == {"error": "model_config_changed", "expected": "0" * 64, "actual": config_sha}
    r = run(tmp_path, "-spec", "spec.json", "-model_config_sha256", config_sha)
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
    assert dev["strings"]["all"]["routed"] == dev["R@cand"] and dev["ms_per_query"] >= 0
    assert dev["acc@1_all"] == dev["acc@1"] and dev["hybrid_all"] == dev["hybrid"]  # every row in vocabulary
    assert checked["search"] == {"dev": dev["search"], "dev_beam1": beam1["search"]}
    assert (run_dir / "model_config.json").read_bytes() == (tmp_path / "cfg.json").read_bytes()
    assert "acc@1" in (run_dir / "run.log").read_text(), "eval report in the log"
    assert "'stage'" not in (run_dir / "run.log").read_text(), "progress events stay out of run.log"
    events, _ = progress.read_events(run_dir / "events.jsonl")
    stages = progress.stages(events, "train", ["dev", "dev_beam1"], "completed")
    assert [x.key for x in stages] == ["load", "encode", "labels", "hierarchy", "save", "eval:dev", "eval:dev_beam1"]
    assert all(x.state == "done" and x.seconds is not None for x in stages), stages
    assert [x.key for x in stages[1].parts] == ["mention_tfidf"] and stages[1].detail == "120 rows"
    assert stages[2].detail == "8 labels" and stages[3].detail.startswith("layer 2/2")
    assert stages[5].detail == f"acc@1 {dev['acc@1']:.4f}"
    assert [p.key for p in stages[5].parts] == ["data", "predict", "metrics", "write"]
    assert all(p.state == "done" for p in stages[5].parts) and stages[0].seconds == events[2]["duration_seconds"]
    assert all(e["v"] == 1 and datetime.fromisoformat(e["t"]).tzinfo for e in events)
    assert (events[0]["stage"], events[0]["phase"], events[0]["kind"], events[0]["splits"]) == \
        ("run", "start", "train", ["dev", "dev_beam1"]) and events[0]["run_id"] == run_dir.name
    assert (events[-1]["stage"], events[-1]["phase"]) == ("run", "complete") and events[-1]["duration_seconds"] > 0
    usage = json.loads((run_dir / "resources.json").read_text())
    assert [(x["stage"], x.get("split"), x["status"]) for x in usage["stages"]] == [
        ("load", None, "completed"), ("encode", None, "completed"), ("labels", None, "completed"),
        ("hierarchy", None, "completed"), ("save", None, "completed"), ("eval", "dev", "completed"),
        ("eval", "dev_beam1", "completed")]
    assert all(x["samples"] >= 2 and x["resources"]["rss_peak_bytes"] > 0 for x in usage["stages"])

    # metrics = evaluate.py's function on the saved tree, minus timings
    want, _ = evaluate_tree(XModel.load(str(tmp_path / tree_path)), Path(tree_path).name,
                            str(tmp_path / "dev.pubtator"),
                            train_path=str(tmp_path / "train.pubtator"))
    for k in ("acc@1", "MRR", "R@5", "R@cand", "hybrid", "strings", "search", "rows"):
        assert dev[k] == want[k], k

    # eval only, spec on stdin: the saved tree is reused, nothing trained
    tree_id = json.loads((run_dir / "tree.json").read_text())["tree_id"]
    r = run(tmp_path, "-spec", "-", "-xmodel_path", tree_path, "-check", stdin=json.dumps(spec))
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout) == {**checked, "model_config_sha256": None, "tree_id": tree_id}
    r = run(tmp_path, "-spec", "-", "-xmodel_path", tree_path, "-tree_id", "other", stdin=json.dumps(spec))
    assert r.returncode == 3 and json.loads(r.stdout)["error"] == "tree_changed", r.stderr
    again = tmp_path / "outputs/runs/given"
    r = run(tmp_path, "-spec", "-", "-xmodel_path", tree_path, "-tree_id", tree_id, "-run_dir", str(again),
            stdin=json.dumps(spec))
    assert r.returncode == 0, r.stderr
    assert len(list((tmp_path / "outputs/saved_trees").iterdir())) == 1
    assert len(list((tmp_path / "outputs/runs").iterdir())) == 2 and not (again / "model_config.json").exists()
    events, _ = progress.read_events(again / "events.jsonl")
    assert [(x.key, x.state) for x in progress.stages(events, "eval", ["dev", "dev_beam1"], "completed")] == \
        [("load", "done"), ("eval:dev", "done"), ("eval:dev_beam1", "done")]
    again_dev = json.loads((again / "metrics_dev.json").read_text())
    assert again_dev["acc@1"] == dev["acc@1"] and again_dev["population_sha256"] == dev["population_sha256"]

    # the TUI reads both real run dirs
    trained, evaluated = sorted(runs.scan(tmp_path), key=lambda x: x.kind == "eval")
    assert (trained.kind, evaluated.kind, trained.recorded) == ("train", "eval", "completed")
    assert trained.tree_id == evaluated.tree_id and trained.tree_here and not trained.problems
    assert [e.split for e in trained.evals] == ["dev", "dev_beam1"] and trained.evals[1].overrides["beam_size"] == 1
    assert runs.training_run([trained, evaluated], evaluated.tree_id) == trained

    # spec errors before any work
    (tmp_path / "bad.json").write_text(json.dumps({**spec, "extra": 1}))
    r = run(tmp_path, "-spec", "bad.json")
    assert r.returncode != 0 and "unknown ['extra']" in r.stderr
    assert len(list((tmp_path / "outputs/runs").iterdir())) == 2


def test_check_errors(tmp_path):
    """Invalid input: exit 1 and one JSON object on stdout, no run dir; an existing -run_dir is refused."""
    (tmp_path / "t").write_text("")
    (tmp_path / "broken.json").write_text("{")
    (tmp_path / "tree").mkdir()
    (tmp_path / "tree/xmodel.pkl").write_text("")
    spec = {"name": "x", "evals": [{"split": "d", "test_path": "t"}]}
    for args, stdin, msg in [(["-spec", "broken.json"], None, "JSONDecodeError"),
                             (["-spec", "-"], json.dumps({**spec, "extra": 1}), "unknown ['extra']"),
                             (["-spec", "-", "-xmodel_path", "tree"], json.dumps(spec), "effective search")]:
        r = run(tmp_path, *args, "-check", stdin=stdin)
        response = json.loads(r.stdout)
        assert r.returncode == 1 and not response["ok"] and msg in response["error"], r.stdout
        assert response["search"] == {} and response["model_config_sha256"] is response["tree_id"] is None
    (tmp_path / "tree/xmodel.json").write_text(json.dumps({"tree_id": "abc", "predict_config": {"beam_size": 5}}))
    r = run(tmp_path, "-spec", "-", "-xmodel_path", "tree", "-check", stdin=json.dumps(spec))
    assert json.loads(r.stdout)["search"] == {"d": {"beam_size": 5, "topk": 0, "knn_beta": 0.0,
                                                    "leaf_candidates": 100}}
    (tmp_path / "exists").mkdir()
    r = run(tmp_path, "-spec", "-", "-xmodel_path", "tree", "-run_dir", "exists", stdin=json.dumps(spec))
    assert r.returncode != 0 and "FileExistsError" in r.stderr and not list((tmp_path / "exists").iterdir())
    assert not (tmp_path / "outputs").exists()


def test_sigterm_records_stopped(tmp_path, monkeypatch):
    """SIGTERM during the run -> status stopped (identity and started kept), exit 128 + 15, heartbeat touched."""
    mod = load_script()
    monkeypatch.chdir(tmp_path)
    (tmp_path / "t").write_text("")
    (tmp_path / "tree").mkdir()
    (tmp_path / "tree/xmodel.pkl").write_text("")
    (tmp_path / "spec.json").write_text(json.dumps({"name": "x", "evals": [{"split": "d", "test_path": "t"}]}))
    monkeypatch.setattr(mod.XModel, "load", staticmethod(lambda path: SimpleNamespace(tree_id="abc")))
    monkeypatch.setattr(mod, "evaluate_tree", lambda *a, **k: os.kill(os.getpid(), signal.SIGTERM))
    monkeypatch.setattr(sys, "argv", ["run_experiment.py", "-spec", "spec.json", "-xmodel_path", "tree",
                                      "-run_dir", "r"])
    old = signal.getsignal(signal.SIGTERM)
    try:
        with pytest.raises(SystemExit) as stop:
            mod.main()
    finally:
        signal.signal(signal.SIGTERM, old)
    assert stop.value.code == 128 + signal.SIGTERM
    status = json.loads((tmp_path / "r/status.json").read_text())
    assert status["status"] == "stopped" and status["error"] == "SIGTERM" and status["started"]
    assert status["pid"] == os.getpid() and status["start"] and status["runtime"] and "isolated_group" in status
    assert (tmp_path / "r/heartbeat").exists() and "stopped by SIGTERM" in (tmp_path / "r/run.log").read_text()
    run = runs.read_run(tmp_path / "r", tmp_path)
    assert run.recorded == "stopped" and run.identity["pid"] == os.getpid()


def eval_only(tmp_path, monkeypatch, mod, evaluate):
    """An eval-only run of a stub tree in tmp_path/r through main(): evaluate_tree replaced by `evaluate`."""
    tmp_path.mkdir(exist_ok=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(mod, "format_metrics", str)
    (tmp_path / "t").write_text("")
    (tmp_path / "tree").mkdir()
    (tmp_path / "tree/xmodel.pkl").write_text("")
    (tmp_path / "spec.json").write_text(json.dumps({"name": "x", "evals": [{"split": "a", "test_path": "t"},
                                                                          {"split": "b", "test_path": "t"}]}))
    monkeypatch.setattr(mod.XModel, "load", staticmethod(lambda path: SimpleNamespace(tree_id="abc")))
    monkeypatch.setattr(mod, "evaluate_tree", evaluate)
    monkeypatch.setattr(sys, "argv", ["run_experiment.py", "-spec", "spec.json", "-xmodel_path", "tree",
                                      "-run_dir", "r", "-resource_interval", "0"])
    old = signal.getsignal(signal.SIGTERM)
    try:
        mod.main()
    finally:
        signal.signal(signal.SIGTERM, old)


def test_failed_stage_and_stopped_split_keep_earlier_results(tmp_path, monkeypatch):
    """A failure in split b: b's stage and the run end failed with the error, a's metrics stay; a stop in b ends
    as stopped (not failed), a later SIGTERM is ignored."""
    mod = load_script()
    ok = {"rows": 1, "acc@1": 0.5}

    def failing(xm, tree, test_path, events=None, **k):
        if events["split"] == "b":
            raise RuntimeError("boom")
        return dict(ok), None
    root = tmp_path
    tmp_path = root / "fail"
    with pytest.raises(RuntimeError):
        eval_only(tmp_path, monkeypatch, mod, failing)
    events, _ = progress.read_events(tmp_path / "r/events.jsonl")
    assert [(e["stage"], e.get("split"), e["phase"]) for e in events if e["phase"] != "start" and "part" not in e] \
        == [("load", None, "complete"), ("eval", "a", "complete"), ("eval", "b", "failed"), ("run", None, "failed")]
    assert events[-1]["error"] == "RuntimeError: boom" and json.loads((tmp_path / "r/metrics_a.json").read_text())
    got = progress.stages(events, "eval", ["a", "b"], "failed")
    assert [(x.key, x.state) for x in got] == [("load", "done"), ("eval:a", "done"), ("eval:b", "failed")]
    assert got[2].error == "RuntimeError: boom" and not (tmp_path / "r/resources.json").exists(), "0 = off"

    def stopping(xm, tree, test_path, events=None, **k):
        if events["split"] == "b":
            os.kill(os.getpid(), signal.SIGTERM)
        return dict(ok), None
    tmp_path = root / "stop"
    with pytest.raises(SystemExit) as stop:
        eval_only(tmp_path, monkeypatch, mod, stopping)
    assert stop.value.code == 128 + signal.SIGTERM
    assert json.loads((tmp_path / "r/status.json").read_text())["status"] == "stopped"
    events, _ = progress.read_events(tmp_path / "r/events.jsonl")
    assert [(e["stage"], e.get("split"), e["phase"]) for e in events[-2:]] == [("eval", "b", "stopped"),
                                                                                ("run", None, "stopped")]
    assert events[-1]["signal"] == "SIGTERM" and (tmp_path / "r/metrics_a.json").exists()
    assert mod.ENDING.is_set() and mod.stop(signal.SIGTERM, None) is None, "a repeated SIGTERM raises nothing"


def test_sigterm_after_completion_keeps_completed(tmp_path, monkeypatch):
    """A stop that lands once the run decided it completed (stop racing completion) changes nothing."""
    mod = load_script()

    class KillOnCompleted(logging.Handler):
        def emit(self, record):
            if record.getMessage().startswith("Run completed"):
                os.kill(os.getpid(), signal.SIGTERM)
    handler = KillOnCompleted()
    logging.getLogger("xmr4el.run_experiment").addHandler(handler)
    try:
        eval_only(tmp_path, monkeypatch, mod, lambda *a, **k: ({"rows": 1}, None))
    finally:
        logging.getLogger("xmr4el.run_experiment").removeHandler(handler)
    assert json.loads((tmp_path / "r/status.json").read_text())["status"] == "completed"
    events, _ = progress.read_events(tmp_path / "r/events.jsonl")
    assert (events[-1]["stage"], events[-1]["phase"]) == ("run", "complete")


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
           ({**good, "model_config": "zero.json"}, "depth"),
           ({**good, "evals": [{"split": "dev", "test_path": "./train.pubtator"}]}, "training file")]
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
    ends = [(e["stage"], e["phase"]) for e in map(json.loads, (run_dir / "events.jsonl").read_text().splitlines())
            if e.get("part") is None and e["phase"] != "start"]
    assert ends[-2:] == [("eval", "failed"), ("run", "failed")], "the zero-row eval is the failed stage"
    assert json.loads((run_dir / "resources.json").read_text())["stages"][-1]["status"] == "failed"
    assert not [h for h in logging.getLogger().handlers if isinstance(h, logging.FileHandler)
                and str(run_dir) in h.baseFilename]


def test_input_changed_after_fingerprint_fails_eval(tmp_path, monkeypatch):
    """The test file is rewritten after provenance.json recorded its sha256: no metrics are written (R07)."""
    mod = load_script()

    def rewriting(xm, tree, test_path, events=None, **k):
        Path(test_path).write_text("other bytes")
        return {"rows": 1}, None
    with pytest.raises(ValueError, match="input changed during the run"):
        eval_only(tmp_path, monkeypatch, mod, rewriting)
    assert not (tmp_path / "r/metrics_a.json").exists()
    assert json.loads((tmp_path / "r/status.json").read_text())["status"] == "failed"

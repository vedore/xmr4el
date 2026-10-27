import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from fake_runs import make_run
from xmr4el_tui import fmt, runs


def test_scan_states_and_fields(tmp_path):
    make_run(tmp_path, "ncbi_disease", tree_here=True)
    make_run(tmp_path, "ncbi_disease_test", stamp="2026-10-20_13-38-45", kind="eval", splits=("test",))
    make_run(tmp_path, "chem", stamp="2026-10-23_14-10-00", status="running", tree_id=None, metrics={"dev": None})
    make_run(tmp_path, "bad", stamp="2026-10-19_10-00-00", status="failed", metrics={"dev": None, "test": {}}, tree_id="dead",
             splits=("dev", "test"))
    make_run(tmp_path, "broken", stamp="2026-10-18_10-00-00", corrupt=("status.json", "metrics_dev.json"),
             tree_id="other")
    (tmp_path / "outputs/runs/.console").mkdir()
    found = runs.scan(tmp_path)
    assert [r.name for r in found] == ["chem", "ncbi_disease_test", "ncbi_disease", "bad", "broken"]
    chem, test, train, bad, broken = found
    assert (chem.recorded, chem.kind, chem.tree_id, chem.started) == ("running", "train", None,
                                                                      datetime(2026, 10, 23, 14, 10))
    assert chem.evals[0].metrics is None and not chem.problems
    assert (test.kind, train.kind, train.tree_here, test.tree_here, bad.tree_here) == ("eval", "train", True, True, False)
    assert train.started == datetime(2026, 10, 20, 13, 36, 20)  # completed status.json has no `started`
    assert train.evals[0].metrics["acc@1"] == 0.9246 and train.identity is None
    assert (bad.recorded, bad.error, [e.metrics is None for e in bad.evals]) == ("failed", "ValueError: boom",
                                                                                 [True, False])
    assert broken.recorded == "unknown"
    assert set(broken.problems) == {"status.json: invalid JSON", "metrics_dev.json: invalid JSON"}
    assert runs.same_tree(found, train) == [test] and runs.training_run(found, test.tree_id) == train


def test_missing_spec_and_status(tmp_path):
    run = make_run(tmp_path, "x")
    (run / "spec.json").unlink()
    (run / "status.json").unlink()
    (r,) = runs.scan(tmp_path)
    assert (r.name, r.kind, r.recorded) == ("x", "train", "unknown")  # kind from tree.json
    assert [e.split for e in r.evals] == ["dev"] and r.evals[0].test_path == "datasets/x/dev.pubtator"
    assert set(r.problems) == {"status.json: missing", "spec.json: missing"}


def test_scan_cache(tmp_path):
    run = make_run(tmp_path, "x")
    cache = {}
    (first,) = runs.scan(tmp_path, cache)
    (again,) = runs.scan(tmp_path, cache)
    assert again == first
    tree = json.loads((run / "tree.json").read_text())["xmodel_path"]
    (tmp_path / tree).mkdir(parents=True)
    (tmp_path / tree / "xmodel.pkl").write_bytes(b"")  # tree synced later: seen without a run change
    assert runs.scan(tmp_path, cache)[0].tree_here
    (run / "status.json").write_text(json.dumps({"status": "failed", "error": "E"}))
    assert runs.scan(tmp_path, cache)[0].recorded == "failed"
    make_run(tmp_path, "y", stamp="2026-10-21_00-00-00")
    shutil.rmtree(run)
    assert [r.name for r in runs.scan(tmp_path, cache)] == ["y"] and set(cache) == {"2026-10-21_00-00-00_y"}


def test_previous_completed(tmp_path):
    make_run(tmp_path, "a", stamp="2026-10-01_00-00-00")
    make_run(tmp_path, "a", stamp="2026-10-02_00-00-00")
    make_run(tmp_path, "a", stamp="2026-10-03_00-00-00", status="failed")
    make_run(tmp_path, "a", stamp="2026-10-04_00-00-00", status="running")
    found = runs.scan(tmp_path)
    assert runs.previous_completed(found, found[0]).id == "2026-10-02_00-00-00_a"
    assert runs.previous_completed(found, found[-1]) is None


def test_normalize_spec_round_trip(tmp_path):
    train = json.loads((make_run(tmp_path, "t") / "spec.json").read_text())
    ev = json.loads((make_run(tmp_path, "e", kind="eval") / "spec.json").read_text())
    spec, path = runs.normalize_spec(train)
    assert path is None and "xmodel_path" not in spec and "xmodel_path" in train
    spec, path = runs.normalize_spec({**ev, "extra": 1})
    assert path == ev["xmodel_path"] and spec["extra"] == 1  # unknown keys left for the runner
    cmd = runs.command_line(spec, path)
    assert cmd.startswith(f"python scripts/run_experiment.py -spec - -xmodel_path {path} <<'EOF'\n")
    assert json.loads(cmd.split("\n", 1)[1].rsplit("\nEOF", 1)[0]) == spec


def test_fmt(monkeypatch):
    assert (fmt.duration(59), fmt.duration(127), fmt.duration(3600 + 62), fmt.duration(None)) == (
        "0:59", "2:07", "1:01:02", "—")
    assert fmt.stamp(datetime(2026, 10, 20, 14, 41)) == "10-20 14:41"
    assert fmt.stamp(datetime(2026, 10, 20, 14, 41), long=True) == "2026-10-20 14:41"
    assert (fmt.metric(0.92461), fmt.count(179696), fmt.pct(729, 757), fmt.ms(2.994)) == (
        "0.9246", "179,696", "96.3%", "2.99 ms/query")
    assert (fmt.tree("f15e98b4ad2b"), fmt.marker("current"), fmt.marker("current", ascii=True)) == (
        "f15e98b4", "▸", ">>")
    monkeypatch.setenv("NO_COLOR", "1")
    assert fmt.use_ascii()


def test_helpers_need_only_the_standard_library():
    """The non-UI modules load under `python -S` (no site-packages: no Textual, no backend, no torch); the runner
    imports procinfo without the tui extra."""
    mods = ", ".join(f"xmr4el_tui.{m}" for m in ("runs", "progress", "explorer", "compare", "fmt", "process",
                                                  "procinfo"))
    subprocess.run([sys.executable, "-S", "-c", f"import {mods}"], cwd=Path(__file__).parents[2], check=True)

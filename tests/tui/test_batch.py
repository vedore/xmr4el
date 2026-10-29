"""Batch backend (tui.md §5.16): snapshot, coordinator (sequential, continue on failure, integrity, stop), locks,
detachment and lost detection. A fake runner stands in for scripts/run_experiment.py: no model is trained."""
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from xmr4el_tui import batch, coordinator, procinfo, review
from xmr4el_tui.datasets import EvalInput, Selection

REPO = Path(__file__).parents[2]
FAKE = f'''
import argparse, hashlib, json, os, pathlib, signal, sys, time
sys.path.insert(0, {str(REPO)!r})
from xmr4el_tui import procinfo
a = argparse.ArgumentParser()
a.add_argument("-spec"); a.add_argument("-run_dir"); a.add_argument("-model_config_sha256")
args = a.parse_args()
spec = json.load(sys.stdin)
cfg = json.load(open(spec["model_config"]))
if os.path.exists("fake_ctl.json"):  # tests that cannot put fake keys into a valid model config
    cfg = {{**cfg, **json.load(open("fake_ctl.json"))}}
mode, secs = cfg.get("mode", "ok"), cfg.get("seconds", 0.2)
run = pathlib.Path(args.run_dir)
run.mkdir(parents=True)
if mode == "mutate_before_read":
    with open(spec["train_path"], "a") as f:
        f.write("changed before the runner captured its inputs")
# Structural fixtures only: no scientific metrics or trained model are invented.
(run / "spec.json").write_text(json.dumps(spec))
(run / "model_config.json").write_text(json.dumps(cfg))
paths = {{spec["train_path"], *(e["test_path"] for e in spec["evals"])}}
(run / "provenance.json").write_text(json.dumps({{"inputs": {{
    p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() for p in paths}}}}))
def status(state, **kw):
    (run / "status.json.tmp").write_text(json.dumps({{"status": state, "started": "2026-10-29T10:00:00", **procinfo.own_identity(), **kw}}))
    os.replace(run / "status.json.tmp", run / "status.json")
def log(what):
    with open("order.log", "a") as f:
        f.write(f"{{what}} {{spec['name']}}\\n")
def stopped(*_):
    status("stopped", error="SIGTERM", finished="2026-10-29T10:00:01"); log("end"); os._exit(143)
if mode != "ignore_term":
    signal.signal(signal.SIGTERM, stopped)
else:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
status("running"); log("start")
time.sleep(secs)
if mode == "fail":
    status("failed", error="boom", finished="x"); log("end"); sys.exit(1)
if mode == "mutate":
    with open(spec["train_path"], "a") as f:
        f.write("changed")
signal.signal(signal.SIGTERM, signal.SIG_IGN)  # the run is over: a late stop must not rewrite its outcome
tree = pathlib.Path("outputs/saved_trees") / ("synthetic_" + spec["name"])
tree.mkdir(parents=True)
(tree / "xmodel.pkl").write_bytes(b"synthetic fixture; never load")
(run / "tree.json").write_text(json.dumps({{"tree_id": spec["name"], "xmodel_path": str(tree), "trained": True}}))
for e in spec["evals"]:
    (run / ("metrics_" + e["split"] + ".json")).write_text("{{}}")
status("completed", finished="x"); log("end")
if mode == "stop_after":  # the stop request arrives right after this tree, before the next one starts
    for p in pathlib.Path("outputs/batches").glob("*/coordinator.lock"):
        (p.parent / "stop").touch()
'''
PUB = "11|t|Heart attack risk\n"


@pytest.fixture
def root(tmp_path, monkeypatch):
    (tmp_path / "outputs/runs").mkdir(parents=True)
    (tmp_path / "datasets/A").mkdir(parents=True)
    for f in ("train.pubtator", "dev.pubtator"):
        (tmp_path / "datasets/A" / f).write_text(PUB)
    (tmp_path / "fake_runner.py").write_text(FAKE)
    monkeypatch.setattr(coordinator, "POLL", 0.05)
    return tmp_path


def make_plan(*modes, batch_id="b1"):
    sel = Selection("datasets/A", "A", "", "datasets/A/train.pubtator",
                    (EvalInput("dev", "datasets/A/dev.pubtator", True),))
    trees = tuple(review.PlannedTree(f"t{i + 1}", f"Tree {i + 1}", m if isinstance(m, dict) else {"mode": m})
                  for i, m in enumerate(modes))
    return review.Plan(batch_id, "Batch one", sel, trees)


def run_inline(root, plan, monkeypatch, *, grace=None, stop_after=None):
    """prepare + coordinator.main in this process (signal handlers restored afterwards)."""
    bdir = batch.prepare(root, plan)
    monkeypatch.chdir(root)
    monkeypatch.setenv("XMR4EL_TEST_RUNNER", str(root / "fake_runner.py"))
    if grace:
        monkeypatch.setattr(coordinator, "GRACE", grace)
    saved = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)}
    if stop_after:
        threading.Timer(stop_after, lambda: (bdir / "stop").touch()).start()
    try:
        code = coordinator.main(["coordinator", str(bdir.relative_to(root))])
    finally:
        for s, h in saved.items():
            signal.signal(s, h)
    return code, bdir, batch.read_status(bdir)


def states(status):
    return [t["state"] for t in status["trees"]]


def order(root):
    return (root / "order.log").read_text().split("\n")[:-1]


# --- snapshot -----------------------------------------------------------------------------------------------------------
def test_prepare_writes_the_immutable_plan(root):
    plan = make_plan("ok", "fail")
    bdir = batch.prepare(root, plan)
    plan_json = json.loads((bdir / "batch.json").read_text())
    assert plan_json["policy"] == "continue_on_tree_failure"
    assert list(plan_json["inputs"]) == ["datasets/A/dev.pubtator", "datasets/A/train.pubtator"]
    assert plan_json["inputs"]["datasets/A/train.pubtator"] == batch.file_sha256(root / "datasets/A/train.pubtator")
    assert [t["id"] for t in plan_json["trees"]] == ["t1", "t2"]
    t1 = plan_json["trees"][0]
    assert json.loads((root / t1["config_path"]).read_text()) == {"mode": "ok"}
    assert t1["spec"] == review.run_spec(plan, plan.trees[0], t1["config_path"])
    assert t1["spec"]["evals"] == [{"split": "dev", "test_path": "datasets/A/dev.pubtator",
                                    "train_path": "datasets/A/train.pubtator"}]
    st = batch.read_status(bdir)
    assert st["state"] == "preparing" and states(st) == ["pending", "pending"] and st["coordinator"] is None
    assert st["dataset"] == "A"
    with pytest.raises(batch.BatchError):  # never overwritten
        batch.prepare(root, plan)
    assert json.loads((bdir / "batch.json").read_text()) == plan_json


def test_prepare_leaves_nothing_on_a_missing_input(root):
    (root / "datasets/A/dev.pubtator").unlink()
    with pytest.raises(batch.BatchError):
        batch.prepare(root, make_plan("ok"))
    assert not (root / "outputs/batches/b1").exists()


def test_status_write_is_atomic(root):
    bdir = batch.prepare(root, make_plan("ok"))
    st = batch.read_status(bdir)
    batch.write_status(bdir, {**st, "state": "running"})
    assert not list(bdir.glob("*.tmp")) and batch.read_status(bdir)["state"] == "running"


# --- execution ---------------------------------------------------------------------------------------------------------
def test_trees_run_in_order_one_at_a_time(root, monkeypatch):
    code, bdir, st = run_inline(root, make_plan("ok", "ok", "ok"), monkeypatch)
    assert code == 0 and st["state"] == "completed" and states(st) == ["succeeded"] * 3
    assert [l.split()[0] for l in order(root)] == ["start", "end"] * 3  # never two running
    assert [l.split("_")[-1] for l in order(root)[::2]] == ["t1", "t2", "t3"]
    runs = [t["run_id"] for t in st["trees"]]
    assert all((root / "outputs/runs" / r / "status.json").is_file() for r in runs)
    assert all(r.endswith(f"b1_{t['id']}") for r, t in zip(runs, st["trees"]))
    assert batch.holder(root) is None and not batch.alive(bdir)


def test_a_failed_tree_does_not_stop_the_batch(root, monkeypatch):
    _, _, st = run_inline(root, make_plan("ok", "fail", "ok"), monkeypatch)
    assert states(st) == ["succeeded", "failed", "succeeded"]
    assert st["state"] == "completed_with_failures" and "boom" in st["trees"][1]["error"]
    assert st["trees"][1]["exit_code"] == 1


def test_one_tree_batch(root, monkeypatch):
    _, _, st = run_inline(root, make_plan("ok"), monkeypatch)
    assert st["state"] == "completed" and states(st) == ["succeeded"]


def test_runner_that_cannot_start_fails_only_its_tree(root, monkeypatch):
    plan = make_plan("ok", "ok")
    bdir = batch.prepare(root, plan)
    (root / "outputs/runs/.console").mkdir()
    # the second tree's run id is used before: spawn refuses (FileExistsError), the first still runs
    monkeypatch.chdir(root)
    monkeypatch.setenv("XMR4EL_TEST_RUNNER", str(root / "fake_runner.py"))
    real = coordinator.process.spawn
    calls = []

    def flaky(*a, **kw):
        calls.append(a[2])
        if len(calls) == 2:
            raise FileExistsError("used")
        return real(*a, **kw)
    monkeypatch.setattr(coordinator.process, "spawn", flaky)
    saved = signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGHUP)
    try:
        coordinator.main(["c", str(bdir.relative_to(root))])
    finally:
        for s, h in zip((signal.SIGTERM, signal.SIGINT, signal.SIGHUP), saved):
            signal.signal(s, h)
    st = batch.read_status(bdir)
    assert states(st) == ["succeeded", "failed"] and st["state"] == "completed_with_failures"


# --- integrity ----------------------------------------------------------------------------------------------------------
def test_changed_input_blocks_the_rest_and_keeps_finished_runs(root, monkeypatch):
    _, _, st = run_inline(root, make_plan("ok", "mutate", "ok", "ok"), monkeypatch)
    assert st["state"] == "blocked" and "datasets/A/train.pubtator" in st["error"]
    assert states(st) == ["succeeded", "succeeded", "cancelled", "cancelled"]
    assert len([l for l in order(root) if l.startswith("start")]) == 2
    assert all((root / "outputs/runs" / t["run_id"] / "status.json").is_file() for t in st["trees"][:2])


def test_edited_plan_blocks_before_any_run(root, monkeypatch):
    plan = make_plan("ok")
    bdir = batch.prepare(root, plan)
    (bdir / "batch.json").write_text((bdir / "batch.json").read_text().replace("Batch one", "Edited"))
    monkeypatch.chdir(root)
    monkeypatch.setenv("XMR4EL_TEST_RUNNER", str(root / "fake_runner.py"))
    assert coordinator.main(["c", str(bdir.relative_to(root))]) == 0
    st = batch.read_status(bdir)
    assert st["state"] == "blocked" and not (root / "order.log").exists()


# --- stop ----------------------------------------------------------------------------------------------------------------
def test_stop_during_training(root, monkeypatch):
    _, bdir, st = run_inline(root, make_plan("ok", {"mode": "ok", "seconds": 30}, "ok"), monkeypatch, stop_after=1.5)
    assert st["state"] == "stopped" and states(st) == ["succeeded", "cancelled", "cancelled"]
    assert st["stop_requested"] and len([l for l in order(root) if l.startswith("start")]) == 2
    run = json.loads((root / "outputs/runs" / st["trees"][1]["run_id"] / "status.json").read_text())
    assert run["status"] == "stopped"


def test_stop_between_trees(root, monkeypatch):
    _, _, st = run_inline(root, make_plan("stop_after", "ok", "ok"), monkeypatch)
    assert st["state"] == "stopped" and states(st) == ["succeeded", "cancelled", "cancelled"]
    assert len([l for l in order(root) if l.startswith("start")]) == 1


def test_stop_before_the_first_tree(root, monkeypatch):
    bdir = batch.prepare(root, make_plan("ok"))
    (bdir / "stop").touch()
    monkeypatch.chdir(root)
    monkeypatch.setenv("XMR4EL_TEST_RUNNER", str(root / "fake_runner.py"))
    coordinator.main(["c", str(bdir.relative_to(root))])
    assert batch.read_status(bdir)["state"] == "stopped" and not (root / "order.log").exists()


def test_a_runner_ignoring_sigterm_is_killed_after_the_grace_period(root, monkeypatch):
    _, _, st = run_inline(root, make_plan({"mode": "ignore_term", "seconds": 30}), monkeypatch, grace=1, stop_after=1.2)
    assert st["state"] == "stopped" and states(st) == ["cancelled"]
    run = json.loads((root / "outputs/runs" / st["trees"][0]["run_id"] / "status.json").read_text())
    assert run["status"] == "stopped" and run["forced"] is True  # recorded by the coordinator: a SIGKILLed runner cannot


def test_repeated_stop_requests_are_harmless(root):
    bdir = batch.prepare(root, make_plan("ok"))
    st = batch.read_status(bdir)
    batch.write_status(bdir, {**st, "state": "running"})
    lock = batch.try_lock(bdir / "coordinator.lock")  # stands in for a live coordinator
    try:
        assert batch.request_stop(root, "b1") == "stop requested"
        assert batch.request_stop(root, "b1") == "stop requested"
        assert batch.info(bdir).state == "stopping"
    finally:
        batch.unlock(lock)
    assert batch.request_stop(root, "b1").startswith("not requested")  # coordinator gone: lost, nothing to stop


# --- detachment, locks, lost -------------------------------------------------------------------------------------------
def wait_for(cond, seconds=15):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.1)
    return False


def test_detached_coordinator_runs_the_batch_and_one_batch_at_a_time(root):
    runner = root / "fake_runner.py"
    bdir, child = batch.launch(root, make_plan({"mode": "ok", "seconds": 1.5}, "ok"), runner)
    try:
        st = batch.read_status(bdir)
        assert st["state"] == "running" and st["coordinator"]["pid"] == child.pid
        assert batch.info(bdir).state == "running" and batch.holder(root) == "Batch one"
        with pytest.raises(batch.BatchError, match="Another batch is running"):
            batch.launch(root, make_plan("ok", batch_id="b2"), runner)
        assert not (root / "outputs/batches/b2").exists()
        assert wait_for(lambda: child.poll() is not None)
    finally:
        child.kill() if child.poll() is None else None
    assert child.returncode == 0
    st = batch.read_status(bdir)
    assert st["state"] == "completed" and states(st) == ["succeeded"] * 2
    assert batch.holder(root) is None and not batch.alive(bdir)


def test_dead_coordinator_is_lost_and_children_never_held_its_lock(root):
    bdir, child = batch.launch(root, make_plan("ok", {"mode": "ok", "seconds": 60}), root / "fake_runner.py")
    assert wait_for(lambda: (batch.read_status(bdir)["trees"][0]["state"] == "running"))
    runner_pid = batch.read_status(bdir)["trees"][0]["pid"]
    try:
        os.kill(child.pid, signal.SIGKILL)
        child.wait()
        assert procinfo.identity(runner_pid) is not None  # the runner outlived the coordinator
        assert not batch.alive(bdir) and batch.holder(root) is None  # so it holds neither lock
        found = batch.info(bdir)
        assert found.state == "lost" and found.status["trees"][0]["state"] == "interrupted"
        assert found.status["trees"][1]["state"] == "pending"  # nothing relaunched or decided
        assert batch.info(bdir).state == "lost"  # stable
        assert batch.read_status(bdir)["state"] == "lost"
    finally:
        os.killpg(runner_pid, signal.SIGKILL)


def test_a_young_preparing_batch_is_not_lost_an_old_one_is(root):
    bdir = batch.prepare(root, make_plan("ok"))
    assert batch.info(bdir).state == "preparing"
    st = batch.read_status(bdir)
    batch.write_atomic(bdir / "status.json", {**st, "updated": "2020-01-01T00:00:00"})
    assert batch.info(bdir).state == "lost"


def test_start_failure_is_reported_and_recorded(root, monkeypatch):
    monkeypatch.setattr(batch.sys, "executable", str(root / "no-such-python"))
    with pytest.raises(batch.BatchError, match="did not start"):
        batch.launch(root, make_plan("ok"), root / "fake_runner.py")
    st = batch.read_status(root / "outputs/batches/b1")
    assert st["state"] == "failed" and states(st) == ["cancelled"]
    assert batch.holder(root) is None and not batch.alive(root / "outputs/batches/b1")


def test_scan_lists_batches_newest_first_and_active_one(root, monkeypatch):
    run_inline(root, make_plan("ok", batch_id="b1"), monkeypatch)
    monkeypatch.undo()
    found = batch.scan(root)
    assert [i.id for i in found] == ["b1"] and found[0].state == "completed" and found[0].count("succeeded") == 1
    assert batch.active(root) is None
    (root / "outputs/batches/b9").mkdir()
    assert batch.scan(root)[0].state == "unreadable" or batch.scan(root)[1].state == "unreadable"


def test_coordinator_imports_without_textual():
    code = "import sys; sys.modules['textual'] = None; import xmr4el_tui.coordinator"
    assert subprocess.run([sys.executable, "-c", code], cwd=REPO).returncode == 0

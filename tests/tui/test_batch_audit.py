"""Step 6 process regressions: isolated files and real synthetic children, no training."""
import os
import signal
import subprocess
import sys
import threading
import time

import pytest

import test_batch
from test_batch import make_plan, run_inline, states, wait_for
from xmr4el_tui import batch, coordinator, procinfo, process

root = test_batch.root


def test_status_remains_parseable_during_concurrent_reads(root):
    bdir = batch.prepare(root, make_plan("ok"))
    status = batch.read_status(bdir)
    done, failures, reads = threading.Event(), [], []

    def read():
        while not done.is_set():
            value = batch.read_status(bdir)
            reads.append(value is not None)
            if value is None:
                failures.append("partial or invalid status")

    reader = threading.Thread(target=read)
    reader.start()
    try:
        for n in range(100):
            batch.write_status(bdir, {**status, "name": f"Batch {n}"})
    finally:
        done.set()
        reader.join(timeout=5)
    assert not reader.is_alive() and reads and not failures


@pytest.mark.parametrize("modes,expected", [
    (("fail", "ok", "ok"), ["failed", "succeeded", "succeeded"]),
    (("ok", "ok", "fail"), ["succeeded", "succeeded", "failed"]),
])
def test_failure_at_either_end_continues(root, monkeypatch, modes, expected):
    _, _, st = run_inline(root, make_plan(*modes), monkeypatch)
    assert states(st) == expected and st["state"] == "completed_with_failures"


def test_fast_completed_batch_is_a_successful_launch(root, monkeypatch):
    bdir = batch.prepare(root, make_plan({"seconds": 0}))
    read = batch.read_status

    def delayed(path):
        time.sleep(0.7)  # coordinator finishes before the launcher's first status read
        return read(path)

    monkeypatch.setattr(batch, "read_status", delayed)
    child = batch.start(root, bdir, root / "fake_runner.py")
    assert child.wait(timeout=5) == 0
    assert read(bdir)["state"] == "completed"


def test_completed_batch_cannot_be_executed_again(root, monkeypatch):
    _, bdir, st = run_inline(root, make_plan("ok"), monkeypatch)
    before = (root / "order.log").read_text()
    time.sleep(1.1)  # a new run-directory timestamp must not permit re-execution
    assert coordinator.main(["c", str(bdir.relative_to(root))]) != 0
    assert batch.read_status(bdir) == st
    assert (root / "order.log").read_text() == before


def test_stop_arriving_during_hashing_prevents_the_next_spawn(root, monkeypatch):
    check = coordinator.Coordinator.check_inputs
    spawn, calls = coordinator.process.spawn, []

    def tracked(*args, **kwargs):
        calls.append(args)
        return spawn(*args, **kwargs)

    def stop_during_check(self):
        check(self)
        (self.dir / "stop").touch()

    monkeypatch.setattr(coordinator.Coordinator, "check_inputs", stop_during_check)
    monkeypatch.setattr(coordinator.process, "spawn", tracked)
    _, _, st = run_inline(root, make_plan("ok", "ok"), monkeypatch)
    assert st["state"] == "stopped" and states(st) == ["cancelled", "cancelled"]
    assert not (root / "order.log").exists()
    assert not calls


def test_stop_cleans_workers_even_when_the_runner_exits_first(root, monkeypatch):
    runner = root / "fake_runner.py"
    runner.write_text('''
import os, pathlib, signal, time
pid = os.fork()
if pid == 0:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    pathlib.Path("worker.pid").write_text(str(os.getpid()))
    time.sleep(30)
else:
    signal.signal(signal.SIGTERM, lambda *_: os._exit(143))
    time.sleep(30)
''')
    try:
        _, _, st = run_inline(root, make_plan("ok", "ok"), monkeypatch, grace=0.3, stop_after=1)
        pid = int((root / "worker.pid").read_text())
        assert wait_for(lambda: procinfo.identity(pid) is None, 3), "worker survived Stop Batch"
        assert st["state"] == "stopped" and states(st) == ["cancelled", "cancelled"]
    finally:
        if (root / "worker.pid").exists():
            pid = int((root / "worker.pid").read_text())
            if procinfo.identity(pid) is not None:
                os.kill(pid, signal.SIGKILL)


def test_no_artifacts_is_not_success(root, monkeypatch):
    (root / "fake_runner.py").write_text("pass\n")
    _, _, st = run_inline(root, make_plan("ok", "ok"), monkeypatch)
    assert st["state"] == "completed_with_failures" and states(st) == ["failed", "failed"]


def test_non_utf8_runner_output_does_not_abort_the_queue(root, monkeypatch):
    (root / "fake_runner.py").write_text("import sys; sys.stdout.buffer.write(b'\\xff\\n'); sys.exit(1)\n")
    _, _, st = run_inline(root, make_plan("ok", "ok"), monkeypatch)
    assert st["state"] == "completed_with_failures" and states(st) == ["failed", "failed"]


def test_completed_status_without_outputs_is_not_success(root, monkeypatch):
    (root / "fake_runner.py").write_text('''
import argparse, json, pathlib
p = argparse.ArgumentParser()
p.add_argument("-spec"); p.add_argument("-run_dir"); p.add_argument("-model_config_sha256")
a = p.parse_args()
run = pathlib.Path(a.run_dir)
run.mkdir(parents=True)
(run / "status.json").write_text(json.dumps({"status": "completed"}))
''')
    _, _, st = run_inline(root, make_plan("ok"), monkeypatch)
    assert states(st) == ["failed"]
    assert "output" in st["trees"][0]["error"]


def test_change_between_coordinator_check_and_runner_read_blocks_batch(root, monkeypatch):
    _, _, st = run_inline(root, make_plan("mutate_before_read", "ok"), monkeypatch)
    assert st["state"] == "blocked" and states(st) == ["failed", "cancelled"]
    assert "approved batch snapshot" in st["error"]


def test_status_write_failure_stops_owned_child_before_releasing_lock(root, monkeypatch):
    write, spawn, children = batch.write_status, process.spawn, []

    def broken(path, status):
        if status.get("current") and status["trees"][0].get("pid"):
            raise OSError("synthetic full disk")
        return write(path, status)

    def tracked(*a, **kw):
        child = spawn(*a, **kw)
        children.append(child)
        return child

    monkeypatch.setattr(batch, "write_status", broken)
    monkeypatch.setattr(process, "spawn", tracked)
    monkeypatch.setattr(coordinator, "GRACE", 0.2)
    run_inline(root, make_plan({"seconds": 30}, "ok"), monkeypatch)
    assert len(children) == 1 and children[0].poll() is not None
    assert batch.holder(root) is None
    assert batch.read_status(root / "outputs/batches/b1")["state"] == "failed"


@pytest.mark.parametrize("trees", [[None], [{}], [{"id": "t1", "state": "imaginary"}], "bad"])
def test_malformed_batch_status_is_unreadable_not_a_scan_crash(root, trees):
    bdir = batch.prepare(root, make_plan("ok"))
    st = batch.read_status(bdir)
    batch.write_status(bdir, {**st, "state": "running", "trees": trees})
    info = batch.scan(root)[0]
    assert info.state == "unreadable"
    assert not info.active and info.count("succeeded") == 0 and info.current is None


def test_unusable_batch_lock_is_reported_without_crashing(root):
    bdir = batch.prepare(root, make_plan("ok"))
    (bdir / "coordinator.lock").mkdir()
    assert batch.info(bdir).state == "unreadable"
    assert batch.request_stop(root, "b1").startswith("not requested")


def test_unusable_global_lock_is_a_launch_error(root):
    (root / batch.GLOBAL_LOCK).mkdir(parents=True)
    with pytest.raises(batch.BatchError, match="Cannot launch"):
        batch.launch(root, make_plan("ok"))


def test_production_launch_does_not_inherit_test_runner(root, monkeypatch):
    (root / "scripts").mkdir()
    (root / "scripts/run_experiment.py").write_text((root / "fake_runner.py").read_text())
    monkeypatch.setenv("XMR4EL_TEST_RUNNER", str(root / "nonexistent.py"))
    bdir, child = batch.launch(root, make_plan("ok"))
    try:
        child.wait(timeout=10)
        assert batch.read_status(bdir)["state"] == "completed"
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)


def test_launch_record_failure_reaps_the_spawned_child(root, monkeypatch):
    children = []
    popen = subprocess.Popen

    def capture(*a, **kw):
        child = popen(*a, **kw)
        if "-spec" in a[0]:
            children.append(child)
        return child

    monkeypatch.setattr(process.subprocess, "Popen", capture)
    console = root / process.CONSOLE_DIR
    console.mkdir(parents=True)
    (console / "test.launch.json").mkdir()  # cannot write ownership evidence after Popen
    (root / "sleep.py").write_text("import time; time.sleep(30)\n")
    try:
        with pytest.raises(OSError):
            process.spawn(root, {}, "test", runner=root / "sleep.py")
        assert children and children[0].poll() is not None, "spawn leaked a runner on an ownership-write failure"
    finally:
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=5)


def test_concurrent_launches_execute_only_one_batch(root):
    script = '''
import pathlib, sys, time
from xmr4el_tui import batch
from tests.tui.test_batch import make_plan
root = pathlib.Path(sys.argv[1])
while not (root / "go").exists():
    time.sleep(.01)
try:
    _, child = batch.launch(root, make_plan({"seconds": 2}, batch_id=sys.argv[2]), root / "fake_runner.py")
    child.wait(timeout=10)
except batch.BatchError:
    sys.exit(2)
'''
    # Use two independent launcher processes; the actual flock must arbitrate them.
    children = [subprocess.Popen([sys.executable, "-c", script, str(root), bid]) for bid in ("b1", "b2")]
    try:
        (root / "go").touch()
        codes = [c.wait(timeout=15) for c in children]
        assert sorted(codes) == [0, 2]
        assert len([l for l in (root / "order.log").read_text().splitlines() if l.startswith("start")]) == 1
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=5)


def test_coordinator_outlives_the_launcher_process_and_its_pipes(root):
    script = '''
import pathlib, sys
from xmr4el_tui import batch
from tests.tui.test_batch import make_plan
root = pathlib.Path(sys.argv[1])
batch.launch(root, make_plan({"seconds": 2}, {"seconds": 2}, {"seconds": 2}), root / "fake_runner.py")
print("launcher exiting")
'''
    try:
        result = subprocess.run([sys.executable, "-c", script, str(root)], capture_output=True,
                                text=True, timeout=5)
        assert result.returncode == 0 and result.stdout.strip() == "launcher exiting", result.stderr
        bdir = root / "outputs/batches/b1"
        assert batch.info(bdir).active
        assert wait_for(lambda: batch.info(bdir).state == "completed", 15)
        assert states(batch.read_status(bdir)) == ["succeeded"] * 3
    finally:
        for info in batch.scan(root):
            if info.active:
                batch.request_stop(root, info.id)
        assert wait_for(lambda: batch.holder(root) is None, 15)

"""process: -check responses (fake runners: malformed output and inconsistent exit codes fail), spawn in a new
session with the spec on stdin, live state of recorded-running runs, stop re-verifies identity before signalling."""
import os
import subprocess
import sys
import time
from pathlib import Path

from fake_runs import make_run
from xmr4el_tui import procinfo, process, runs
from xmr4el_tui.runs import state_word

OK = {"ok": True, "error": None, "model_config_sha256": "a" * 64, "tree_id": None,
      "search": {"dev": {"beam_size": 10}}}


def fake_runner(root, body):
    script = Path(root, process.RUNNER)
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text("import json, sys\n" + body)


def test_check(tmp_path):
    spec = {"name": "x", "evals": []}
    fake_runner(tmp_path, f"spec = json.load(sys.stdin)\nprint(json.dumps({{**{OK!r}, 'tree_id': spec['name']}}))\n")
    got = process.check(tmp_path, spec, None, revision=7)
    assert got == process.CheckResult(7, True, None, "a" * 64, "x", {"dev": {"beam_size": 10}})
    for body, msg in [("print('not json')", "no valid response"),
                      (f"print(json.dumps({OK!r})); sys.exit(1)", "exit 1"),  # ok with a failing exit code
                      ("print(json.dumps({'ok': True})); print('x', file=sys.stderr)", "exit 0): x"),
                      ("print(json.dumps({'ok': False, 'error': 'missing input files', 'model_config_sha256': None,"
                       " 'tree_id': None, 'search': {}})); sys.exit(1)", "missing input files"),
                      ("raise SystemExit(2)", "exit 2")]:
        fake_runner(tmp_path, body)
        got = process.check(tmp_path, spec, "tree", revision=1)
        assert not got.ok and msg in got.error and got.search == {}, (body, got)


def test_spawn_and_rejection(tmp_path):
    fake_runner(tmp_path, "import os\nspec = json.load(sys.stdin)\n"
                          "print(json.dumps({'error': 'tree_changed', 'expected': 'a', 'actual': 'b'}))\n"
                          "print(spec['name'], sys.argv[1:], os.getpgid(0) == os.getpid() == os.getsid(0))\n"
                          "sys.exit(3)\n")
    child = process.spawn(tmp_path, {"name": "syn"}, "2026-10-23_15-00-00_syn", xmodel_path="t", tree_id="a")
    assert child.wait(timeout=30) == 3
    log = (tmp_path / process.CONSOLE_DIR / "2026-10-23_15-00-00_syn.log").read_text()
    assert "syn ['-spec', '-', '-run_dir', 'outputs/runs/2026-10-23_15-00-00_syn', '-tree_id', 'a', " \
           "'-xmodel_path', 't'] True" in log
    assert process.rejection(tmp_path, "2026-10-23_15-00-00_syn")["error"] == "tree_changed"
    assert process.rejection(tmp_path, "missing") is None
    assert not (tmp_path / "outputs/runs/2026-10-23_15-00-00_syn").exists()
    assert [r.id for r in runs.scan(tmp_path)] == [], "the console dir is not a run"


def sleeper(new_session=True):
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=new_session)
    deadline = time.monotonic() + 10
    while procinfo.identity(child.pid) is None or procinfo.identity(child.pid)["start"] is None:
        if time.monotonic() > deadline:  # inspection blocked (sandbox): do not leave the sleeper behind
            child.kill()
            child.wait()
            raise AssertionError("child identity not readable")
        time.sleep(0.05)
    return child


def recorded_running(root, name, child, **override):
    live = procinfo.identity(child.pid)
    identity = {"pid": child.pid, "host": "h", **live, "runtime": procinfo.runtime(),
                "isolated_group": live["pgid"] == live["sid"] == child.pid, **override}
    path = make_run(root, name, status="running", tree_id=None, metrics={"dev": None}, identity=identity)
    return runs.read_run(path, root)


def test_derived_state_and_stop(tmp_path):
    here = procinfo.runtime()
    child = sleeper()
    try:
        run = recorded_running(tmp_path, "a", child)
        live = process.derived_state(run, here)
        assert live == process.Live("RUNNING", stoppable=True)
        reused = recorded_running(tmp_path, "b", child, start="Thu Jan  1 00:00:00 2026")
        assert process.derived_state(reused, here).state == "LOST"
        assert process.stop(reused).startswith("not signalled") and child.poll() is None, "reused pid not signalled"
        other = recorded_running(tmp_path, "c", child, runtime="elsewhere")
        assert process.derived_state(other, here) == process.Live("UNVERIFIED", "another machine or container",
                                                                    other_runtime=True)
        shared = recorded_running(tmp_path, "d", child, isolated_group=False)
        assert process.derived_state(shared, here) == process.Live("RUNNING", "process group is shared")
        assert process.stop(shared) == "not signalled: process group is shared" and child.poll() is None
        assert process.stop(run) == f"SIGTERM sent to pid {child.pid}"
        assert child.wait(timeout=10) == -15
        gone = replace_live(run, here)
        assert state_word(gone) == "LOST"
        assert process.stop(run).startswith("not signalled")
    finally:
        child.kill()
        child.wait()
    old = runs.read_run(make_run(tmp_path, "e", status="running", tree_id=None, metrics={"dev": None}), tmp_path)
    assert process.derived_state(old, here).state == "UNVERIFIED" and state_word(old) == "UNVERIFIED"


def test_shared_group_not_stoppable(tmp_path):
    """A child in the launcher's process group (an ordinary CLI launch) is RUNNING but never signalled."""
    child = sleeper(new_session=False)
    try:
        run = recorded_running(tmp_path, "cli", child)
        assert run.identity["isolated_group"] is False and run.identity["pgid"] == os.getpgid(0)
        assert process.derived_state(run, procinfo.runtime()).state == "RUNNING"
        assert process.stop(run, force=True).startswith("not signalled") and child.poll() is None
    finally:
        child.kill()
        child.wait()


def replace_live(run, here):
    return runs.replace(run, live=process.derived_state(run, here))

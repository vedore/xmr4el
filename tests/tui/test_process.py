"""process: -check responses (fake runners: malformed output and inconsistent exit codes fail), spawn in a new
session with the spec on stdin, live state of recorded-running runs, stop re-verifies identity before signalling."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from fake_runs import make_run
from xmr4el_tui import procinfo, process, runs
from xmr4el_tui.runs import state_word

OK = {"ok": True, "error": None, "model_config_sha256": "a" * 64, "tree_id": None,
      "search": {"dev": {"beam_size": 10}}}


def fake_runner(root, body):
    script = Path(root, process.RUNNER)
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text("import json, sys\n" + body)


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


def sleeper(new_session=True, ignore_term=False):
    code = "import signal, time\n" + ("signal.signal(signal.SIGTERM, signal.SIG_IGN)\n" if ignore_term else "") + \
        "print('ready', flush=True)\ntime.sleep(60)"
    child = subprocess.Popen([sys.executable, "-c", code], start_new_session=new_session, stdout=subprocess.PIPE)
    child.stdout.readline()  # its SIGTERM handler is in place
    deadline = time.monotonic() + 10
    while procinfo.identity(child.pid) is None or procinfo.identity(child.pid)["start"] is None:
        if time.monotonic() > deadline:  # inspection blocked (sandbox): do not leave the sleeper behind
            child.kill()
            child.wait()
            raise AssertionError("child identity not readable")
        time.sleep(0.05)
    return child


def recorded_running(root, name, child, launched=True, **override):
    """`launched`: the launch record spawn writes (TUI ownership) for the child's real identity."""
    live = procinfo.identity(child.pid)
    identity = {"pid": child.pid, "host": "h", **live, "runtime": procinfo.runtime(),
                "isolated_group": live["pgid"] == live["sid"] == child.pid, **override}
    path = make_run(root, name, status="running", tree_id=None, metrics={"dev": None}, identity=identity)
    if launched:
        console = Path(root, process.CONSOLE_DIR)
        console.mkdir(parents=True, exist_ok=True)
        (console / f"{path.name}.launch.json").write_text(json.dumps(
            {"run_id": path.name, "run_dir": str(path.resolve()), "pid": child.pid, "start": live["start"],
             "uid": live["uid"], "runtime": procinfo.runtime()}))
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
        understated = recorded_running(tmp_path, "d", child, isolated_group=False)
        assert process.derived_state(understated, here).stoppable, "the live group decides, not the recorded flag"
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


def test_external_isolated_run_is_monitor_only(tmp_path, monkeypatch):
    """An isolated process the TUI did not launch (no launch record, one for another run / directory / process, or
    a status.json replaced after it was read) is RUNNING, never signalled."""
    child = sleeper()
    sent = []
    monkeypatch.setattr(os, "killpg", lambda *a: sent.append(a))
    try:
        here = procinfo.runtime()
        external = recorded_running(tmp_path, "ext", child, launched=False)
        assert process.derived_state(external, here) == process.Live("RUNNING",
                                                                     "not launched by the TUI (monitor only)")
        assert process.stop(external).startswith("not signalled")
        run = recorded_running(tmp_path, "own", child)
        launch = tmp_path / process.CONSOLE_DIR / f"{run.id}.launch.json"
        good = launch.read_text()
        for field, value in (("run_id", "other"), ("run_dir", "/elsewhere"), ("pid", 1), ("start", "x")):
            launch.write_text(json.dumps({**json.loads(good), field: value}))
            assert not process.derived_state(run, here).stoppable, field
        launch.write_text(good)
        assert process.derived_state(run, here).stoppable
        status = run.path / "status.json"
        status.write_text(json.dumps({**json.loads(status.read_text()), "pid": 1}))
        assert process.stop(run) == "not signalled: status.json changed since it was read"
        assert sent == [] and child.poll() is None
    finally:
        child.kill()
        child.wait()


def test_spawn_writes_launch_record(tmp_path):
    fake_runner(tmp_path, "import time\njson.load(sys.stdin)\ntime.sleep(30)\n")
    child = process.spawn(tmp_path, {"name": "own"}, "2026-10-23_15-00-00_own")
    try:
        launch = json.loads((tmp_path / process.CONSOLE_DIR / "2026-10-23_15-00-00_own.launch.json").read_text())
        live = procinfo.identity(child.pid)
        assert launch == {"run_id": "2026-10-23_15-00-00_own",
                          "run_dir": str((tmp_path / "outputs/runs/2026-10-23_15-00-00_own").resolve()),
                          "pid": child.pid, "start": live["start"], "uid": live["uid"],
                          "runtime": procinfo.runtime()}
    finally:
        child.kill()
        child.wait()


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


def test_false_isolation_claim_is_never_signalled(tmp_path, monkeypatch):
    """Recorded and live identity agree, but pid 101 does not lead group/session 7: a status.json claiming
    isolated_group must not get group 7 signalled."""
    ident = {"uid": 1, "start": "s", "pgid": 7, "sid": 7}
    monkeypatch.setattr(procinfo, "runtime", lambda: "here")
    monkeypatch.setattr(procinfo, "identity", lambda pid: dict(ident))
    sent = []
    monkeypatch.setattr(os, "killpg", lambda *a: sent.append(a))
    for override in ({}, {"pgid": 8}, {"start": None}):
        run = runs.read_run(make_run(tmp_path, f"claim{len(override)}{list(override)}", status="running",
                                     tree_id=None, metrics={"dev": None},
                                     identity={"pid": 101, "runtime": "here", **ident, "isolated_group": True,
                                               **override}), tmp_path)
        assert not process.derived_state(run, "here").stoppable
        assert process.stop(run).startswith("not signalled") and process.stop(run, force=True).startswith(
            "not signalled")
    assert sent == []
    assert all(json.loads(p.read_text())["status"] == "running" for p in tmp_path.glob("outputs/runs/*/status.json")), \
        "nothing recorded for a process that was never signalled"


def test_term_ignored_then_force_stop(tmp_path):
    child = sleeper(ignore_term=True)
    try:
        run = recorded_running(tmp_path, "stubborn", child)
        assert process.stop(run) == f"SIGTERM sent to pid {child.pid}"
        with pytest.raises(subprocess.TimeoutExpired):
            child.wait(timeout=1)
        assert process.stop(run, force=True) == f"SIGKILL sent to pid {child.pid}; recorded stopped (forced)"
        assert child.wait(timeout=10) == -9
    finally:
        child.kill()
        child.wait()


def test_stop_refused_once_final_status_is_written(tmp_path):
    """A run that wrote its final status while the stop was being asked for is not signalled, alive or not."""
    child = sleeper()
    try:
        run = recorded_running(tmp_path, "done", child)
        status = run.path / "status.json"
        status.write_text(status.read_text().replace('"running"', '"completed"'))
        assert process.stop(run) == "not signalled: run already completed" and child.poll() is None
    finally:
        child.kill()
        child.wait()


def test_stop_reaches_the_whole_process_group(tmp_path):
    """The run's own children (same group) stop with it; a process outside the group is untouched."""
    code = ("import subprocess, sys, time\n"
            "kid = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
            "print(kid.pid, flush=True)\ntime.sleep(60)")
    leader = subprocess.Popen([sys.executable, "-c", code], start_new_session=True, stdout=subprocess.PIPE, text=True)
    kid = int(leader.stdout.readline())
    outsider = sleeper()
    try:
        while procinfo.identity(leader.pid)["start"] is None:
            time.sleep(0.05)
        run = recorded_running(tmp_path, "tree", leader)
        assert process.stop(run) == f"SIGTERM sent to pid {leader.pid}"
        assert leader.wait(timeout=10) == -15
        deadline = time.monotonic() + 10
        while procinfo.identity(kid) is not None and time.monotonic() < deadline:
            time.sleep(0.05)
        assert procinfo.identity(kid) is None or procinfo.identity(kid)["pgid"] != leader.pid
        assert outsider.poll() is None
    finally:
        for p in (leader, outsider):
            p.kill()
            p.wait()
        try:
            os.kill(kid, 9)
        except ProcessLookupError:
            pass


def test_forced_stop_is_recorded_external_kill_is_lost(tmp_path):
    """A verified force stop records stopped (SIGKILL, forced, identity kept) once the process is gone, even before
    it is reaped, so it never reads as LOST; the same SIGKILL from outside leaves running -> LOST."""
    here = procinfo.runtime()
    child = sleeper()
    try:
        run = recorded_running(tmp_path, "forced", child)
        assert process.stop(run, force=True) == f"SIGKILL sent to pid {child.pid}; recorded stopped (forced)"
        assert child.poll() is None or child.returncode == -9, "not reaped by stop: a zombie counted as gone"
        status = json.loads((run.path / "status.json").read_text())
        assert (status["status"], status["error"], status["forced"], status["pid"]) == ("stopped", "SIGKILL", True,
                                                                                         child.pid)
        again = runs.read_run(run.path, tmp_path)
        assert state_word(replace_live(again, here)) == "STOPPED" and "forced" not in (again.identity or {})
        assert process.stop(again, force=True) == "not signalled: run already stopped", "repeat: nothing sent"
    finally:
        child.kill()
        child.wait()
    child = sleeper()
    try:
        run = recorded_running(tmp_path, "oom", child)
        child.kill()  # external SIGKILL (OOM killer, kill -9): nobody records anything
        child.wait()
        assert json.loads((run.path / "status.json").read_text())["status"] == "running"
        assert state_word(replace_live(run, here)) == "LOST"
    finally:
        child.kill()
        child.wait()


def test_forced_stop_keeps_the_runners_final_status(tmp_path, monkeypatch):
    """The runner wrote stopped (handling an earlier SIGTERM) just before the SIGKILL: its record is kept."""
    child = sleeper()
    try:
        run = recorded_running(tmp_path, "raced", child)
        status = run.path / "status.json"
        kill = os.killpg

        def runner_finishes_first(pgid, sig):
            status.write_text(status.read_text().replace('"running"', '"stopped"'))
            kill(pgid, sig)
        monkeypatch.setattr(os, "killpg", runner_finishes_first)
        assert process.stop(run, force=True).endswith("; run recorded stopped")
        assert "SIGKILL" not in status.read_text() and "forced" not in status.read_text()
    finally:
        child.kill()
        child.wait()


def test_forced_stop_not_recorded_while_the_process_lives(tmp_path, monkeypatch):
    """No proof the process is gone (here: the signal never arrives) -> nothing written, status stays running."""
    child = sleeper()
    try:
        run = recorded_running(tmp_path, "stuck", child)
        monkeypatch.setattr(process, "FORCE_WAIT", 0.3)
        monkeypatch.setattr(os, "killpg", lambda *a: None)
        assert process.stop(run, force=True).endswith("; still exiting, stop not recorded")
        assert json.loads((run.path / "status.json").read_text())["status"] == "running" and child.poll() is None
    finally:
        child.kill()
        child.wait()

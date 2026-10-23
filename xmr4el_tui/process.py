"""Process side of the TUI (tui.md §9, §10): -check in a child, spawn a run, liveness of recorded-running runs, stop.
Stdlib only. A run's state always comes from its files plus a live identity check, never from a Popen handle.
"""
import json
import os
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from xmr4el_tui import procinfo
from xmr4el_tui.runs import RUNS_DIR

RUNNER = Path("scripts", "run_experiment.py")
CONSOLE_DIR = RUNS_DIR / ".console"
CHECK_KEYS = {"ok", "error", "model_config_sha256", "tree_id", "search"}
BINDING_ERRORS = ("model_config_changed", "tree_changed")


@dataclass(frozen=True)
class CheckResult:
    revision: int             # the draft revision this check validated (stale results are ignored)
    ok: bool
    error: str | None
    model_config_sha256: str | None
    tree_id: str | None
    search: dict              # split -> effective search


@dataclass(frozen=True)
class Live:
    state: str                # RUNNING | LOST | UNVERIFIED
    reason: str = ""          # why LOST / UNVERIFIED
    other_runtime: bool = False
    stoppable: bool = False   # RUNNING, identity verified, isolated process group


def runner_args(xmodel_path, *extra):
    args = [sys.executable, str(RUNNER), "-spec", "-", *extra]
    return args + (["-xmodel_path", xmodel_path] if xmodel_path else [])


def check(root, spec, xmodel_path, revision, timeout=300):
    """Run the runner's -check on `spec`. A startup failure, malformed response or an exit code that disagrees
    with `ok` is a failed check, never a success."""
    fail = lambda error: CheckResult(revision, False, error, None, None, {})
    try:
        out = subprocess.run(runner_args(xmodel_path, "-check"), cwd=root, input=json.dumps(spec),
                             capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as err:
        return fail(f"check did not run: {err}")
    try:
        response = json.loads(out.stdout)
    except ValueError:
        response = None
    if not isinstance(response, dict) or set(response) != CHECK_KEYS or out.returncode != (0 if response["ok"]
                                                                                         else 1):
        tail = (out.stderr.strip().splitlines() or [""])[-1]
        return fail(f"check failed (exit {out.returncode}): {tail or 'no valid response'}")
    if response["ok"] is not True or not isinstance(response["search"], dict):
        return fail(response["error"] or "check failed")
    return CheckResult(revision, True, None, response["model_config_sha256"], response["tree_id"],
                       response["search"])


def spawn(root, spec, run_id, xmodel_path=None, config_sha=None, tree_id=None):
    """Start the runner for `spec` in outputs/runs/<run_id> (new session: its own process group), stdout/stderr to
    outputs/runs/.console/<run_id>.log. Returns the Popen (only to reap it)."""
    extra = ["-run_dir", str(RUNS_DIR / run_id)]
    extra += ["-model_config_sha256", config_sha] if config_sha else []
    extra += ["-tree_id", tree_id] if tree_id else []
    console = Path(root, CONSOLE_DIR)
    console.mkdir(parents=True, exist_ok=True)
    with open(console / f"{run_id}.log", "w") as log:
        child = subprocess.Popen(runner_args(xmodel_path, *extra), cwd=root, stdin=subprocess.PIPE, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True, text=True)
    try:
        child.stdin.write(json.dumps(spec))
        child.stdin.close()
    except BrokenPipeError:  # exited before reading: the console log says why
        pass
    return child


def rejection(root, run_id):
    """The binding-rejection JSON line (exit 3) in the run's console log, or None."""
    try:
        lines = Path(root, CONSOLE_DIR, f"{run_id}.log").read_text().splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if isinstance(obj, dict) and obj.get("error") in BINDING_ERRORS:
            return obj
    return None


def derived_state(run, here):
    """Live state of a run recorded running, seen from runtime `here` (procinfo.runtime())."""
    rec = run.identity or {}
    if not isinstance(rec.get("pid"), int):
        return Live("UNVERIFIED", "no process identity recorded (written before the runner change)")
    if not here or rec.get("runtime") != here:
        return Live("UNVERIFIED", "another machine or container", other_runtime=True)
    if rec.get("start") is None or rec.get("uid") is None:
        return Live("UNVERIFIED", "incomplete recorded identity")
    live = procinfo.identity(rec["pid"])
    if live is None:
        return Live("LOST", f"process {rec['pid']} is gone")
    if live["start"] is None or live["uid"] is None:
        return Live("UNVERIFIED", "inspection denied")
    if (live["start"], live["uid"]) != (rec["start"], rec["uid"]):
        return Live("LOST", f"process {rec['pid']} is gone (pid reused)")
    stoppable = procinfo.same(rec, live) and rec.get("isolated_group") is True
    return Live("RUNNING", "" if stoppable else "process group is shared", stoppable=stoppable)


def stop(run, force=False):
    """SIGTERM (force: SIGKILL) to the run's process group after re-verifying its identity, runtime and isolated
    group. Returns what happened; on any mismatch nothing is sent."""
    live = derived_state(run, procinfo.runtime())
    if not live.stoppable:
        return f"not signalled: {live.reason or live.state}"
    sig = signal.SIGKILL if force else signal.SIGTERM
    try:
        os.killpg(run.identity["pgid"], sig)
    except ProcessLookupError:
        return "not signalled: process exited"
    except PermissionError:
        return "not signalled: permission denied"
    return f"{sig.name} sent to pid {run.identity['pid']}"

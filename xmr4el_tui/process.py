"""Process side of the TUI (tui.md §9, §10): -check in a child, spawn a run, liveness of recorded-running runs, stop.
Stdlib only. A run's state always comes from its files plus a live identity check, never from a Popen handle.
"""
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from xmr4el_tui import procinfo
from xmr4el_tui.runs import RUNS_DIR, STATUS_KEYS, read_dict

RUNNER = Path("scripts", "run_experiment.py")
CONSOLE_DIR = RUNS_DIR / ".console"
CHECK_KEYS = {"ok", "error", "model_config_sha256", "tree_id", "search"}
BINDING_ERRORS = ("model_config_changed", "tree_changed")
FINAL = ("completed", "failed", "stopped")  # recorded final states
FORCE_WAIT = 10  # seconds a forced stop waits for the killed process to be gone before recording it


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
    stoppable: bool = False   # RUNNING, identity verified, leads its own group and session, launched by a TUI


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
    outputs/runs/.console/<run_id>.log. Returns the Popen (only to reap it). An existing console log (the run id was
    used before) raises FileExistsError: nothing is started or overwritten. The child's identity goes to
    .console/<run_id>.launch.json: the ownership evidence `owned` requires before a run may be cancelled."""
    extra = ["-run_dir", str(RUNS_DIR / run_id)]
    extra += ["-model_config_sha256", config_sha] if config_sha else []
    extra += ["-tree_id", tree_id] if tree_id else []
    console = Path(root, CONSOLE_DIR)
    console.mkdir(parents=True, exist_ok=True)
    with open(console / f"{run_id}.log", "x") as log:
        child = subprocess.Popen(runner_args(xmodel_path, *extra), cwd=root, stdin=subprocess.PIPE, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True, text=True)
    ident = procinfo.identity(child.pid) or {}
    with open(console / f"{run_id}.launch.json", "x") as f:
        json.dump({"run_id": run_id, "run_dir": str(Path(root, RUNS_DIR, run_id).resolve()), "pid": child.pid,
                   "start": ident.get("start"), "uid": ident.get("uid"), "runtime": procinfo.runtime()}, f)
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


def owned(run, rec):
    """True when a TUI launched `run` as the process in identity `rec`: its launch record (written by spawn) names
    this run id and directory and the same pid, start token, UID and runtime. An isolated group alone is no
    ownership: a shell run under setsid is monitored, never cancelled."""
    launch = read_dict(run.path.parent / ".console" / f"{run.id}.launch.json") or {}
    return (launch.get("run_id") == run.id and launch.get("run_dir") == str(run.path.resolve())
            and all(launch.get(k) is not None and launch.get(k) == rec.get(k) for k in ("pid", "start", "uid",
                                                                                         "runtime")))


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
    # the recorded isolated_group flag is no proof: the live process must lead its own group and session now
    if not (procinfo.same(rec, live) and live["pgid"] == live["sid"] == rec["pid"]):
        return Live("RUNNING", "process group is shared")
    if not owned(run, rec):
        return Live("RUNNING", "not launched by the TUI (monitor only)")
    return Live("RUNNING", stoppable=True)


def stop(run, force=False):
    """Cancel `run` (tui.md §9): SIGTERM (force: SIGKILL) to its process group after re-reading status.json (a run
    that already wrote a final status, or whose recorded identity changed since `run` was read, is not signalled)
    and re-verifying its identity, runtime, TUI launch record and that it leads its group. Returns what happened; on any mismatch nothing is sent. Safe to repeat: the runner records the first
    SIGTERM as stopped and ignores later ones, and ignores one that arrives after its final status is decided.
    A SIGKILLed runner records nothing, so a forced stop records it here (`record_forced`): blocks up to
    FORCE_WAIT seconds (call it from a worker)."""
    status = read_dict(run.path / "status.json") or {}
    if status.get("status") in FINAL:
        return f"not signalled: run already {status['status']}"
    if {k: v for k, v in status.items() if k not in STATUS_KEYS} != run.identity:
        return "not signalled: status.json changed since it was read"
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
    sent = f"{sig.name} sent to pid {run.identity['pid']}"
    return f"{sent}; {record_forced(run)}" if force else sent


def record_forced(run):
    """After a verified SIGKILL: once the recorded process is gone (absent, reaped or a zombie, or its pid reused),
    a status.json still `running` becomes `stopped` with error SIGKILL and forced: true, identity kept (it cannot
    write any more, so nothing races this write). A final status the runner wrote first (it was already handling a
    SIGTERM) is kept. An unrequested SIGKILL never passes here: that run stays `running` and shows LOST."""
    rec, deadline = run.identity, time.monotonic() + FORCE_WAIT
    while (live := procinfo.identity(rec["pid"])) is not None and live["start"] == rec["start"]:
        if time.monotonic() > deadline:
            return "still exiting, stop not recorded"
        time.sleep(0.05)
    path = run.path / "status.json"
    status = read_dict(path)
    if not status or status.get("status") != "running":
        return f"run recorded {(status or {}).get('status', 'nothing')}"
    tmp = path.with_name("status.json.forced")
    tmp.write_text(json.dumps({**status, "status": "stopped", "finished": datetime.now().isoformat(timespec="seconds"),
                               "error": "SIGKILL", "forced": True}, indent=2))
    os.replace(tmp, path)
    return "recorded stopped (forced)"


class Launches:
    """Runs this TUI started: the one owner of their Popen handles. A run's state still comes from its files; a
    handle only tells what became of a launch that wrote no final status. Detached children outlive the TUI."""

    def __init__(self, root):
        self.root, self.children = root, {}  # run id -> Popen

    def start(self, spec, run_id, **kwargs):
        """spawn(); FileExistsError when the run id was used before."""
        self.children[run_id] = spawn(self.root, spec, run_id, **kwargs)

    def reap(self):
        """[(run id, what happened)] for children that exited without a final status.json, each reported once;
        finished children are reaped, running ones kept."""
        out = []
        for run_id, child in list(self.children.items()):
            code = child.poll()
            if code is None:
                continue
            del self.children[run_id]
            status = read_dict(Path(self.root, RUNS_DIR, run_id, "status.json")) or {}
            if status.get("status") in FINAL:
                continue
            rejected = rejection(self.root, run_id) if code == 3 else None
            out.append((run_id, f"not started: {rejected['error']}" if rejected else
                        f"exited {code} without a final status; see {CONSOLE_DIR / f'{run_id}.log'}"))
        return out

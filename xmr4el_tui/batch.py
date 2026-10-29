"""Batch files and launch (tui.md §5.16). Stdlib only (the coordinator imports this without the tui extra).

A batch lives in outputs/batches/<batch_id>/: `batch.json` (the immutable plan, written once), `status.json` (mutable;
only the coordinator writes it, atomically, except `mark_lost` for a dead coordinator), `stop` (empty file: the stop
request), `coordinator.log`, `coordinator.lock`, `configs/<tree_id>.json` (the model configs the runs read). Runs are
ordinary outputs/runs/ runs. outputs/batches/.active.lock allows one running batch (V1: one GPU, one batch).

Liveness of a coordinator is its flock on `coordinator.lock`: the kernel drops it when the process dies, so it cannot be
stale or reused like a PID. The locks are non-inheritable descriptors (PEP 446) and the runners are started with
close_fds, so no training child ever holds them. The coordinator's procinfo identity is recorded for display.
"""
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

from xmr4el_tui import review
from xmr4el_tui.runs import read_dict

BATCHES_DIR = Path("outputs", "batches")
GLOBAL_LOCK = BATCHES_DIR / ".active.lock"
POLICY = "continue_on_tree_failure"
ACTIVE = ("preparing", "running", "stopping")
TERMINAL = ("completed", "completed_with_failures", "stopped", "blocked", "failed", "lost")
WORDS = {"preparing": "PREPARING", "running": "RUNNING", "stopping": "STOPPING", "completed": "COMPLETED",
         "completed_with_failures": "COMPLETED WITH FAILURES", "stopped": "STOPPED", "blocked": "BLOCKED",
         "failed": "FAILED", "lost": "LOST", "unreadable": "UNREADABLE"}
TREE_WORDS = {"pending": "PENDING", "running": "RUNNING", "succeeded": "SUCCEEDED", "failed": "FAILED",
              "cancelled": "CANCELLED", "interrupted": "INTERRUPTED"}
START_TIMEOUT = 20      # seconds the launch waits for the coordinator to report running
PREPARING_GRACE = 60    # seconds a batch may stay `preparing` without a coordinator before it is called lost


class BatchError(Exception):
    """A launch that did not start a batch; the message says what to do."""


def now():
    return datetime.now().isoformat(timespec="seconds")


def directory(root, batch_id):
    return Path(root, BATCHES_DIR, batch_id)


# --- files ----------------------------------------------------------------------------------------------------------
def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_atomic(path, obj):
    """Write-then-rename: a reader sees the old or the new JSON, never half of it."""
    tmp = Path(f"{path}.tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def read_status(batch_dir):
    st = read_dict(Path(batch_dir, "status.json"))
    if not st or not isinstance(st.get("state"), str) or st["state"] not in (*ACTIVE, *TERMINAL):
        return None
    if not all(isinstance(st.get(k), str) for k in ("batch_id", "name", "created", "updated")):
        return None
    if st.get("coordinator") is not None and not isinstance(st["coordinator"], dict):
        return None
    if not isinstance(st.get("trees"), list):
        return None
    for t in st["trees"]:
        if not isinstance(t, dict) or not all(isinstance(t.get(k), str) for k in ("id", "name", "state")):
            return None
        if t["state"] not in TREE_WORDS or not all(k in t for k in ("run_id", "started", "finished", "exit_code", "error")):
            return None
        if any(t[k] is not None and not isinstance(t[k], str) for k in ("run_id", "started", "finished", "error")):
            return None
    if any(st.get(k) is not None and not isinstance(st[k], str) for k in ("created", "updated", "started", "finished", "current", "error")):
        return None
    ids = [t["id"] for t in st["trees"]]
    if len(ids) != len(set(ids)) or st.get("current") is not None and st["current"] not in ids:
        return None
    return st


def write_status(batch_dir, status):
    write_atomic(Path(batch_dir, "status.json"), {**status, "updated": now()})


def stop_requested(batch_dir):
    return Path(batch_dir, "stop").exists()


# --- locks ------------------------------------------------------------------------------------------------------------
def try_lock(path, wait=0.0):
    """Exclusive flock on `path`: the open descriptor, or None when another process holds it (after `wait` seconds).
    The descriptor is non-inheritable: runners and any other child never hold the lock."""
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    end = time.monotonic() + wait
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fd
        except BlockingIOError:
            if time.monotonic() >= end:
                os.close(fd)
                return None
            time.sleep(0.05)


def unlock(fd):
    os.close(fd)  # closing releases the flock


def alive(batch_dir):
    """True while a coordinator holds the batch's lock."""
    fd = try_lock(Path(batch_dir, "coordinator.lock"))
    if fd is None:
        return True
    unlock(fd)
    return False


def holder(root):
    """Name of the batch holding the one-active-batch lock, None when it is free."""
    path = Path(root, GLOBAL_LOCK)
    if not path.parent.is_dir():
        return None
    fd = try_lock(path)
    if fd is None:
        try:
            return path.read_text().strip() or "another batch"
        except OSError:
            return "another batch"
    unlock(fd)
    return None


# --- snapshot -----------------------------------------------------------------------------------------------------------
def git_state(root):
    def git(*a):
        try:
            return subprocess.run(["git", *a], capture_output=True, text=True, cwd=root, timeout=20).stdout
        except (OSError, subprocess.TimeoutExpired):
            return ""
    commit = git("rev-parse", "HEAD").strip()
    return {"commit": commit or None, "dirty": bool(git("status", "--porcelain").strip()) if commit else None}


def input_paths(plan_):
    d = plan_.dataset
    return sorted({p for p in (d.train, *(e.path for e in d.enabled())) if p})


def hash_inputs(root, paths):
    return {p: file_sha256(Path(root, p)) for p in paths}


def changed_inputs(root, inputs):
    """Paths of `inputs` ({path: sha256}) whose content differs now (a missing file counts)."""
    out = []
    for p, sha in inputs.items():
        try:
            same = file_sha256(Path(root, p)) == sha
        except OSError:
            same = False
        if not same:
            out.append(p)
    return out


def prepare(root, plan_):
    """Write the batch's immutable plan, model configs and first status; returns the batch directory. `plan_` is a
    review.Plan of a validated batch (ids already unique: the batch id names the directory). Slow (hashes the inputs):
    call it from a worker. Nothing is left behind on failure."""
    bdir = directory(root, plan_.id)
    try:
        hashes = hash_inputs(root, input_paths(plan_))
        bdir.mkdir(parents=True)
    except FileExistsError:
        raise BatchError(f"Batch directory {BATCHES_DIR / plan_.id} already exists.") from None
    except OSError as err:
        raise BatchError(f"Cannot prepare the batch: {err}") from err
    try:
        (bdir / "configs").mkdir()
        trees = []
        for t in plan_.trees:
            rel = BATCHES_DIR / plan_.id / "configs" / f"{t.id}.json"
            raw = json.dumps(t.config, indent=2).encode()
            Path(root, rel).write_bytes(raw)
            trees.append({"id": t.id, "name": t.name, "config_path": str(rel),
                          "config_sha256": hashlib.sha256(raw).hexdigest(),
                          "spec": review.run_spec(plan_, t, str(rel))})
        d = plan_.dataset
        created = now()
        text = json.dumps({
            "v": 1, "id": plan_.id, "name": plan_.name, "created": created, "policy": POLICY,
            "dataset": {"key": d.key, "name": d.name, "variant": d.variant, "format": d.format, "train": d.train,
                        "evals": [{"split": e.split, "path": e.path} for e in d.enabled()]},
            "inputs": hashes,
            "provenance": {**git_state(root), "python": sys.version.split()[0], "executable": sys.executable},
            "trees": trees}, indent=2)
        with open(bdir / "batch.json", "x") as f:  # the plan is written once, completely, before anything runs
            f.write(text)
        write_status(bdir, {
            "v": 1, "batch_id": plan_.id, "name": plan_.name, "state": "preparing",
            "dataset": f"{d.name} — {d.variant}" if d.variant else d.name,
            "plan_sha256": hashlib.sha256(text.encode()).hexdigest(), "coordinator": None, "current": None,
            "stop_requested": None, "created": created, "started": None, "finished": None, "error": None,
            "trees": [{"id": t["id"], "name": t["name"], "state": "pending", "run_id": None, "pid": None,
                       "exit_code": None, "error": None, "started": None, "finished": None} for t in trees]})
    except OSError as err:
        discard(bdir)
        raise BatchError(f"Cannot write the batch files: {err}") from err
    return bdir


def discard(bdir):
    """Remove a batch directory that never ran (prepare's own leftovers)."""
    for p in sorted(Path(bdir).rglob("*"), reverse=True):
        p.unlink() if not p.is_dir() else p.rmdir()
    try:
        Path(bdir).rmdir()
    except OSError:
        pass


def fail(batch_dir, message):
    """Record a batch that never got a coordinator as failed (only while no coordinator holds the lock)."""
    fd = try_lock(Path(batch_dir, "coordinator.lock"))
    if fd is None:
        return
    try:
        st = read_status(batch_dir)
        if st and st.get("state") not in TERMINAL:
            write_status(batch_dir, {**st, "state": "failed", "error": message, "finished": now(),
                                     "trees": [{**t, "state": "cancelled"} if t["state"] == "pending" else t
                                               for t in st["trees"]]})
    finally:
        unlock(fd)


def start(root, bdir, runner=None):
    """Start the detached coordinator of a prepared batch and wait until it reports running. Returns its Popen (the
    caller reaps it; it is not the coordinator's owner: it keeps running when the TUI exits). On any failure the batch
    is recorded failed and BatchError says why. `runner`: a test seam (environment of the coordinator), never
    offered to the user."""
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(Path(__file__).parents[1]),
                                                                     os.environ.get("PYTHONPATH")]))}
    if runner:
        env["XMR4EL_TEST_RUNNER"] = str(runner)
    else:
        env.pop("XMR4EL_TEST_RUNNER", None)
    try:
        with open(bdir / "coordinator.log", "a") as log:
            child = subprocess.Popen([sys.executable, "-m", "xmr4el_tui.coordinator", str(bdir.relative_to(root))],
                                     cwd=root, env=env, stdin=subprocess.DEVNULL, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
    except OSError as err:
        fail(bdir, f"coordinator did not start: {err}")
        raise BatchError(f"The coordinator did not start: {err}") from err
    deadline = time.monotonic() + START_TIMEOUT
    while True:
        st = read_status(bdir) or {}
        if st.get("started") and (st.get("coordinator") or {}).get("pid") == child.pid:
            return child  # a fast batch may already have finished before this first read
        if st.get("state") in TERMINAL:
            child.wait(timeout=5)
            raise BatchError(f"The batch did not start: {st.get('error') or st['state']}")
        if child.poll() is not None:
            tail = (Path(bdir, "coordinator.log").read_text().strip().splitlines() or ["no output"])[-1]
            fail(bdir, f"coordinator exited {child.returncode} at startup: {tail}")
            raise BatchError(f"The coordinator exited at startup ({tail}); see {BATCHES_DIR / bdir.name / 'coordinator.log'}.")
        if time.monotonic() > deadline:
            child.terminate()  # our own unreaped child: safe to signal
            try:
                child.wait(5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            fail(bdir, f"coordinator did not report running within {START_TIMEOUT} s")
            raise BatchError(f"The coordinator did not report running within {START_TIMEOUT} s.")
        time.sleep(0.1)


def launch(root, plan_, runner=None):
    """Prepare and start a batch (a worker's job). One running batch at a time: another one's lock refuses it before
    anything is written. Returns (batch directory, coordinator Popen)."""
    try:
        other = holder(root)
        if other:
            raise BatchError(f"Another batch is running ({other}). Stop it or wait for it to finish.")
        bdir = prepare(root, plan_)
        return bdir, start(root, bdir, runner)
    except OSError as err:
        raise BatchError(f"Cannot launch the batch: {err}") from err


# --- discovery ----------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Info:
    id: str
    path: Path
    name: str
    state: str                # the status.json state, or unreadable; `stopping` once a stop was asked of a live batch
    status: dict
    problem: str | None = None

    @property
    def active(self):
        return self.state in ACTIVE

    def count(self, state):
        return sum(t.get("state") == state for t in self.status.get("trees", ()))

    @property
    def current(self):
        return next((t for t in self.status.get("trees", ()) if t["id"] == self.status.get("current")), None)


def mark_lost(batch_dir):
    """A recorded-active batch whose coordinator is gone (the lock is free and held here while writing): lost. The tree
    that was running becomes `interrupted` (its run directory says what the run itself recorded); nothing is relaunched
    and no process is signalled. Returns the new status, None when a coordinator is alive after all."""
    fd = try_lock(Path(batch_dir, "coordinator.lock"))
    if fd is None:
        return None
    try:
        st = read_status(batch_dir)
        if not st or st.get("state") not in ACTIVE:
            return st
        trees = [{**t, "state": "interrupted", "finished": now(),
                  "error": "the coordinator ended while this tree was running"} if t["state"] == "running" else t
                 for t in st["trees"]]
        st = {**st, "state": "lost", "finished": now(), "trees": trees,
              "error": "The coordinator process ended unexpectedly; unfinished trees were not restarted."}
        write_status(batch_dir, st)
        return st
    finally:
        unlock(fd)


def info(batch_dir):
    """One batch as the TUI shows it: recorded state corrected by liveness (a dead coordinator makes it lost)."""
    batch_dir = Path(batch_dir)
    st = read_status(batch_dir)
    if st is None or not isinstance(st.get("trees"), list) or not isinstance(st.get("state"), str):
        return Info(batch_dir.name, batch_dir, batch_dir.name, "unreadable", {}, "status.json missing or invalid")
    state = st["state"]
    try:
        if state in ACTIVE and not alive(batch_dir):
            try:
                age = (datetime.now() - datetime.fromisoformat(st.get("updated") or st["created"])).total_seconds()
            except (ValueError, KeyError, TypeError):
                age = PREPARING_GRACE
            if state != "preparing" or age > PREPARING_GRACE:  # a young preparing batch is still being started
                st = mark_lost(batch_dir) or st
                state = st["state"]
    except OSError as err:
        return Info(batch_dir.name, batch_dir, st["name"], "unreadable", {}, f"Cannot inspect batch: {err}")
    if state == "running" and stop_requested(batch_dir):
        state = "stopping"
    return Info(batch_dir.name, batch_dir, str(st.get("name") or batch_dir.name), state, st)


def scan(root, cache=None):
    """[Info], newest first. Finished batches are read once per status.json change; active ones are checked each scan."""
    base = Path(root, BATCHES_DIR)
    cache = {} if cache is None else cache
    try:
        dirs = [p for p in base.iterdir() if p.is_dir() and not p.name.startswith(".")]
    except OSError:
        return []
    out = []
    for p in dirs:
        try:
            sig = (p.stat().st_mtime_ns, (p / "status.json").stat().st_mtime_ns)
        except OSError:
            sig = None
        hit = cache.get(p.name)
        if hit and sig and hit[0] == sig and hit[1].state in TERMINAL:
            out.append(hit[1])
            continue
        try:
            found = info(p)
        except OSError as err:
            found = Info(p.name, p, p.name, "unreadable", {}, f"Cannot inspect batch: {err}")
        cache[p.name] = (sig, found)
        out.append(found)
    for gone in set(cache) - {p.name for p in dirs}:
        del cache[gone]
    return sorted(out, key=lambda i: i.status.get("created") or "", reverse=True)


def active(root):
    return next((i for i in scan(root) if i.active), None)


def request_stop(root, batch_id):
    """Ask the coordinator to stop (idempotent: the request is a file). Returns what happened."""
    bdir = directory(root, batch_id)
    found = info(bdir)
    if not found.active:
        return f"not requested: batch is {WORDS.get(found.state, found.state).lower()}"
    try:
        Path(bdir, "stop").touch()
    except OSError as err:
        return f"not requested: {err}"
    return "stop requested"


def new_id(draft_id):
    """Unique batch id: launch time (to the second) + the draft's id. Safe in run names and paths."""
    return f"{datetime.now():%Y%m%d-%H%M%S}-{draft_id}"


def with_id(plan_, batch_id):
    return replace(plan_, id=batch_id)

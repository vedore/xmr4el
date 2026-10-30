"""The detached batch coordinator (tui.md §5.7): `python -m xmr4el_tui.coordinator <batch dir>`, run from the repo root.

Runs the trees of outputs/batches/<id>/batch.json one at a time through the existing runner (`process.spawn`), writes
status.json, and exits when the batch is final. It is not a Textual app and does not depend on the TUI staying open.
Stdlib only. The user-facing contract is in batch.py; the failure rules are:
- a tree whose runner fails never stops the batch (policy continue_on_tree_failure, no retries);
- a stop request, a changed shared input, a plan that does not match its recorded hash, or a status file that cannot be
  written end the batch: unstarted trees become `cancelled`, completed runs are untouched.
Stops signal only the runner's process group, which this process created and still owns (an unreaped child).
"""
import hashlib
import json
import os
import signal
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

from xmr4el_tui import batch, procinfo, process, runs
from xmr4el_tui.logs import LogTail

POLL = 0.2          # seconds between checks while a runner runs
GRACE = 30          # seconds between SIGTERM and SIGKILL of a stopped runner
LOCK_WAIT = 3       # seconds to wait for the batch lock (a TUI's liveness probe holds it for an instant)


class Fatal(Exception):
    """A batch-wide failure: the message is the batch's error."""

    def __init__(self, state, message):
        super().__init__(message)
        self.state = state


def log(message):
    print(f"{datetime.now().isoformat(timespec='seconds')} {message}", flush=True)


class Coordinator:
    def __init__(self, root, bdir, runner):
        self.root, self.dir, self.runner = Path(root), Path(bdir), runner
        self.child = None
        self.signalled = False
        self.stopped = False  # a stop request ended a running tree
        self.status = batch.read_status(self.dir)
        self.plan = None

    # --- status ------------------------------------------------------------------------------------------------------
    def save(self, **changes):
        """Persist status.json; a second failure is a batch-wide failure (the TUI then reports the batch lost)."""
        self.status.update(changes)
        for attempt in (1, 2):
            try:
                batch.write_status(self.dir, self.status)
                return
            except OSError as err:
                log(f"status.json not written (attempt {attempt}): {err}")
                time.sleep(0.5)
        raise OSError("status.json could not be written after two attempts")

    def tree(self, tree_id):
        return next(t for t in self.status["trees"] if t["id"] == tree_id)

    def stopping(self):
        return self.signalled or batch.stop_requested(self.dir)

    def on_signal(self, signum, frame):
        self.signalled = True

    # --- the batch -----------------------------------------------------------------------------------------------------
    def run(self):
        self.plan = self.load_plan()
        self.save(state="running", started=batch.now(), coordinator=procinfo.own_identity())
        log(f"batch {self.status['batch_id']} running: {len(self.plan['trees'])} trees")
        reason = None
        for spec in self.plan["trees"]:
            if self.stopping():
                reason = Fatal("stopped", "Stop Batch requested")
                break
            try:
                self.check_inputs()
            except Fatal as err:
                reason = err
                break
            if self.stopping():  # hashing large inputs can outlast a Stop request
                reason = Fatal("stopped", "Stop Batch requested")
                break
            self.run_tree(spec)
        if reason is None and self.stopped:  # the stop ended the last tree
            reason = Fatal("stopped", "Stop Batch requested")
        self.finish(reason)

    def load_plan(self):
        path = self.dir / "batch.json"
        try:
            raw = path.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            plan = json.loads(raw)
        except (OSError, ValueError) as err:
            raise Fatal("blocked", f"batch.json is unreadable: {err}") from err
        if digest != self.status.get("plan_sha256") or [t["id"] for t in plan.get("trees", ())] != \
                [t["id"] for t in self.status["trees"]]:
            raise Fatal("blocked", "batch.json does not match the plan recorded at launch (corrupted or edited)")
        return plan

    def check_inputs(self):
        changed = batch.changed_inputs(self.root, self.plan["inputs"])
        if changed:
            raise Fatal("blocked", "Shared input changed since launch (sha256 differs or file missing): "
                        + ", ".join(changed))

    def finish(self, reason):
        trees = self.status["trees"]
        for t in trees:
            if t["state"] == "pending":
                t.update(state="cancelled", finished=batch.now())
        if reason is not None:
            state, error = reason.state, str(reason)
        else:
            bad = any(t["state"] in ("failed", "cancelled") for t in trees)
            state, error = ("completed_with_failures" if bad else "completed"), None
        self.save(state=state, error=error, current=None, finished=batch.now())
        log(f"batch {state}" + (f": {error}" if error else ""))

    # --- one tree ------------------------------------------------------------------------------------------------------
    def run_tree(self, spec):
        entry = self.tree(spec["id"])
        run_id = f"{datetime.now():%Y-%m-%d_%H-%M-%S}_{spec['spec']['name']}"
        entry.update(state="running", run_id=run_id, started=batch.now())
        self.save(current=spec["id"])
        log(f"tree {spec['id']} ({spec['name']}) starting as {run_id}")
        try:
            self.child = process.spawn(self.root, spec["spec"], run_id, config_sha=spec["config_sha256"],
                                       runner=self.runner)
        except (OSError, ValueError) as err:  # FileExistsError is an OSError: the run id was used before
            entry.update(state="failed", finished=batch.now(), error=f"runner did not start: {err}")
            self.child = None
            self.save(current=None)
            return
        entry.update(pid=self.child.pid)
        self.save()
        code, killed = self.wait()
        self.child = None
        self.classify(entry, run_id, code, killed, spec)
        self.save(current=None)
        log(f"tree {spec['id']} {entry['state']}" + (f": {entry['error']}" if entry["error"] else ""))

    def signal_group(self, sig):
        """Signal the runner's process group. The child is this process's own unreaped child that leads the group it
        created with start_new_session, so the group id cannot be reused before we reap it."""
        try:
            if self.child.returncode is None:  # getpgid on an exited leader fails on macOS even before wait()
                os.killpg(self.child.pid, sig)
        except (ProcessLookupError, PermissionError):
            pass

    def wait(self):
        """Wait for the runner; on a stop request SIGTERM its group, SIGKILL after GRACE. -> (exit code, how killed)."""
        killed, since = None, 0.0
        while True:
            if killed is None:
                code = self.child.poll()
                if code is not None:
                    return code, None
            elif killed == "SIGKILL" or procinfo.identity(self.child.pid) is None:
                # Keep the leader unreaped until its workers are killed: its PID/group cannot be reused.
                self.signal_group(signal.SIGKILL)
                return self.child.wait(), killed
            if self.stopping() and killed is None:
                self.save(stop_requested=self.status["stop_requested"] or batch.now())
                log(f"stop: SIGTERM to the runner's group (pid {self.child.pid})")
                self.signal_group(signal.SIGTERM)
                killed, since = "SIGTERM", time.monotonic()
            elif killed == "SIGTERM" and time.monotonic() - since > GRACE:
                log("stop: grace period over, SIGKILL")
                self.signal_group(signal.SIGKILL)
                killed = "SIGKILL"
            time.sleep(POLL)

    def classify(self, entry, run_id, code, killed, spec):
        run = runs.read_run(self.root / runs.RUNS_DIR / run_id, self.root)
        if killed == "SIGKILL" and run.recorded == "running" and run.identity:
            log(f"recording the forced stop: {process.record_forced(run)}")
            run = runs.read_run(run.path, self.root)
        entry.update(exit_code=code, finished=batch.now())
        if code == 0 and run.recorded == "completed":
            expected = ["spec.json", "model_config.json", "provenance.json", "tree.json",
                        *(f"metrics_{e['split']}.json" for e in spec["spec"]["evals"])]
            outputs = {name: runs.read_dict(run.path / name) for name in expected}
            missing = [name for name, data in outputs.items() if data is None]
            if not run.tree_here:
                missing.append("saved tree xmodel.pkl")
            if missing:
                entry.update(state="failed", error="Missing or invalid runner outputs: " + ", ".join(missing))
                return
            recorded = outputs["provenance.json"].get("inputs")
            changed = [p for p, sha in self.plan["inputs"].items()
                       if not isinstance(recorded, dict) or recorded.get(p) != sha]
            if changed:
                error = "Runner inputs differ from the approved batch snapshot: " + ", ".join(changed)
                entry.update(state="failed", error=error)
                raise Fatal("blocked", error)
            entry.update(state="succeeded", error=None)
        elif killed or run.recorded == "stopped":
            self.stopped = bool(killed)
            entry.update(state="cancelled", error=run.error or f"stopped ({killed or 'externally'})")
        else:
            rejected = process.rejection(self.root, run_id) if code == 3 else None
            tail = self.console_tail(run_id)
            entry.update(state="failed", error=f"not started: {rejected['error']}" if rejected else
                         run.error or f"runner exited {code} with run status {run.recorded}" +
                         (f"; {tail}" if tail else ""))

    def console_tail(self, run_id):
        lines, _ = LogTail(self.root / process.CONSOLE_DIR / f"{run_id}.log").read(final=True)
        return lines[-1].text[:300] if lines else ""

    def abort(self, err):
        """Unexpected error: stop the runner (our child), cancel what has not run, record the batch failed."""
        log("coordinator error:\n" + "".join(traceback.format_exception(err)))
        if self.child is not None and self.child.returncode is None:
            self.signal_group(signal.SIGTERM)
            deadline = time.monotonic() + GRACE
            while procinfo.identity(self.child.pid) is not None and time.monotonic() < deadline:
                time.sleep(POLL)
            self.signal_group(signal.SIGKILL)
            self.child.wait()
        for t in self.status["trees"]:
            if t["state"] == "running":
                t.update(state="failed", finished=batch.now(), error=f"coordinator error: {err}")
        self.finish(Fatal("failed", f"coordinator error: {type(err).__name__}: {err}"))


def main(argv=None):
    argv = sys.argv if argv is None else argv
    root, bdir = Path.cwd(), Path(argv[1])
    runner = os.environ.get("XMR4EL_TEST_RUNNER") or process.RUNNER  # the test seam; unset in production
    here = root / bdir
    held = []
    try:
        mine = batch.try_lock(here / "coordinator.lock", LOCK_WAIT)
        if mine is None:
            log("another coordinator holds this batch; exiting without touching it")
            return 2
        held.append(mine)
        status = batch.read_status(here)
        if not status or status.get("state") != "preparing":
            log("batch is not a fresh prepared plan; refusing to restart it")
            return 2
        (root / batch.BATCHES_DIR).mkdir(parents=True, exist_ok=True)
        slot = batch.try_lock(root / batch.GLOBAL_LOCK)
        if slot is None:
            Coordinator(root, here, runner).finish(
                Fatal("failed", f"Another batch is running ({batch.holder(root) or 'unknown'})."))
            return 2
        held.append(slot)
        os.ftruncate(slot, 0)
        os.write(slot, (status or {}).get("name", here.name).encode())
        coord = Coordinator(root, here, runner)
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, coord.on_signal)
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        try:
            coord.run()
        except Fatal as err:
            coord.finish(err)
        except Exception as err:  # noqa: BLE001 - recorded as a failed batch, never silent
            coord.abort(err)
    except SystemExit as err:
        return err.code
    finally:
        for fd in reversed(held):
            batch.unlock(fd)
    return 0


if __name__ == "__main__":
    sys.exit(main())

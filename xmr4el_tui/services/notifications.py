"""Session notifications (tui.md §4 Notifications): experiment lifecycle events seen by the app's scans.

Memory only: nothing is written, nothing survives the TUI. The first scan only records each run's state (no flood
of old results); a notice is made when a run seen active (RUNNING / UNVERIFIED) is later seen ended, or when a run
started during the session first appears already ended (shorter than a scan interval). LOST is provisional: the run
stays watched, and a final status recorded later (COMPLETED / FAILED / STOPPED) replaces the interruption notice; an
unreadable status (UNKNOWN) in between is waited out. Runs are told apart by incarnation (directory name, recorded
start and process identity), so a new run reusing a directory name is a new run and an old notice never opens it.
One notice per (incarnation, end state) per session. Stdlib only, no Textual.
"""
from collections import deque
from dataclasses import dataclass
from datetime import datetime

from xmr4el_tui.explorer import ACTIVE, descriptor
from xmr4el_tui.runs import state_word, training_run

HISTORY = 50
KIND = {"train": "Training", "eval": "Evaluation"}
FINAL = ("COMPLETED", "FAILED", "STOPPED")  # recorded outcomes: nothing about the run changes after them


TERMINAL_NOTES = {"completed": ("success", "Batch completed", "{ok} of {n} trees succeeded."),
                  "completed_with_failures": ("warning", "Batch completed with failures",
                                              "{ok} succeeded, {bad} failed."),
                  "stopped": ("information", "Batch stopped", "Stopped on request."),
                  "blocked": ("error", "Batch blocked", "A batch-wide problem stopped it."),
                  "failed": ("error", "Batch failed", "The batch could not run."),
                  "lost": ("warning", "Batch interrupted", "The coordinator ended unexpectedly.")}


@dataclass(frozen=True)
class Notice:
    key: tuple             # (incarnation, end state) or ("scan", problem): the deduplication identity
    severity: str          # success | error | warning | information (also the toast's -<severity> class)
    title: str
    subject: str           # run name · features / depth of its tree
    text: str
    at: datetime
    target: tuple | None   # ("results", explorer eval key, incarnation) | ("monitor", run id, incarnation) | None


def incarnation(run):
    """Which run behind a directory name: its recorded start and process identity (pid and its start token)."""
    rec = run.identity or {}
    return run.id, run.started, rec.get("pid"), rec.get("start")


def subject(run, all_runs):
    trainer = run if run.kind == "train" else training_run(all_runs, run.tree_id)
    return " · ".join(p for p in (run.name, descriptor(trainer)) if p)


def lifecycle(run, state, all_runs, now, requested=False):
    """The notice for `run` having ended in `state`, None for states that are not an end. `requested`: this TUI sent
    the stop (without it a STOPPED run names its signal and claims no intent)."""
    kind, name, inc = KIND.get(run.kind, "Run"), subject(run, all_runs), incarnation(run)
    monitor = ("monitor", run.id, inc)
    if state == "COMPLETED":
        done = [e for e in run.evals if e.metrics]
        # Results needs one baseline: never pick among several evaluations
        target = ("results", ("eval", run.id, done[0].split), inc) if len(done) == 1 else monitor
        return Notice((inc, state), "success", "Experiment completed" if run.kind == "train" else
                      f"{kind} completed", name, "Results are available." if done else "No metrics were written.",
                      now, target)
    if state == "FAILED":
        error = (run.error or "").splitlines()
        return Notice((inc, state), "error", f"{kind} failed", name, error[0] if error else
                      "The run ended with an error.", now, monitor)
    if state == "LOST":
        return Notice((inc, state), "warning", "Run interrupted", name,
                      "The process exited without a final status.", now, monitor)
    if state == "STOPPED":
        if requested:
            return Notice((inc, state), "information", "Run cancelled", name, "Stopped on request.", now, monitor)
        return Notice((inc, state), "warning", "Run stopped", name,
                      f"Stopped by {run.error}." if run.error else "Recorded as stopped.", now, monitor)
    return None


class Notifications:
    """Bounded history (newest first), unseen count, and the runs being watched for their outcome."""

    def __init__(self):
        self.items = deque(maxlen=HISTORY)
        self.unseen = 0
        self.watched = None    # incarnations seen active (or LOST) whose final outcome is still due; None: no scan yet
        self.known = set()     # incarnations of the last scan
        self.since = None      # start of the first scan, to the second: runs started at or after it are this session's
        self.requested = set() # incarnations this TUI sent a stop to
        self.batches = None    # batch id -> (state, {tree id: state}) of the last scan; None: no scan yet
        self.sent = set()      # notice keys made this session (Clear all keeps them: an event is reported once)

    def observe(self, all_runs, now=None, problem=None):
        """New notices for runs that ended since the last scan (none on the first scan) and for a new scan
        `problem`. `now`: when the scan started (a run started after it is new even if found already ended)."""
        now = now or datetime.now()
        states = {incarnation(r): (r, state_word(r)) for r in all_runs}
        new = [Notice(("scan", problem), "warning", "Scan failed", "outputs/runs/", problem, now, None)] if problem \
            else []
        if self.watched is None:
            self.since = now.replace(microsecond=0)  # the runner records start times to the second
            self.watched = {inc for inc, (_, w) in states.items() if w in ACTIVE}
            self.known = set(states)
            return self.add(new)
        for inc, (run, word) in states.items():
            if word in ACTIVE:
                self.watched.add(inc)
                continue
            # new and already ended: only if it started in this session (a copied-in old run directory is not an event)
            if inc in self.watched or (inc not in self.known and run.started is not None and run.started >= self.since):
                n = lifecycle(run, word, all_runs, now, inc in self.requested)
                new += [n] if n else []
                if word in FINAL:
                    self.watched.discard(inc)
                else:  # LOST (a final status may still be read) or UNKNOWN (status.json unreadable): keep watching
                    self.watched.add(inc)
        # runs whose directory is gone and cleared scan problems are forgotten: memory bounded by outputs/runs/. By
        # directory, not incarnation: an incarnation read while status.json was unreadable must not drop the real one
        ids = {r.id for r in all_runs}
        self.known = set(states)
        self.watched = {inc for inc in self.watched if inc[0] in ids}
        self.requested = {inc for inc in self.requested if inc[0] in ids}
        self.sent = {k for k in self.sent if k == ("scan", problem) or isinstance(k[0], tuple) and (k[0][0] in ids or k[0][0] == "batch")}
        return self.add(new)

    def observe_batches(self, infos, now=None):
        """Notices for batch transitions since the last scan: batch started, each tree started / finished, batch ended.
        The first scan only records (old batches are no events); a batch first seen later has its whole history
        still ahead of it, so a short one is reported from its first state."""
        now = now or datetime.now()
        seen = {i.id: (i.state, {t["id"]: t["state"] for t in i.status.get("trees", ())}) for i in infos}
        if self.batches is None:
            self.batches = seen
            return []
        new = []
        for i in infos:
            state, trees = seen[i.id]
            old_state, old_trees = self.batches.get(i.id, (None, {}))
            ref, target = ("batch", i.id), ("batch", i.id, i.id)

            def note(key, severity, title, text):
                new.append(Notice((ref, key), severity, title, i.name, text, now, target))
            if i.id not in self.batches and state != "unreadable":
                note("started", "information", "Batch started", f"{len(trees)} trees queued.")
            names = {t["id"]: t["name"] for t in i.status.get("trees", ())}
            for tid, tstate in trees.items():
                if tstate == old_trees.get(tid):
                    continue
                if tstate == "running":
                    note(f"{tid}:running", "information", "Tree started", names[tid])
                elif tstate == "succeeded":
                    note(f"{tid}:succeeded", "success", "Tree completed", names[tid])
                elif tstate in ("failed", "interrupted"):
                    err = next((t["error"] for t in i.status["trees"] if t["id"] == tid), None) or "see its log"
                    note(f"{tid}:{tstate}", "error", "Tree failed" if tstate == "failed" else "Tree interrupted",
                         f"{names[tid]}: {str(err).splitlines()[0]}")
            if state != old_state and state in TERMINAL_NOTES:
                severity, title, text = TERMINAL_NOTES[state]
                note(state, severity, title, i.status.get("error") or text.format(
                    ok=i.count("succeeded"), bad=i.count("failed"), n=len(trees)))
        self.batches = seen
        return self.add(new)

    def add(self, notices):
        new = [n for n in notices if n.key not in self.sent]
        self.sent.update(n.key for n in new)
        # a recorded outcome replaces the provisional interruption notice of the same run
        stale = {(n.key[0], "LOST") for n in new if n.key[1] in FINAL}
        if stale:  # the unseen notices are the newest `unseen` items
            self.unseen -= sum(n.key in stale for n in list(self.items)[:self.unseen])
            self.items = deque((n for n in self.items if n.key not in stale), maxlen=HISTORY)
        self.items.extendleft(new)
        self.unseen = min(self.unseen + len(new), len(self.items))
        return new

    def seen(self):
        self.unseen = 0

    def clear(self):
        self.items.clear()
        self.unseen = 0

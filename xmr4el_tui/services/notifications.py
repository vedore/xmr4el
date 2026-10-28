"""Session notifications (tui.md §4 Notifications): experiment lifecycle events seen by the app's scans.

Memory only: nothing is written, nothing survives the TUI. The first scan only records each run's state (no flood
of old results); a notice is made when a run seen active (RUNNING / UNVERIFIED) is next seen ended, or when a run
started during the session first appears already ended (shorter than a scan interval). One notice per (run id, end
state) per session, so repeated scans of an ended run add nothing. Stdlib only, no Textual.
"""
from collections import deque
from dataclasses import dataclass
from datetime import datetime

from xmr4el_tui.explorer import ACTIVE, descriptor
from xmr4el_tui.runs import state_word, training_run

HISTORY = 50
KIND = {"train": "Training", "eval": "Evaluation"}


@dataclass(frozen=True)
class Notice:
    key: tuple             # (run id, end state) or ("scan", problem): the deduplication identity
    severity: str          # success | error | warning | information (also the toast's -<severity> class)
    title: str
    subject: str           # run name · features / depth of its tree
    text: str
    at: datetime
    target: tuple | None   # ("results", explorer eval key) | ("monitor", run id) | None


def subject(run, all_runs):
    trainer = run if run.kind == "train" else training_run(all_runs, run.tree_id)
    return " · ".join(p for p in (run.name, descriptor(trainer)) if p)


def lifecycle(run, state, all_runs, now):
    """The notice for `run` having ended in `state`, None for states that are not an end."""
    kind, name = KIND.get(run.kind, "Run"), subject(run, all_runs)
    monitor = ("monitor", run.id)
    if state == "COMPLETED":
        done = [e for e in run.evals if e.metrics]
        # Results needs one baseline: never pick among several evaluations
        target = ("results", ("eval", run.id, done[0].split)) if len(done) == 1 else monitor
        return Notice((run.id, state), "success", "Experiment completed" if run.kind == "train" else
                      f"{kind} completed", name, "Results are available." if done else "No metrics were written.",
                      now, target)
    if state == "FAILED":
        error = (run.error or "").splitlines()
        return Notice((run.id, state), "error", f"{kind} failed", name, error[0] if error else
                      "The run ended with an error.", now, monitor)
    if state == "LOST":
        return Notice((run.id, state), "warning", "Run interrupted", name,
                      "The process exited without a final status.", now, monitor)
    if state == "STOPPED":
        return Notice((run.id, state), "information", "Run cancelled", name, "Stopped on request.", now, monitor)
    return None


class Notifications:
    """Bounded history (newest first), unseen count, and the run states of the last scan."""

    def __init__(self):
        self.items = deque(maxlen=HISTORY)
        self.unseen = 0
        self.states = None  # run id -> state word at the last scan; None before the first
        self.since = None   # time of the first scan: runs started after it are this session's
        self.sent = set()   # notice keys made this session (Clear all keeps it: an event is reported once)

    def observe(self, all_runs, now=None):
        """New notices for runs that ended since the last scan (none on the first scan)."""
        now = now or datetime.now()
        before, self.states = self.states, {r.id: state_word(r) for r in all_runs}
        if before is None:
            self.since = now
            return []
        # new and already ended: only if it started in this session (a copied-in old run directory is not an event)
        return self.add([n for r in all_runs if before.get(r.id) in ACTIVE
                         or (r.id not in before and r.started is not None and r.started >= self.since)
                         for n in (lifecycle(r, self.states[r.id], all_runs, now),) if n])

    def add(self, notices):
        new = [n for n in notices if n.key not in self.sent]
        self.sent.update(n.key for n in new)
        self.items.extendleft(new)
        self.unseen = min(self.unseen + len(new), len(self.items))
        return new

    def seen(self):
        self.unseen = 0

    def clear(self):
        self.items.clear()
        self.unseen = 0

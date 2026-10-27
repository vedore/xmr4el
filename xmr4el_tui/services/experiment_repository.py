"""Scan of outputs/runs/ and outputs/saved_trees/ into one Snapshot (tui.md §10). Stdlib only, no UI.

The repository owns the read cache, so one scan at a time may call `snapshot` (the app serializes its scans).
"""
from dataclasses import dataclass, field, replace
from pathlib import Path

from xmr4el_tui import procinfo, process, runs
from xmr4el_tui.progress import current_stage


@dataclass(frozen=True)
class Snapshot:
    runs: tuple = ()          # newest first, recorded-running runs carry their live state (§9)
    found: bool = False       # outputs/runs exists
    trees: tuple = ()
    active: dict = field(default_factory=dict)  # recorded-running run id -> (current stage, device)
    scanned: bool = False     # False: the first scan has not landed yet
    problem: str | None = None  # why this scan failed (the other fields are then the last successful scan's)


class ExperimentRepository:
    def __init__(self, root):
        self.root = Path(root)
        self.cache = {}  # runs.scan cache
        self.last = Snapshot()  # last successful scan

    def snapshot(self):
        """Read everything the Dashboard shows. Only reads files."""
        try:
            found = runs.scan(self.root, self.cache)
        except FileNotFoundError:
            self.last = Snapshot(trees=tuple(runs.trees(self.root, [])), scanned=True)
            return self.last
        except OSError as err:  # outputs/runs cannot be listed: keep the last good scan, say why
            return replace(self.last, scanned=True, problem=f"outputs/runs/ cannot be read ({type(err).__name__}); "
                                                            "showing the last successful scan")
        here = procinfo.runtime()
        found = [replace(r, live=process.derived_state(r, here)) if r.recorded == "running" else r for r in found]
        active = {r.id: (current_stage(r), device(r)) for r in found if r.recorded == "running"}
        self.last = Snapshot(tuple(found), True, tuple(runs.trees(self.root, found)), active, scanned=True)
        return self.last


def device(run):
    return runs.text(run.provenance(), "torch_device")

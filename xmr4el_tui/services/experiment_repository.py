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


class ExperimentRepository:
    def __init__(self, root):
        self.root = Path(root)
        self.cache = {}  # runs.scan cache

    def snapshot(self):
        """Read everything the Dashboard shows. Only reads files."""
        try:
            found = runs.scan(self.root, self.cache)
        except FileNotFoundError:
            return Snapshot(trees=tuple(runs.trees(self.root, [])), scanned=True)
        here = procinfo.runtime()
        found = [replace(r, live=process.derived_state(r, here)) if r.recorded == "running" else r for r in found]
        active = {r.id: (current_stage(r), device(r)) for r in found if r.recorded == "running"}
        return Snapshot(tuple(found), True, tuple(runs.trees(self.root, found)), active, scanned=True)


def device(run):
    prov = run.provenance()
    return prov.get("torch_device") if isinstance(prov, dict) else None

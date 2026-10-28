"""Incremental reading of a run's log for the log viewer (Dashboard View Logs; the Run monitor reuses it). Stdlib
only. The viewer polls `LogTail.read` while the run is active; a finished run is read once.
"""
import os
import re
from dataclasses import dataclass
from pathlib import Path

from xmr4el_tui.fmt import clean
from xmr4el_tui.process import CONSOLE_DIR

LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
# scripts/run_experiment.py's format: "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
HEADER = re.compile(r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3} \[(" + "|".join(LEVELS) + r")\] ")
TAIL_BYTES = 8 << 20  # at most this much is read per call: the first read (and a burst) starts this far from the end


@dataclass(frozen=True)
class Line:
    text: str
    level: str | None  # its record's level: continuation lines (tracebacks, multi-line messages) inherit it


def log_path(root, run):
    """run.log, else the console log of a TUI launch (stdout/stderr, e.g. a crash before logging started), else
    run.log (missing)."""
    for path in (run.path / "run.log", Path(root, CONSOLE_DIR, f"{run.id}.log")):
        if path.is_file():
            return path
    return run.path / "run.log"


class LogTail:
    """New complete lines of one file per `read`. A torn last line waits for its newline unless `final` (the writer
    exited). A truncated or replaced file (smaller than the offset, other inode) restarts: reset=True. Bytes never
    read (older than TAIL_BYTES, or a burst larger than it) are counted in `skipped`."""

    def __init__(self, path, tail_bytes=TAIL_BYTES):
        self.path, self.tail_bytes = Path(path), tail_bytes
        self.offset = self.inode = self.level = self.error = None
        self.pending, self.skipped, self.fragment = b"", 0, False  # fragment: inside a line older than the cut

    def read(self, final=False):
        """-> (new lines, reset). Missing or unreadable: ([], False) and `error` says why (None once readable)."""
        try:
            with open(self.path, "rb") as f:
                st = os.fstat(f.fileno())
                reset = self.offset is not None and (st.st_ino != self.inode or st.st_size < self.offset)
                if self.offset is None or reset:
                    self.offset, self.inode, self.level, self.skipped = 0, st.st_ino, None, 0
                    self.pending, self.fragment = b"", False
                if st.st_size - self.tail_bytes > self.offset:
                    self.skipped += st.st_size - self.tail_bytes - self.offset
                    self.offset, self.pending, self.fragment = st.st_size - self.tail_bytes, b"", True
                f.seek(self.offset)
                data = f.read(st.st_size - self.offset)
        except FileNotFoundError:
            self.error = "not created yet" if self.offset is None else "deleted"
            return [], False
        except OSError as err:
            self.error = err.strerror or type(err).__name__
            return [], False
        self.error = None
        self.offset += len(data)
        data = self.pending + data
        if self.fragment:
            nl = data.find(b"\n")
            data, self.fragment = (b"", True) if nl < 0 else (data[nl + 1:], False)
        end = len(data) if final else data.rfind(b"\n") + 1
        self.pending = data[end:]
        return self.lines(data[:end]), reset

    def lines(self, data):
        out = []
        for raw in data.decode("utf-8", "replace").splitlines():
            if m := HEADER.match(raw):
                self.level = m.group(1)
            out.append(Line(clean(raw.expandtabs(4)), self.level))
        return out

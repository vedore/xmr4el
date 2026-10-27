"""events.jsonl writer (docs/design/tui.md §8): one JSON object per line, appended in emission order."""
import json
import logging
import time
from datetime import datetime

SCHEMA_VERSION = 1
ENDS = ("complete", "failed", "stopped")  # phases that close a started stage / part
DETAIL = ("layer", "node")  # hierarchy detail: start events only, never open
WHOLE = "run"               # the whole run: a duration, no resource summary


def stage_key(event):
    """(stage, split, part): what a start event opens and an end event closes."""
    return event.get("stage"), event.get("split"), event.get("part")


class EventLog(logging.FileHandler):
    """Writes the `event` of each log record that has one as {"v", "t", **event}; other records are ignored.
    An end phase gets duration_seconds (monotonic, from its start); top-level stage starts/ends go to
    `resources` (a ResourceMonitor or None). One write + flush per line: a reader sees whole lines promptly,
    no fsync."""

    def __init__(self, path, resources=None):
        super().__init__(path)
        self.resources, self.open = resources, {}  # stage key -> monotonic start, in start order

    def emit(self, record):
        event = getattr(record, "event", None)
        if event is not None:
            self.write(dict(event), record.created)

    def write(self, event, created=None):
        with self.lock:  # RLock: emit already holds it
            now, key, phase = time.monotonic(), stage_key(event), event.get("phase")
            top = key[0] not in (*DETAIL, WHOLE) and key[2] is None
            if phase == "start" and key[0] not in DETAIL:
                self.open[key] = now
                if top and self.resources:
                    self.resources.start(key)
            elif phase in ENDS and key in self.open:
                event["duration_seconds"] = round(now - self.open.pop(key), 3)
                if top and self.resources:
                    self.resources.finish(key, phase, event["duration_seconds"])
            t = datetime.fromtimestamp(created or time.time()).astimezone().isoformat(timespec="milliseconds")
            self.stream.write(json.dumps({"v": SCHEMA_VERSION, "t": t, **event}) + "\n")
            self.flush()

    def end_open(self, phase, **fields):
        """Close every started stage / part still open (innermost first) with `phase` (failed | stopped)."""
        with self.lock:
            for stage, split, part in reversed(list(self.open)):
                self.write({"stage": stage, **({"split": split} if split else {}), **({"part": part} if part else {}),
                            "phase": phase, **fields})

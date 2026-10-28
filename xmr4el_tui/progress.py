"""events.jsonl -> stage list (tui.md §8). Stdlib only.

Runs without events.jsonl (before increment 2) get an empty stage list; events without `v` / duration_seconds /
failed / stopped phases (before schema v1) still read.
"""
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from xmr4el_tui.fmt import count, metric
from xmr4el_tui.runs import parse_time, state_word

TRAIN_STAGES = ("load", "encode", "labels", "hierarchy", "save")
ENDS = ("complete", "failed", "stopped")
EVAL_STAGES = ("load",)


@dataclass(frozen=True)
class Stage:
    key: str                  # load, encode, labels, hierarchy, save, eval:<split>; parts by part name
    state: str                # done | current | pending | failed | stopped
    seconds: float | None     # ended: its duration_seconds (monotonic), else start -> end; current: None
    start: datetime | None
    detail: str = ""          # "layer 1/2 · node 1/1 · 474,854 rows", "cache hit", ...
    parts: tuple["Stage", ...] = ()     # encode parts in fit order (§5.2), eval parts data/predict/metrics/write,
                                        # hierarchy layers ("layer 1": ended when the next layer or hierarchy ends)
    bar: tuple[int, int] | None = None  # (done, total) distinct mentions
    error: str | None = None  # a failed stage's recorded error


def read_events(path, offset=0):
    """Events appended after byte `offset` -> (events, new offset). A torn last line stays unread until complete;
    a malformed complete line is skipped. Missing or unreadable file -> ([], offset)."""
    try:
        with open(path, "rb") as f:
            f.seek(offset)
            data = f.read()
    except OSError:
        return [], offset
    end = data.rfind(b"\n") + 1
    events = []
    for line in data[:end].splitlines():
        try:
            event = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(event, dict):
            events.append(event)
    return events, offset + end


class EventTail:
    """events.jsonl read incrementally (the scan's and the Run monitor's reader): only bytes appended since the last
    `read`; a replaced or truncated file is read again from the start; an unchanged one is not read."""

    def __init__(self, path):
        self.path, self.file, self.size, self.offset, self.events = Path(path), None, 0, 0, []

    def read(self):
        """-> True when `events` changed. `events` is replaced, never mutated (a reader may hold the old list)."""
        try:
            st = self.path.stat()
            file, size = (st.st_dev, st.st_ino), st.st_size
        except OSError:
            file, size = None, 0
        if (file, size) == (self.file, self.size):
            return False
        if file != self.file or size < self.offset:
            self.events, self.offset = [], 0
        self.file, self.size = file, size
        if size > self.offset:
            new, self.offset = read_events(self.path, self.offset)
            self.events = self.events + new
        return True


def event_time(event):
    return parse_time(event.get("t"))


def _seconds(a, b):
    return (b - a).total_seconds() if a and b else None


def _detail(key, ev):
    if key == "encode" and "rows" in ev:
        return f"{count(ev['rows'])} rows"
    if key == "labels" and "labels" in ev:
        return f"{count(ev['labels'])} labels"
    return ""


def stages(events, kind, splits, recorded):
    """Stage list for a run of `kind` ("train" | "eval") with eval `splits`; [] without events. A stage or part ends
    with complete (done), failed or stopped; one started but never ended shows failed / stopped when the run
    recorded so (runs from before failed/stopped events, or killed)."""
    if not events:
        return []
    keys = [*(TRAIN_STAGES if kind == "train" else EVAL_STAGES), *(f"eval:{s}" for s in splits)]
    start, end, detail, parts = {}, {}, {}, {}  # key -> ...; parts: key -> {part: same dicts}
    layer = node = None
    layers = []  # [layer start event, its last node start event, start time]
    for ev in events:
        stage, phase, t = ev.get("stage"), ev.get("phase"), event_time(ev)
        if stage in ("layer", "node"):  # hierarchy detail only
            if phase == "start":
                layer, node = (ev, None) if stage == "layer" else (layer, ev)
                if stage == "layer":
                    layers.append([ev, None, t])
                elif layers:
                    layers[-1][1] = ev
            continue
        key = f"eval:{ev.get('split')}" if stage == "eval" else stage
        if key not in keys:
            continue
        part = ev.get("part")
        if isinstance(part, str) and part:
            p = parts.setdefault(key, {}).setdefault(part, {"detail": ""})
            if phase == "start":
                p["start"] = t
            elif phase in ENDS or phase == "cache_hit":
                p["end"] = ev, t
                p["detail"] = "cache hit" if phase == "cache_hit" else p["detail"]
            elif phase == "tick" and "done" in ev and "total" in ev:
                p["bar"] = (ev["done"], ev["total"])
                p["detail"] = f"{count(ev['done'])} / {count(ev['total'])} distinct mentions"
            continue
        if phase == "start":
            start[key], detail[key] = t, _detail(key, ev)
        elif phase in ENDS:
            end[key] = ev, t
            if "acc@1" in ev:
                detail[key] = f"acc@1 {metric(ev['acc@1'])}"
            elif _detail(key, ev):
                detail[key] = _detail(key, ev)
    if layer:
        detail["hierarchy"] = " · ".join(
            [f"layer {layer.get('layer')}/{layer.get('layers')}"]
            + ([f"node {node.get('i')}/{node.get('n')}", f"{count(node.get('rows'))} rows"] if node else []))
    unfinished = {"failed": "failed", "stopped": "stopped"}.get(recorded, "current")

    def stage(k, begun, ended, text="", sub=(), bar=None):
        if ended:
            ev, t = ended
            phase = ev.get("phase")
            seconds = ev.get("duration_seconds")
            return Stage(k, {"failed": "failed", "stopped": "stopped"}.get(phase, "done"),
                         seconds if isinstance(seconds, (int, float)) else _seconds(begun, t), begun, text, sub,
                         bar, ev.get("error") if isinstance(ev.get("error"), str) else None)
        return Stage(k, unfinished if begun else "pending", None, begun, text, sub, bar)

    def layer_parts():
        """A layer has only a start event: it ended when the next one started, or with the hierarchy."""
        out = []
        for i, (ev, last, t) in enumerate(layers):
            ended = ({"phase": "complete"}, layers[i + 1][2]) if i + 1 < len(layers) else end.get("hierarchy")
            text = f"{count(ev.get('nodes'))} nodes" if ended or not last else \
                f"node {last.get('i')}/{last.get('n')} · {count(last.get('rows'))} rows"
            out.append(stage(f"layer {ev.get('layer')}", t, ended and ({"phase": ended[0].get("phase")}, ended[1]),
                             text))
        return tuple(out)

    return [stage(k, start.get(k), end.get(k), detail.get(k, ""),
                  layer_parts() if k == "hierarchy" else
                  tuple(stage(name, p.get("start"), p.get("end"), p["detail"], bar=p.get("bar"))
                        for name, p in parts.get(k, {}).items()))
            for k in keys]


def last_event_age(events, now):
    """Seconds since the last event with a readable time, None without one."""
    t = next((t for t in map(event_time, reversed(events)) if t), None)
    return _seconds(t, now)


def current_stage(run, events):
    """`hierarchy  layer 1/2 · …` of the run's current stage in its `events`, None without one."""
    recorded = "failed" if state_word(run) == "LOST" else run.recorded  # a lost run's current stage failed
    s = next((s for s in stages(events, run.kind, [e.split for e in run.evals], recorded) if s.state == "current"),
             None)
    return f"{s.key.replace(':', ' ')}  {s.detail}".rstrip() if s else None


def last_stage_event(events):
    """(stage, time) of the last event with both, None without one."""
    return next(((e["stage"], t) for e in reversed(events) if e.get("stage") and (t := event_time(e))), None)

"""events.jsonl -> stage list (tui.md §8). Stdlib only.

The runner starts writing events.jsonl in increment 2; runs without it get an empty stage list.
"""
import json
from dataclasses import dataclass
from datetime import datetime

from xmr4el_tui.fmt import count
from xmr4el_tui.runs import state_word

TRAIN_STAGES = ("load", "encode", "labels", "hierarchy", "save")
EVAL_STAGES = ("load",)


@dataclass(frozen=True)
class Stage:
    key: str                  # load, encode, labels, hierarchy, save, eval:<split>; encode parts by part name
    state: str                # done | current | pending | failed
    seconds: float | None     # done/failed: start -> complete; current: None (UI counts from `start`)
    start: datetime | None
    detail: str = ""          # "layer 1/2 · node 1/1 · 474,854 rows", "cache hit", ...
    parts: tuple["Stage", ...] = ()     # encode parts in fit order (§5.2)
    bar: tuple[int, int] | None = None  # (done, total) distinct mentions


def read_events(path, offset=0):
    """Events appended after byte `offset` -> (events, new offset). A torn last line stays unread until complete;
    a malformed complete line is skipped. Missing file -> ([], offset)."""
    try:
        with open(path, "rb") as f:
            f.seek(offset)
            data = f.read()
    except FileNotFoundError:
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


def event_time(event):
    try:
        return datetime.fromisoformat(event["t"])
    except (KeyError, TypeError, ValueError):
        return None


def _seconds(a, b):
    return (b - a).total_seconds() if a and b else None


def _detail(key, ev):
    if key == "encode" and "rows" in ev:
        return f"{count(ev['rows'])} rows"
    if key == "labels" and "labels" in ev:
        return f"{count(ev['labels'])} labels"
    return ""


def stages(events, kind, splits, recorded):
    """Stage list for a run of `kind` ("train" | "eval") with eval `splits`; [] without events. When the run
    recorded failed/stopped, its current stage shows failed."""
    if not events:
        return []
    keys = [*(TRAIN_STAGES if kind == "train" else EVAL_STAGES), *(f"eval:{s}" for s in splits)]
    start, end, detail = {}, {}, {}
    parts, part_start, part_end, part_detail, bar = [], {}, {}, {}, {}
    layer = node = None
    for ev in events:
        stage, phase, t = ev.get("stage"), ev.get("phase"), event_time(ev)
        if stage in ("layer", "node"):  # hierarchy detail only
            if phase == "start":
                layer, node = (ev, None) if stage == "layer" else (layer, ev)
            continue
        key = f"eval:{ev.get('split')}" if stage == "eval" else stage
        part = ev.get("part")
        if stage == "encode" and part:
            if part not in parts:
                parts.append(part)
            if phase == "start":
                part_start[part] = t
            elif phase in ("complete", "cache_hit"):
                part_end[part] = t
                part_detail[part] = "cache hit" if phase == "cache_hit" else part_detail.get(part, "")
            elif phase == "tick" and "done" in ev and "total" in ev:
                bar[part] = (ev["done"], ev["total"])
                part_detail[part] = f"{count(ev['done'])} / {count(ev['total'])} distinct mentions"
            continue
        if key not in keys:
            continue
        if phase == "start":
            start[key], detail[key] = t, _detail(key, ev)
        elif phase == "complete":
            end[key] = t
            if "acc@1" in ev:
                detail[key] = f"acc@1 {ev['acc@1']:.4f}"
            elif _detail(key, ev):
                detail[key] = _detail(key, ev)
    if layer:
        detail["hierarchy"] = " · ".join(
            [f"layer {layer.get('layer')}/{layer.get('layers')}"]
            + ([f"node {node.get('i')}/{node.get('n')}", f"{count(node.get('rows'))} rows"] if node else []))
    stopped = recorded in ("failed", "stopped")

    def state(k, starts, ends):
        if k in ends:
            return "done"
        if k in starts:
            return "failed" if stopped else "current"
        return "pending"

    out = []
    for k in keys:
        sub = ()
        if k == "encode":
            sub = tuple(Stage(p, state(p, part_start, part_end), _seconds(part_start.get(p), part_end.get(p)),
                              part_start.get(p), part_detail.get(p, ""), bar=bar.get(p)) for p in parts)
        out.append(Stage(k, state(k, start, end), _seconds(start.get(k), end.get(k)), start.get(k),
                         detail.get(k, ""), sub))
    return out


def last_event_age(events, now):
    """Seconds since the last event with a readable time, None without one."""
    t = next((t for t in map(event_time, reversed(events)) if t), None)
    return _seconds(t, now)


def stage_list(run):
    """Stages from events.jsonl, or None when the run has no events file (written before the runner change)."""
    path = run.path / "events.jsonl"
    if not path.exists():
        return None
    events, _ = read_events(path)
    recorded = "failed" if state_word(run) == "LOST" else run.recorded  # a lost run's current stage failed
    return stages(events, run.kind, [e.split for e in run.evals], recorded)


def current_stage(run):
    """`hierarchy  layer 1/2 · …` of the run's current stage, None without one."""
    s = next((s for s in stage_list(run) or () if s.state == "current"), None)
    return f"{s.key.replace(':', ' ')}  {s.detail}".rstrip() if s else None

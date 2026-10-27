import json
from datetime import datetime

from xmr4el_tui.progress import last_event_age, read_events, stages

EVENTS = [
    {"t": "2026-10-20T13:44:00", "stage": "load", "phase": "start"},
    {"t": "2026-10-20T13:44:06", "stage": "load", "phase": "complete"},
    {"t": "2026-10-20T13:44:11", "stage": "encode", "phase": "start", "rows": 474854},
    {"t": "2026-10-20T13:44:11", "stage": "encode", "part": "mention_tfidf", "phase": "start"},
    {"t": "2026-10-20T13:48:09", "stage": "encode", "part": "mention_tfidf", "phase": "complete"},
    {"t": "2026-10-20T13:48:09", "stage": "encode", "part": "transformer", "phase": "start"},
    {"t": "2026-10-20T13:49:00", "stage": "encode", "part": "transformer", "phase": "tick", "unit": "distinct_mentions",
     "done": 50000, "total": 100000},
]


def test_read_events_torn_and_bad_lines(tmp_path):
    path = tmp_path / "events.jsonl"
    assert read_events(path) == ([], 0)
    path.write_text(json.dumps(EVENTS[0]) + "\nnot json\n" + '{"t": "2026')
    events, offset = read_events(path)
    assert events == [EVENTS[0]] and offset == len(path.read_bytes()) - len('{"t": "2026')
    with open(path, "a") as f:
        f.write('-10-20T13:44:06", "stage": "load", "phase": "complete"}\n')
    more, offset2 = read_events(path, offset)
    assert more == [EVENTS[1]] and offset2 == len(path.read_bytes())


def test_stages():
    assert stages([], "train", ["dev"], "running") == []  # runs from before events.jsonl
    got = stages(EVENTS, "train", ["dev"], "running")
    assert [(s.key, s.state) for s in got] == [("load", "done"), ("encode", "current"), ("labels", "pending"),
                                               ("hierarchy", "pending"), ("save", "pending"),
                                               ("eval:dev", "pending")]
    load, encode = got[:2]
    assert load.seconds == 6 and encode.seconds is None and encode.detail == "474,854 rows"
    assert [(p.key, p.state, p.bar) for p in encode.parts] == [("mention_tfidf", "done", None),
                                                               ("transformer", "current", (50000, 100000))]
    assert stages(EVENTS, "train", ["dev"], "failed")[1].state == "failed"
    assert [s.key for s in stages(EVENTS[:2], "eval", ["test"], "running")] == ["load", "eval:test"]


def test_stage_details():
    events = [{"t": "2026-10-20T13:52:09", "stage": "encode", "part": "transformer", "phase": "cache_hit"},
              {"t": "2026-10-20T13:52:09", "stage": "hierarchy", "phase": "start"},
              {"t": "2026-10-20T13:52:09", "stage": "layer", "phase": "start", "layer": 1, "layers": 2},
              {"t": "2026-10-20T13:52:09", "stage": "node", "phase": "start", "layer": 1, "i": 1, "n": 1,
               "rows": 474854},
              {"t": "2026-10-20T14:00:00", "stage": "eval", "split": "dev", "phase": "complete", "acc@1": 0.94634}]
    got = {s.key: s for s in stages(events, "train", ["dev"], "running")}
    assert got["hierarchy"].detail == "layer 1/2 · node 1/1 · 474,854 rows"
    assert got["encode"].parts[0].detail == "cache hit" and got["encode"].parts[0].state == "done"
    assert got["eval:dev"].detail == "acc@1 0.9463"
    assert last_event_age(events + [{"t": "bad"}], datetime(2026, 10, 20, 14, 1)) == 60
    assert last_event_age([], datetime(2026, 10, 20)) is None


def test_ended_stages_schema_v1():
    """failed / stopped end events, monotonic duration_seconds over timestamps, eval parts; old events still read."""
    events = [{"v": 1, "t": "2026-10-20T13:44:00+00:00", "stage": "run", "phase": "start"},
              {"v": 1, "t": "2026-10-20T13:44:00+00:00", "stage": "load", "phase": "start"},
              {"v": 1, "t": "2026-10-20T13:44:09+00:00", "stage": "load", "phase": "complete",
               "duration_seconds": 8.5},
              {"t": "2026-10-20T13:44:10+00:00", "stage": "eval", "split": "dev", "phase": "start"},
              {"t": "2026-10-20T13:44:10+00:00", "stage": "eval", "split": "dev", "part": "data", "phase": "start"},
              {"t": "2026-10-20T13:44:12+00:00", "stage": "eval", "split": "dev", "part": "data", "phase": "complete"},
              {"t": "2026-10-20T13:44:12+00:00", "stage": "eval", "split": "dev", "part": "predict", "phase": "start"},
              {"t": "2026-10-20T13:44:20+00:00", "stage": "eval", "split": "dev", "part": "predict",
               "phase": "stopped", "signal": "SIGTERM"},
              {"t": "2026-10-20T13:44:20+00:00", "stage": "eval", "split": "dev", "phase": "stopped"},
              {"t": "2026-10-20T13:44:20+00:00", "stage": "run", "phase": "stopped"}]
    load, dev, test = stages(events, "eval", ["dev", "test"], "stopped")
    assert (load.state, load.seconds) == ("done", 8.5), "the recorded monotonic duration wins"
    assert (dev.state, dev.seconds, test.state) == ("stopped", 10, "pending")
    assert [(p.key, p.state, p.seconds) for p in dev.parts] == [("data", "done", 2), ("predict", "stopped", 8)]
    assert stages(events[:7], "eval", ["dev"], "stopped")[1].parts[1].state == "stopped", "killed: from status"
    failed = events[:4] + [{"t": "2026-10-20T13:44:11", "stage": "eval", "split": "dev", "phase": "failed",
                            "error": "ValueError: x"}]
    assert stages(failed, "eval", ["dev"], "failed")[1].error == "ValueError: x"

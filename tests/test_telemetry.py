"""xmr4el.telemetry: events.jsonl lines (schema version, offset times, monotonic durations, open stages closed on
failure) and per-stage resource summaries (exact CPU, sampled memory, null when a metric is unavailable)."""
import json
import logging
import threading
from datetime import datetime

from xmr4el.telemetry import events as ev_mod
from xmr4el.telemetry import resources
from xmr4el.telemetry.events import EventLog
from xmr4el.telemetry.resources import ResourceMonitor


class Calls:
    def __init__(self):
        self.calls = []

    def start(self, key):
        self.calls.append(("start", key))

    def finish(self, key, phase, seconds):
        self.calls.append(("finish", key, phase, seconds))


def lines(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def test_event_log(tmp_path, monkeypatch):
    clock = iter(range(100))  # one tick per written line
    monkeypatch.setattr(ev_mod.time, "monotonic", lambda: next(clock))
    calls = Calls()
    log = EventLog(str(tmp_path / "events.jsonl"), calls)
    logger = logging.getLogger("test_telemetry")
    logger.addHandler(log)
    logger.propagate = False
    logger.setLevel(logging.INFO)
    try:
        for event in ({"stage": "run", "phase": "start"}, {"stage": "encode", "phase": "start"},
                      {"stage": "encode", "part": "mention_tfidf", "phase": "start"},
                      {"stage": "encode", "part": "mention_tfidf", "phase": "complete"},
                      {"stage": "encode", "phase": "complete"},
                      {"stage": "hierarchy", "phase": "start"}, {"stage": "layer", "phase": "start", "layer": 1},
                      {"stage": "node", "phase": "start", "i": 1}, {"stage": "eval", "split": "x", "phase": "tick"}):
            logger.warning("not an event")
            logger.info("e", extra={"event": event})
        log.end_open("failed", error="ValueError: boom")
    finally:
        logger.removeHandler(log)
        log.close()
    got = lines(tmp_path / "events.jsonl")
    assert all(e["v"] == 1 and datetime.fromisoformat(e["t"]).tzinfo for e in got), "versioned, offset times"
    assert [(e["stage"], e.get("part"), e["phase"]) for e in got] == [
        ("run", None, "start"), ("encode", None, "start"), ("encode", "mention_tfidf", "start"),
        ("encode", "mention_tfidf", "complete"), ("encode", None, "complete"), ("hierarchy", None, "start"),
        ("layer", None, "start"), ("node", None, "start"), ("eval", None, "tick"),
        ("hierarchy", None, "failed"), ("run", None, "failed")], "layer/node never stay open; innermost first"
    assert [e.get("duration_seconds") for e in got if e["phase"] != "start"] == [1, 3, None, 4, 10]
    assert got[-1]["error"] == "ValueError: boom" and got[-2]["error"] == "ValueError: boom"
    assert calls.calls == [("start", ("encode", None, None)), ("finish", ("encode", None, None), "complete", 3),
                           ("start", ("hierarchy", None, None)), ("finish", ("hierarchy", None, None), "failed", 4)]


def test_resource_summaries(tmp_path, monkeypatch):
    path = str(tmp_path / "resources.json")
    cpu = iter([10.0, 11.5])
    monkeypatch.setattr(resources, "cpu_seconds", lambda: next(cpu))
    mon = ResourceMonitor(path, "cpu", interval=3600)  # no thread sample during the test
    try:
        mon.start(("encode", None, None))
        mon.sample()
        mon.finish(("encode", None, None), "complete", 3.0)
        mon.finish(("labels", None, None), "complete", 1.0)  # never started: ignored
        monkeypatch.setattr(resources, "rss_now", lambda: None)  # unreadable memory: null, never invented
        monkeypatch.setattr(resources, "maxrss", lambda: 0)
        monkeypatch.setattr(resources, "cpu_seconds", lambda: 0.0)
        mon.start(("eval", "dev", None))
        mon.finish(("eval", "dev", None), "stopped", 0.0)
    finally:
        mon.close()
    doc = json.loads(open(path).read())
    assert doc["v"] == 1 and doc["device"] == "cpu" and doc["scope"]["gpu_util"] is doc["scope"]["gpu_memory"] is None
    enc, ev = doc["stages"]
    assert (enc["stage"], enc["status"], enc["samples"]) == ("encode", "completed", 3)
    r = enc["resources"]
    assert r["cpu_avg_percent"] == 50.0 and r["rss_peak_bytes"] >= r["rss_avg_bytes"] > 0
    assert r["gpu_util_avg_percent"] is None and r["torch_gpu_allocated_peak_bytes"] is None, "CPU run: no GPU numbers"
    assert (ev["stage"], ev["split"], ev["status"]) == ("eval", "dev", "stopped")
    assert ev["resources"] == {"cpu_avg_percent": None, "rss_peak_bytes": None, "rss_avg_bytes": None,
                               "gpu_util_avg_percent": None, "torch_gpu_allocated_peak_bytes": None}


def test_missing_gpu_telemetry_is_null(tmp_path, monkeypatch):
    """A CUDA device without NVML (or without CUDA at all here): utilisation null and never retried; no CUDA
    context is created to read memory."""
    tried = []
    monkeypatch.setattr(resources.torch.cuda, "utilization", lambda d: tried.append(d) or 1 / 0)
    monkeypatch.setattr(resources.torch.cuda, "is_initialized", lambda: False)
    mon = ResourceMonitor(str(tmp_path / "r.json"), "cuda", interval=3600)
    try:
        mon.start(("load", None, None))
        mon.sample()
        mon.finish(("load", None, None), "complete", 1.0)
    finally:
        mon.close()
    (stage,) = json.loads((tmp_path / "r.json").read_text())["stages"]
    assert stage["resources"]["gpu_util_avg_percent"] is None and stage["resources"]["torch_gpu_allocated_peak_bytes"] is None
    assert len(tried) == 1


def test_sampler_thread_samples_the_current_stage(tmp_path):
    mon = ResourceMonitor(str(tmp_path / "r.json"), "cpu", interval=0.01)
    try:
        mon.start(("hierarchy", None, None))
        while mon.current["n"] < 5:
            pass
        mon.finish(("hierarchy", None, None), "complete", 0.1)
    finally:
        mon.close()
    assert json.loads((tmp_path / "r.json").read_text())["stages"][0]["samples"] >= 6
    assert resources.rss_now() > 0, "resident memory readable on this platform"


def test_nvml_present_absent_and_never_initialising_cuda(tmp_path, monkeypatch):
    """With NVML (stubbed): device-wide utilisation averaged, queried by an int index (a bare cuda device makes
    torch call current_device(), which initialises CUDA). Without nvidia-ml-py (torch's own check): null. The real
    call on this machine never initialises CUDA and yields a percentage or null."""
    asked = []
    monkeypatch.setattr(resources.torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(resources.torch.cuda, "utilization", lambda d: asked.append(d) or 40 + 20 * (len(asked) % 2))
    mon = ResourceMonitor(str(tmp_path / "with.json"), "cuda", interval=3600)
    try:
        mon.start(("encode", None, None))
        mon.sample()
        mon.finish(("encode", None, None), "complete", 1.0)
    finally:
        mon.close()
    (stage,) = json.loads((tmp_path / "with.json").read_text())["stages"]
    assert stage["resources"]["gpu_util_avg_percent"] == round(sum(40 + 20 * (i % 2) for i in (1, 2, 3)) / 3, 1)
    assert asked == [0, 0, 0] and stage["resources"]["torch_gpu_allocated_peak_bytes"] is None, "CUDA unused"
    monkeypatch.undo()

    monkeypatch.setattr(resources.torch.cuda, "_HAS_PYNVML", False)  # as if nvidia-ml-py were not installed
    assert ResourceMonitor(str(tmp_path / "w.json"), "cuda", interval=3600).gpu_util() is None
    monkeypatch.undo()

    initialised = resources.torch.cuda.is_initialized()
    mon = ResourceMonitor(str(tmp_path / "real.json"), "cuda", interval=3600)
    mon.close()
    util = mon.gpu_util()
    assert util is None or 0 <= util <= 100
    assert resources.torch.cuda.is_initialized() == initialised, "the sampler never creates a CUDA context"


def test_cuda_peak_is_the_allocators_and_reset_per_stage(tmp_path, monkeypatch):
    """CUDA initialised: the stage's value is torch.cuda.max_memory_allocated after a reset at its start."""
    calls = []
    cuda = resources.torch.cuda
    monkeypatch.setattr(cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(cuda, "utilization", lambda d: 1 / 0)
    monkeypatch.setattr(cuda, "reset_peak_memory_stats", lambda d: calls.append("reset"))
    monkeypatch.setattr(cuda, "max_memory_allocated", lambda d: calls.append("peak") or 123)
    mon = ResourceMonitor(str(tmp_path / "r.json"), "cuda", interval=3600)
    try:
        mon.start(("hierarchy", None, None))
        mon.finish(("hierarchy", None, None), "complete", 1.0)
    finally:
        mon.close()
    doc = json.loads((tmp_path / "r.json").read_text())
    assert calls == ["reset", "peak"] and doc["stages"][0]["resources"]["torch_gpu_allocated_peak_bytes"] == 123
    assert "max_memory_allocated" in doc["scope"]["gpu_memory"]


def test_sample_taken_across_a_stage_change_is_dropped(tmp_path, monkeypatch):
    """A sample measured while stage A was open and stored after B started would mix A's memory into B."""
    rss, measuring, release = [10], threading.Event(), threading.Event()

    def slow_rss():
        value = rss[0]
        if threading.current_thread().name == "late":
            measuring.set()
            release.wait(5)
        return value
    monkeypatch.setattr(resources, "rss_now", slow_rss)
    mon = ResourceMonitor(str(tmp_path / "r.json"), "cpu", interval=3600)
    try:
        mon.start(("encode", None, None))
        late = threading.Thread(target=mon.sample, name="late")
        late.start()
        measuring.wait(5)
        mon.finish(("encode", None, None), "complete", 1.0)
        rss[0] = 200
        mon.start(("labels", None, None))
        release.set()
        late.join(5)
        mon.finish(("labels", None, None), "complete", 1.0)
    finally:
        mon.close()
    enc, lab = json.loads((tmp_path / "r.json").read_text())["stages"]
    assert lab["resources"]["rss_avg_bytes"] == 200 and lab["samples"] == 2, "only labels' own start/end samples"
    assert enc["resources"]["rss_avg_bytes"] == 10

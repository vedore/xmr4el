"""Run monitor: layout and resizing, pipeline stages and following, stage -> resources, live logs (the LogPanel the
fullscreen viewer uses), Cancel Run through process.stop, the run ending while the screen is open, Dashboard
navigation. Process liveness and stop are mocked; nothing is signalled."""
import asyncio
import json

import pytest

pytest.importorskip("textual")

from fake_runs import make_run, patch_json  # noqa: E402
from xmr4el_tui import process  # noqa: E402
from xmr4el_tui.app import XMR4ELApp  # noqa: E402
from xmr4el_tui.logs import Line  # noqa: E402
from xmr4el_tui.progress import stages  # noqa: E402
from xmr4el_tui.screens.dashboard_screen import DashboardScreen  # noqa: E402
from xmr4el_tui.screens.log_screen import LogScreen  # noqa: E402
from xmr4el_tui.screens.monitor_details import RUN_ROW, resource_text, summary_of  # noqa: E402
from xmr4el_tui.screens.monitor_screen import RunMonitorScreen, TelemetryRead  # noqa: E402
from xmr4el_tui.screens.stop_dialog import StopDialog  # noqa: E402
from xmr4el_tui.widgets.log_panel import LogPanel  # noqa: E402
from xmr4el_tui.widgets import log_view  # noqa: E402
from xmr4el_tui.widgets.log_view import LogView  # noqa: E402
from xmr4el_tui.widgets.monitor_panels import ResourcePanel, RunOverview  # noqa: E402
from xmr4el_tui.widgets.pipeline import Pipeline  # noqa: E402

T = "2026-10-23T14:1{}:00"
EVENTS = [{"t": T.format(0), "stage": "load", "phase": "start"},
          {"t": T.format(0), "stage": "load", "phase": "complete", "duration_seconds": 6.2},
          {"t": T.format(1), "stage": "encode", "phase": "start", "rows": 344},
          {"t": T.format(1), "stage": "encode", "part": "mention_tfidf", "phase": "start"},
          {"t": T.format(2), "stage": "encode", "part": "mention_tfidf", "phase": "complete"},
          {"t": T.format(3), "stage": "encode", "phase": "complete", "duration_seconds": 120},
          {"t": T.format(3), "stage": "labels", "phase": "start"},
          {"t": T.format(3), "stage": "labels", "phase": "complete", "labels": 91},
          {"t": T.format(4), "stage": "hierarchy", "phase": "start"},
          {"t": T.format(4), "stage": "layer", "phase": "start", "layer": 1, "layers": 2, "nodes": 1},
          {"t": T.format(4), "stage": "node", "phase": "start", "layer": 1, "i": 1, "n": 1, "rows": 344},
          {"t": T.format(5), "stage": "layer", "phase": "start", "layer": 2, "layers": 2, "nodes": 4},
          {"t": T.format(5), "stage": "node", "phase": "start", "layer": 2, "i": 3, "n": 4, "rows": 150}]
SCOPE = {"cpu": "this process (all threads) + exited children; 100 = one core", "rss": "this process",
         "gpu_util": "whole CUDA device, sampled (NVML)",
         "gpu_memory": "this process, PyTorch tensors: torch.cuda.max_memory_allocated, exact"}


def summary(stage, seconds, cpu, rss_peak, util=None, gpu=None, split=None):
    return {"stage": stage, **({"split": split} if split else {}), "status": "completed",
            "duration_seconds": seconds, "samples": 4,
            "resources": {"cpu_avg_percent": cpu, "rss_peak_bytes": rss_peak, "rss_avg_bytes": rss_peak // 2,
                          "gpu_util_avg_percent": util, "torch_gpu_allocated_peak_bytes": gpu}}


RESOURCES = {"v": 1, "device": "cuda", "interval_seconds": 5, "scope": SCOPE,
             "stages": [summary("load", 6.2, 100.0, 2 << 30), summary("encode", 120, 400.0, 8 << 30, 81.5, 3 << 30)]}
LIVE = "2026-10-23_14-10-00_live"


@pytest.fixture
def live(tmp_path, monkeypatch):
    """A recorded-running train run the TUI may stop (liveness mocked), plus a completed one; process.stop is
    recorded, never signals."""
    run = make_run(tmp_path, "live", stamp="2026-10-23_14-10-00", status="running", tree_id=None,
                   splits=("dev",), metrics={"dev": None}, events=EVENTS, identity={"pid": 4242, "pgid": 4242})
    (run / "resources.json").write_text(json.dumps(RESOURCES))
    make_run(tmp_path, "done", splits=("dev",))
    monkeypatch.setattr("xmr4el_tui.process.derived_state",
                        lambda r, here: process.Live("RUNNING", stoppable=True))
    stops = []
    monkeypatch.setattr("xmr4el_tui.process.stop", lambda r, force=False: stops.append((r.id, force)) or
                        f"SIGTERM sent to pid {r.identity['pid']}")
    return run, stops


async def settle(app, pilot):
    await app.workers.wait_for_complete()
    await pilot.pause()
    await app.workers.wait_for_complete()
    await pilot.pause()


def run_monitor(root, test, run_id=LIVE, split=None, size=(140, 40)):
    async def go():
        app = XMR4ELApp(root)
        async with app.run_test(size=size) as pilot:
            await settle(app, pilot)
            app.screen.open_run(run_id, split)
            await settle(app, pilot)
            assert isinstance(app.screen, RunMonitorScreen) and app.screen.run.id == run_id
            await test(app, pilot, app.screen)
    asyncio.run(go())


def text(widget):
    return str(widget.render())


def test_log_update_before_mount_uses_latest_run(live):
    run, _ = live

    async def check(app, pilot, screen):
        panel = LogPanel(screen.run, app.root)
        panel.set_run(screen.run)  # scan can land between mount() and composition
        await screen.mount(panel)
        await pilot.pause()
        assert panel.query_one(LogView) is not None
        await panel.remove()

    run_monitor(run.parents[2], check)


def resources_text(screen):
    return text(screen.query_one("#resources-text"))


async def rescan(app, pilot):
    app.apply_scan(app.repo.snapshot(fresh=True))
    await settle(app, pilot)


def test_layout_resize_and_small_terminals(live):
    run, _ = live

    async def check(app, pilot, screen):
        panels = screen.panels()
        assert [p.id for p in panels] == ["overview", "pipeline", "resources", "logs"]
        overview, pipeline, resources, logs = (p.region for p in panels)
        assert overview.x == pipeline.x == 0 and resources.x == logs.x > 0, "left: overview, pipeline; right"
        assert overview.width == round(140 * 0.35) and overview.bottom == pipeline.y < pipeline.bottom
        assert overview.height < 14 < pipeline.height, "the overview is content-sized, the pipeline fills"
        right = resources.height + logs.height
        assert resources.y < logs.y and abs(resources.height - 0.35 * right) <= 1
        assert isinstance(logs and screen.query_one("#logs"), LogPanel)
        await pilot.press("greater_than_sign", "greater_than_sign")
        assert screen.query_one("#mon-left").region.width == 49 + 4
        await pilot.resize_terminal(100, 30)
        await pilot.pause()
        assert screen.query_one("#mon-left").region.width == 53, "the chosen width is kept"
        screen.set_left_width(999)  # mouse drag: clamped so the right side keeps RIGHT_MIN
        await pilot.pause()
        assert screen.query_one("#mon-left").region.width == 56 and screen.maximized is None
        await pilot.resize_terminal(70, 24)  # too small for two columns: one panel at a time
        await pilot.pause()
        assert screen.single and screen.maximized is screen.query_one(Pipeline) and app.focused.id == "pipeline"
        await pilot.press("ctrl+tab")
        assert screen.maximized is screen.query_one(ResourcePanel) and app.focused.id == "resources"
        await pilot.press("escape")  # esc goes back, there is no two-column layout to restore
        await pilot.pause()
        assert isinstance(app.screen, DashboardScreen)
    run_monitor(run.parent.parent.parent, check)


def test_pipeline_follows_and_selection_drives_resources(live):
    run, _ = live

    async def check(app, pilot, screen):
        pipe = screen.query_one(Pipeline)
        assert pipe.selected_key() == ("hierarchy",) and pipe.following, "opens on the current stage"
        assert pipe.rows[("hierarchy",)].is_expanded, "its layers are shown"
        rows = [str(n.label) for n in (pipe.rows[k] for k in (("hierarchy", "layer 1"), ("hierarchy", "layer 2")))]
        assert "1 nodes" in rows[0] and "node 3/4 · 150 rows" in rows[1]
        assert screen.query_one(ResourcePanel).border_title == "Resources — Hierarchy training"
        assert "Collecting resource statistics" in resources_text(screen)
        await pilot.press("up", "up")  # Label embeddings (no summary), Encode features
        assert pipe.selected_key() == ("encode",) and not pipe.following, "the user moved: stop following"
        body = resources_text(screen)
        assert screen.query_one(ResourcePanel).border_title == "Resources — Encode features"
        assert "400.0 %" in body and "8.0 GiB" in body and "81.5 %" in body and "3.0 GiB" in body
        assert "whole CUDA device" in body and "this process, PyTorch tensors" in body, "scopes from resources.json"
        await pilot.press("down")
        assert "No summary recorded for this stage." in resources_text(screen)
        await pilot.press("up", "up")
        assert pipe.selected_key() == ("load",) and "N/A" in resources_text(screen), "no GPU figures: N/A, not 0"

        # the hierarchy ends and save starts: the user's row is kept, the new summary appears when selected
        with open(run / "events.jsonl", "a") as f:
            for ev in ({"t": T.format(6), "stage": "hierarchy", "phase": "complete", "duration_seconds": 90},
                       {"t": T.format(6), "stage": "save", "phase": "start"}):
                f.write(json.dumps(ev) + "\n")
        patch_json(run / "resources.json", stages=RESOURCES["stages"] + [summary("hierarchy", 90, 150.0, 4 << 30)])
        screen.tick()
        await settle(app, pilot)
        assert pipe.selected_key() == ("load",), "selection kept across a refresh"
        await pilot.press("c")  # follow again: the current stage
        assert pipe.selected_key() == ("save",) and pipe.following
        await pilot.press("up")  # the hierarchy's last layer (expanded while it ran)
        assert pipe.selected_key() == ("hierarchy", "layer 2")
        assert "Parts are not sampled separately" in resources_text(screen) and "150.0 %" in resources_text(screen)
        await pilot.press("up", "up")
        assert pipe.selected_key() == ("hierarchy",) and "Completed  1:30" in resources_text(screen)
        await pilot.press("home")
        assert pipe.selected_key() == RUN_ROW
        body = resources_text(screen)
        assert "3 ended stages" in body and "11.0 GiB" not in body and "8.0 GiB" in body, "peaks: the highest"
    run_monitor(run.parent.parent.parent, check)


def test_resource_texts():
    """Repeated names: the eval summary of the right split; overall CPU = CPU time / wall time, exact."""
    doc = {**RESOURCES, "stages": [summary("eval", 10, 100.0, 1, split="dev"), summary("eval", 30, 300.0, 5,
                                                                                         split="test")]}
    assert summary_of(doc, "eval:test")["resources"]["cpu_avg_percent"] == 300.0
    assert summary_of(doc, "eval") is None and summary_of(None, "load") is None
    run = type("R", (), {"kind": "eval", "recorded": "completed", "live": None, "finished": None, "started": None})
    _, t = resource_text(RUN_ROW, run, [], doc)
    assert "250.0 %" in t.plain, "(100*10 + 300*30) / 40, not the mean of the averages"
    evs = [{"t": T.format(0), "stage": "eval", "split": "test", "phase": "start"}]
    title, t = resource_text(("eval:test",), run, stages(evs, "eval", ["dev", "test"], "running"), None)
    assert title == "Resources — Evaluate test" and "Collecting resource statistics" in t.plain
    title, t = resource_text(("eval:dev",), run, stages(evs, "eval", ["dev", "test"], "running"), None)
    assert "Not started." in t.plain
    evs.append({"t": T.format(1), "stage": "eval", "split": "test", "phase": "complete"})
    _, t = resource_text(("eval:test",), run, stages(evs, "eval", ["dev", "test"], "completed"), None)
    assert "No resources.json for this run" in t.plain


def test_live_logs_follow_search_fullscreen_and_run_end(live):
    run, _ = live
    (run / "run.log").write_text("".join(f"2026-10-23 14:10:{i:02d},000 [{'WARNING' if i == 3 else 'INFO'}] x: "
                                         f"line {i}\n" for i in range(10)))

    async def check(app, pilot, screen):
        panel = screen.query_one(LogPanel)
        view = panel.query_one(LogView)
        assert len(view.all) == 10 and panel.timer is not None, "the same reader as the fullscreen viewer"
        with open(run / "run.log", "a") as f:
            f.write("2026-10-23 14:11:00,000 [ERROR] x: boom\nTraceback (most recent call last)\n")
        panel.tick()
        await settle(app, pilot)
        assert len(view.all) == 12 and view.all[-1].level == "ERROR", "appended; the traceback keeps its level"
        screen.focus_panel(panel)
        await pilot.press("slash", *"line 3", "enter", "l", "l")
        assert view.min_level == "WARNING" and len(view.shown) == 3 and view.matches == [0], "line 3 is a WARNING"
        await pilot.press("f")
        assert screen.maximized is panel and app.focused is view, "fullscreen: the same panel, search and filter"
        assert view.min_level == "WARNING" and view.query == "line 3"
        await pilot.press("escape")
        assert screen.maximized is None and isinstance(app.screen, RunMonitorScreen)
        patch_json(run / "status.json", status="completed", finished="2026-10-23T14:20:00")
        await rescan(app, pilot)
        panel.tick()  # the final read, then the log is no longer polled
        await settle(app, pilot)
        assert panel.timer is None and screen.timer is None and len(view.all) == 12
        assert "COMPLETED" in text(screen.query_one("#overview-text"))
        assert isinstance(app.screen, RunMonitorScreen), "stays on the run"
    run_monitor(run.parent.parent.parent, check)


def test_cancel_run(live):
    run, stops = live

    async def check(app, pilot, screen):
        button = screen.query_one("#cancel")
        assert not button.disabled
        await pilot.press("s")
        dialog = app.screen
        assert isinstance(dialog, StopDialog) and LIVE in text(dialog.query_one("#stop-text"))
        await pilot.press("enter")  # focus starts on Keep running
        assert app.screen is screen and stops == []
        await pilot.press("s")
        await pilot.press("tab", "enter")  # Cancel run
        await settle(app, pilot)
        assert stops == [(LIVE, False)], "graceful stop of this run, through process.stop"
        assert button.disabled and "cancellation requested" in text(screen.query_one("#overview-text"))
        await pilot.press("s")
        assert app.screen is screen and stops == [(LIVE, False)], "no second request"
        patch_json(run / "status.json", status="stopped", error="SIGTERM", finished="2026-10-23T14:20:00")
        await rescan(app, pilot)
        assert "STOPPED" in text(screen.query_one("#overview-text")) and button.disabled
        assert "Stopped by SIGTERM" in text(screen.query_one("#overview-banner"))
        assert [(n.title, n.text) for n in app.notices.items] == [("Run cancelled", "Stopped on request.")], \
            "the stop this TUI sent"
    run_monitor(run.parent.parent.parent, check)


def test_cancel_races_and_finished_runs(live, monkeypatch):
    run, stops = live

    async def check(app, pilot, screen):
        await pilot.press("s")
        assert isinstance(app.screen, StopDialog)
        patch_json(run / "status.json", status="completed", finished="2026-10-23T14:20:00")
        await rescan(app, pilot)  # it completed while the dialog was open
        assert app.screen is screen and stops == [], "dialog closed, nothing sent"
        assert screen.query_one("#cancel").disabled and not screen.check_action("cancel", ())
    run_monitor(run.parent.parent.parent, check)

    async def refused(app, pilot, screen):
        monkeypatch.setattr("xmr4el_tui.process.stop", lambda r, force=False: "not signalled: process exited")
        await pilot.press("s", "tab", "enter")
        await settle(app, pilot)
        assert not screen.cancelling and not screen.query_one("#cancel").disabled, "refused: may be retried"
        assert not app.notices.requested, "nothing sent: a later stop is not this TUI's"
    patch_json(run / "status.json", status="running", finished=None)
    run_monitor(run.parent.parent.parent, refused)

    async def completed(app, pilot, screen):
        assert screen.query_one("#cancel").disabled and screen.timer is None
        await pilot.press("s")
        assert app.screen is screen
        assert screen.query_one(Pipeline).selected_key() == RUN_ROW
    run_monitor(run.parent.parent.parent, completed, run_id="2026-10-20_13-36-20_done")


def test_failed_and_lost_runs(tmp_path, monkeypatch):
    failed = EVENTS[:9] + [{"t": T.format(5), "stage": "hierarchy", "phase": "failed", "error": "ValueError: boom"}]
    make_run(tmp_path, "bad", stamp="2026-10-23_14-10-00", status="failed", tree_id=None, metrics={"dev": None},
             events=failed)
    make_run(tmp_path, "gone", stamp="2026-10-23_15-10-00", status="running", tree_id=None, metrics={"dev": None},
             events=EVENTS, identity={"pid": 4242})
    monkeypatch.setattr("xmr4el_tui.process.derived_state", lambda r, here: process.Live("LOST", "process 4242 is "
                                                                                         "gone"))

    async def bad(app, pilot, screen):
        pipe = screen.query_one(Pipeline)
        assert pipe.selected_key() == ("hierarchy",) and "failed" in str(pipe.rows[("hierarchy",)].label)
        assert "ValueError: boom" in resources_text(screen) and "ValueError: boom" in text(
            screen.query_one("#overview-banner"))
        assert screen.query_one("#cancel").disabled
    run_monitor(tmp_path, bad, run_id="2026-10-23_14-10-00_bad")

    async def gone(app, pilot, screen):
        assert "LOST" in text(screen.query_one("#overview-text")) and screen.query_one("#cancel").disabled
        assert "Process 4242 is gone" in text(screen.query_one("#overview-banner"))
        assert "failed" in str(screen.query_one(Pipeline).rows[("hierarchy",)].label), "as the Dashboard shows it"
    run_monitor(tmp_path, gone, run_id="2026-10-23_15-10-00_gone")


def test_dashboard_navigation_and_focus(live):
    run, _ = live

    async def check(app, pilot):
        await settle(app, pilot)
        dash = app.screen
        await pilot.press("ctrl+tab", "ctrl+tab")
        assert app.focused.id == "active-runs"
        await pilot.press("enter")
        await settle(app, pilot)
        screen = app.screen
        assert isinstance(screen, RunMonitorScreen) and screen.run.id == LIVE
        assert app.focused.id == "pipeline"
        order = []
        for _ in range(4):
            await pilot.press("ctrl+tab")
            order.append(screen.current_panel().id)
        assert order == ["resources", "logs", "overview", "pipeline"] and app.focused.id == "pipeline"
        await pilot.press("ctrl+shift+tab")
        assert app.focused.id == "cancel", "the overview's control"
        await pilot.press("escape")
        await pilot.pause()
        assert app.screen is dash and app.focused.id == "active-runs", "the Dashboard as it was left"
        done = run.parent / "2026-10-20_13-36-20_done"
        (done / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in EVENTS[:2] + [
            {"t": T.format(1), "stage": "eval", "split": "dev", "phase": "start"},
            {"t": T.format(2), "stage": "eval", "split": "dev", "phase": "complete", "acc@1": 0.9}]))
        dash.open_run(done.name, "dev")  # an evaluation: its stage is selected
        await settle(app, pilot)
        assert app.screen.query_one(Pipeline).selected_key() == ("eval:dev",)
        assert "acc@1 0.9000" in str(app.screen.query_one(Pipeline).rows[("eval:dev",)].label)
        app.screen.dismiss()
        await pilot.pause()
        app.push_screen(LogScreen(app.snapshot.runs[0], app.root))  # the Dashboard's viewer: same LogPanel
        await settle(app, pilot)
        assert isinstance(app.screen.query_one(LogPanel), LogPanel) and app.screen.query_one(LogPanel).named

    async def go():
        app = XMR4ELApp(run.parent.parent.parent)
        async with app.run_test(size=(140, 40)) as pilot:
            await check(app, pilot)
    asyncio.run(go())


def test_overview_is_compact_on_long_names(tmp_path, monkeypatch):
    name = "a_very_long_run_name_" * 6
    make_run(tmp_path, name, stamp="2026-10-23_14-10-00", status="running", tree_id=None, metrics={"dev": None},
             events=EVENTS[:2], identity={"pid": 4242})
    monkeypatch.setattr("xmr4el_tui.process.derived_state", lambda r, here: process.Live("RUNNING", stoppable=True))

    async def check(app, pilot, screen):
        assert screen.query_one("#overview-text").region.height == 7, "one line per field, cut, never wrapped"
        assert screen.query_one(RunOverview).region.height <= 12
    run_monitor(tmp_path, check, run_id=f"2026-10-23_14-10-00_{name}")


def test_final_events_read_after_a_read_in_flight(live):
    """The run ends while a telemetry read is in flight: that older read cannot stop polling; one more read lands
    the final events (the runner writes them before its final status)."""
    run, _ = live

    async def check(app, pilot, screen):
        old = screen.events
        screen.reading = True  # a read started while the run was running, not landed yet
        with open(run / "events.jsonl", "a") as f:
            for e in ({"t": T.format(6), "stage": "hierarchy", "phase": "complete"},
                      {"t": T.format(6), "stage": "save", "phase": "start"},
                      {"t": T.format(6), "stage": "save", "phase": "complete"},
                      {"t": T.format(7), "stage": "eval", "split": "dev", "phase": "start"},
                      {"t": T.format(7), "stage": "eval", "split": "dev", "phase": "complete"}):
                f.write(json.dumps(e) + "\n")
        patch_json(run / "status.json", status="completed", finished="2026-10-23T14:20:00")
        await rescan(app, pilot)
        assert screen.again and screen.timer is not None
        screen.post_message(TelemetryRead(old, screen.resources, screen.device, final=False))
        await settle(app, pilot)
        assert screen.timer is None and not screen.reading
        assert [s.state for s in screen.stages] == ["done"] * 6
    run_monitor(run.parent.parent.parent, check)


def test_malformed_resources_and_truncated_events(live):
    """resources.json with a non-list `stages` and non-string scopes shows no summary; a truncated events.jsonl
    leaves no stale Pipeline rows."""
    run, _ = live
    (run / "resources.json").write_text(json.dumps({"stages": 7, "scope": {"cpu": 5}}))

    async def check(app, pilot, screen):
        pipe = screen.query_one(Pipeline)
        pipe.move_cursor(pipe.rows[RUN_ROW])
        await pilot.pause()
        assert "No stage has ended yet" in resources_text(screen)
        pipe.move_cursor(pipe.rows[("encode",)])
        await pilot.pause()
        assert "No summary recorded" in resources_text(screen)
        (run / "events.jsonl").write_text("")
        screen.tick()
        await settle(app, pilot)
        assert list(pipe.rows) == [RUN_ROW] and pipe.selected_key() == RUN_ROW
        assert len(pipe.root.children) == 1
    run_monitor(run.parent.parent.parent, check)


def test_log_retention_keeps_a_scrolled_back_view(live, monkeypatch):
    """At the line cap, dropping the oldest lines keeps the text a scrolled-back reader is looking at and the
    current search match."""
    run, _ = live
    monkeypatch.setattr(log_view, "MAX_LINES", 200)
    (run / "run.log").write_text("".join(f"row {i}\n" for i in range(200)))

    async def check(app, pilot, screen):
        view = screen.query_one(LogView)
        await settle(app, pilot)
        view.search("row 120")  # scrolls back: the match mid-view
        await pilot.pause()
        top = view.all[view.shown[view.scroll_offset.y]].text
        assert top == f"row {120 - view.size.height // 2}"
        view.append([Line(f"row {i}", None) for i in range(200, 220)])
        await pilot.pause()
        assert len(view.all) == 200 and view.all[view.shown[view.scroll_offset.y]].text == top
        assert view.all[view.shown[view.matches[view.current]]].text == "row 120"
    run_monitor(run.parent.parent.parent, check)


def test_long_error_keeps_the_pipeline_on_screen(tmp_path):
    error = "ValueError: " + "x" * 2000 + "\nTraceback line\n" * 20
    path = make_run(tmp_path, "bad", stamp="2026-10-23_14-10-00", status="failed", tree_id=None,
                    metrics={"dev": None}, events=EVENTS[:9])
    patch_json(path / "status.json", error=error)
    for size in ((140, 40), (100, 30)):
        async def check(app, pilot, screen):
            overview, pipeline = screen.query_one(RunOverview).region, screen.query_one(Pipeline).region
            assert overview.height <= screen.query_one("#mon-left").region.height // 2
            assert pipeline.y == overview.bottom and pipeline.bottom <= app.size.height - 1 and pipeline.height > 3
        run_monitor(tmp_path, check, run_id=path.name, size=size)

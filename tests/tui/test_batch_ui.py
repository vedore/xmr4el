"""Batch execution in the TUI (tui.md §5.16): Review Batch launch, batch monitor, Stop Batch, Dashboard discovery, the
quit dialog and batch notifications. A fake runner stands in for the real one; nothing is trained."""
import copy
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from test_batch import FAKE, wait_for  # noqa: E402
from test_review_batch import Queue, footer, make_root, run_review, settle  # noqa: E402

from textual.widgets import Button, Input  # noqa: E402

from xmr4el_tui import batch  # noqa: E402
from xmr4el_tui.batch import BatchError  # noqa: E402
from xmr4el_tui.screens.batch_monitor import BatchesScreen, BatchMonitorScreen  # noqa: E402
from xmr4el_tui.screens.dataset_setup import DatasetSetupScreen  # noqa: E402
from xmr4el_tui.screens.model_builder import PipelineGrid  # noqa: E402
from xmr4el_tui.services.notifications import Notice  # noqa: E402
from xmr4el_tui.screens.dashboard_screen import ActiveRuns, DashboardScreen  # noqa: E402
from xmr4el_tui.screens.review_batch import ReviewBatchScreen  # noqa: E402
from xmr4el_tui.screens.stop_dialog import QuitDialog, StopBatchDialog  # noqa: E402
from xmr4el_tui.services.notifications import Notifications  # noqa: E402


def with_runner(root, monkeypatch, seconds=None):
    """Batches launched by the TUI use the fake runner (the seam is the coordinator's environment); `seconds`: how
    long each fake run takes."""
    (root / "fake_runner.py").write_text(FAKE)
    if seconds:
        (root / "fake_ctl.json").write_text(json.dumps({"seconds": seconds}))
    real, calls = batch.launch, []

    def launch(r, plan_, runner=None):
        calls.append(plan_)
        return real(r, plan_, root / "fake_runner.py")
    monkeypatch.setattr(batch, "launch", launch)
    return calls


def stop_all(root):
    """Leave nothing running: stop active batches and wait for their coordinators."""
    for info in batch.scan(root):
        if info.active:
            batch.request_stop(root, info.id)
    assert wait_for(lambda: not any(i.active for i in batch.scan(root)), 40)


async def until(pilot, cond, seconds=15):
    """Wait for `cond` while the app keeps running (a blocking sleep would starve its workers and messages)."""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if cond():
            return True
        await pilot.pause(0.1)
    return False


async def launch(app, pilot):
    await pilot.press("l")
    assert await until(pilot, lambda: isinstance(app.screen, BatchMonitorScreen))
    await pilot.pause()
    return app.screen


def test_launch_starts_a_real_batch_and_opens_its_monitor(tmp_path, monkeypatch):
    root = make_root(tmp_path)
    calls = with_runner(root, monkeypatch)

    async def go(app, pilot, screen):
        assert isinstance(screen, ReviewBatchScreen)
        before = [(t.id, t.name, t.config) for t in app.draft.trees]
        await pilot.press("l")
        await pilot.press("l")  # a repeated press while preparing must not launch twice
        assert await until(pilot, lambda: isinstance(app.screen, BatchMonitorScreen))
        await settle(app, pilot)
        assert len(calls) == 1 and len(list((root / "outputs/batches").glob("2*"))) == 1
        assert [(t.id, t.name, t.config) for t in app.draft.trees] == before  # the draft is untouched
        monitor = app.screen
        assert [len(str(o.prompt).split("\n")) for o in monitor.query_one(Queue).options] == [2, 2]
        assert len(app.coordinators) == 1
        assert await until(pilot, lambda: batch.scan(root)[0].state == "completed")
        assert await until(pilot, lambda: monitor.info.state == "completed")
        assert "COMPLETED" in str(monitor.query_one("#bm-overview").render())
        assert monitor.info.count("succeeded") == 2
    run_review(root, go)


def test_launch_is_reachable_by_tab_and_the_monitor_esc_goes_to_the_dashboard(tmp_path, monkeypatch):
    root = make_root(tmp_path)
    with_runner(root, monkeypatch, 60)

    async def go(app, pilot, screen):
        assert app.focused is screen.query_one(Queue) or app.focused.id == "rv-queue"
        await pilot.press("tab")
        assert app.focused.id == "rv-launch"
        await pilot.press("enter")
        assert await until(pilot, lambda: isinstance(app.screen, BatchMonitorScreen))
        batch_id = app.screen.batch_id
        await pilot.press("escape")
        await pilot.pause()
        assert isinstance(app.screen, DashboardScreen)
        assert not any(isinstance(s, (ReviewBatchScreen, BatchMonitorScreen)) for s in app.screen_stack)
        assert batch.info(batch.directory(root, batch_id)).state in ("running", "preparing")  # still running
    run_review(root, go)


def stack_names(app):
    return [type(s).__name__ for s in app.screen_stack]


def test_monitor_esc_paths_reopen_and_the_batch_keeps_running(tmp_path, monkeypatch):
    root = make_root(tmp_path)
    with_runner(root, monkeypatch, 60)

    async def go(app, pilot, screen):
        monitor = await launch(app, pilot)  # A: Review Batch -> launch -> monitor
        batch_id = monitor.batch_id
        await pilot.press("escape")
        await pilot.pause()
        assert isinstance(app.screen, DashboardScreen) and app.asking is None
        assert not {"ReviewBatchScreen", "BatchMonitorScreen", "ModelBuilderScreen"} & set(stack_names(app))
        assert batch.info(batch.directory(root, batch_id)).state in ("running", "preparing")
        app.push_screen(BatchMonitorScreen(batch_id))  # B: an existing batch from the Dashboard
        await pilot.pause()
        assert isinstance(app.screen, BatchMonitorScreen) and app.screen.info.state in ("running", "preparing")
        await pilot.press("escape")
        await pilot.pause()
        assert isinstance(app.screen, DashboardScreen) and "ReviewBatchScreen" not in stack_names(app)
        assert batch.info(batch.directory(root, batch_id)).state in ("running", "preparing")
    run_review(root, go)


def test_unapplied_edit_survives_monitor_esc_and_resume_restores_the_hierarchy(tmp_path, monkeypatch):
    root = make_root(tmp_path)
    with_runner(root, monkeypatch, 60)

    async def go(app, pilot, screen):
        batch_id = (await launch(app, pilot)).batch_id
        await pilot.press("escape")  # Review Batch is closed; the batch keeps running
        await pilot.pause()
        app.draft.resume = None
        app.open_setup()  # the normal workflow: Dataset Setup > Model Builder
        await pilot.pause()
        app.open_builder()
        await pilot.pause()
        builder, tree = app.screen, app.draft.trees[0]
        before = copy.deepcopy(tree.config)
        builder.select_slot("matcher")
        builder.query_one(PipelineGrid).focus()
        await pilot.press("enter", "enter")  # a real component, a real attribute
        i = next(i for i, f in enumerate(builder.editing.fields) if f.path[-1] == "C")
        box = builder.query_one(f"#mbf-{i}", Input)
        box.focus()
        box.value = "2"
        await pilot.pause()
        assert builder.dirty() and tree.config == before
        app.open_notice(Notice(("x",), "information", "t", "s", "x", datetime.now(), ("batch", batch_id, None)))
        await pilot.pause()
        assert isinstance(app.screen, BatchMonitorScreen)
        await pilot.press("escape")
        await pilot.pause()
        assert isinstance(app.screen, DashboardScreen) and app.asking is None  # no discard prompt
        assert [n for n in stack_names(app) if n != "Screen"] == ["DashboardScreen"]
        assert tree.config == before and [k for k, _ in app.draft.resume] == ["setup", "builder"]
        assert app.draft.confirmed is not None
        assert app.screen.check_action("app.resume_experiment", ()) and "resume experiment" in str(
            [b.description for b in app.screen.BINDINGS if b.key == "c"])
        assert batch.info(batch.directory(root, batch_id)).state in ("running", "preparing")
        await pilot.press("c")  # Resume Experiment: the screens are rebuilt over the Dashboard
        await pilot.pause()
        await pilot.pause()
        assert [n for n in stack_names(app) if n != "Screen"] == ["DashboardScreen", "DatasetSetupScreen",
                                                                   "ModelBuilderScreen"]
        builder = app.screen
        assert builder.dirty() and builder.editing.comp.id == "jointlogisticregression" and app.draft.slot == "matcher"
        assert builder.editing.draft["matcher_config"]["kwargs"]["C"] == 2.0 and tree.config == before
        await pilot.press("escape")  # leaving the editor itself still asks
        assert "Discard unapplied changes to Matcher?" in footer(app) and builder.editing is not None
        await pilot.press("u")
        assert builder.editing is None and tree.config == before
        for _ in range(8):  # library > grid > Tree Selection > Dataset Setup, never straight to the Dashboard
            if isinstance(app.screen, DatasetSetupScreen):
                break
            await pilot.press("escape")
            await pilot.pause()
        assert isinstance(app.screen, DatasetSetupScreen) and app.draft.confirmed is not None
        assert batch.info(batch.directory(root, batch_id)).state in ("running", "preparing")
    run_review(root, go, through_builder=False)


def test_resume_restores_review_batch_as_the_parent(tmp_path, monkeypatch):
    root = make_root(tmp_path)
    with_runner(root, monkeypatch, 60)

    async def go(app, pilot, screen):
        batch_id = (await launch(app, pilot)).batch_id
        await pilot.press("escape")
        await pilot.pause()
        assert [k for k, _ in app.draft.resume] == ["review"]
        await pilot.press("c")
        await pilot.pause()
        await pilot.pause()
        assert isinstance(app.screen, ReviewBatchScreen)
        assert [n for n in stack_names(app) if n != "Screen"] == ["DashboardScreen", "ReviewBatchScreen"]
        assert batch.info(batch.directory(root, batch_id)).state in ("running", "preparing")
    run_review(root, go, through_builder=False)


def test_dashboard_monitor_cycles_keep_one_dashboard(tmp_path, monkeypatch):
    root = make_root(tmp_path)
    with_runner(root, monkeypatch, 60)

    async def go(app, pilot, screen):
        batch_id = (await launch(app, pilot)).batch_id
        await pilot.press("escape")
        await pilot.pause()
        for _ in range(3):
            app.push_screen(BatchMonitorScreen(batch_id))  # from the Dashboard
            await pilot.pause()
            assert isinstance(app.screen, BatchMonitorScreen)
            await pilot.press("escape")
            await pilot.pause()
            assert isinstance(app.screen, DashboardScreen) and app.asking is None
            assert [n for n in stack_names(app) if n != "Screen"] == ["DashboardScreen"]
        assert batch.info(batch.directory(root, batch_id)).state in ("running", "preparing")
    run_review(root, go)


def test_a_failed_launch_keeps_the_review_screen_and_the_draft(tmp_path, monkeypatch):
    root = make_root(tmp_path)

    def refuse(r, plan_):
        raise BatchError("The coordinator did not start: boom")
    monkeypatch.setattr(batch, "launch", refuse)

    async def go(app, pilot, screen):
        trees = [t.id for t in app.draft.trees]
        await pilot.press("l")
        assert await until(pilot, lambda: "Batch not started" in footer(app), 10)
        assert app.screen is screen and "The coordinator did not start: boom" in footer(app)
        assert [t.id for t in app.draft.trees] == trees and not screen.launching
        assert not screen.query_one("#rv-launch").disabled and not app.coordinators
        assert not (root / "outputs/batches").exists()
    run_review(root, go)


def test_a_second_batch_is_refused_while_one_runs(tmp_path, monkeypatch):
    root = make_root(tmp_path)
    with_runner(root, monkeypatch, 30)

    async def go(app, pilot, screen):
        await launch(app, pilot)
        app.go_to(screen, 0)
        await pilot.pause()
        app.draft.id = "other"
        await pilot.press("l")
        assert await until(pilot, lambda: "Another batch is running" in footer(app), 10)
        assert app.screen is screen and len(list((root / "outputs/batches").glob("2*"))) == 1
        stop_all(root)
    run_review(root, go)


def test_stop_batch_from_the_monitor(tmp_path, monkeypatch):
    root = make_root(tmp_path)
    with_runner(root, monkeypatch, 60)

    async def go(app, pilot, screen):
        monitor = await launch(app, pilot)
        assert await until(pilot, lambda: monitor.info.current is not None and monitor.info.current["pid"])
        stop_file = root / "outputs/batches" / monitor.batch_id / "stop"
        await pilot.press("s")
        assert isinstance(app.screen, StopBatchDialog) and app.focused.id == "keep"  # the safe default
        await pilot.press("escape")
        assert not stop_file.exists()
        await pilot.press("s")
        app.screen.query_one("#confirm", Button).press()
        assert await until(pilot, stop_file.exists, 5) and await until(pilot, lambda: "stop requested" in footer(app), 5)
        assert await until(pilot, lambda: batch.info(monitor.info.path).state == "stopped", 40)
        assert await until(pilot, lambda: monitor.info.state == "stopped", 5)
        assert monitor.info.count("cancelled") == 2
        assert "STOPPED" in str(monitor.query_one("#bm-overview").render())
    run_review(root, go)


def test_quit_with_an_active_batch_offers_three_choices(tmp_path, monkeypatch):
    root = make_root(tmp_path)
    with_runner(root, monkeypatch, 60)

    async def go(app, pilot, screen):
        monitor = await launch(app, pilot)
        stop_file = root / "outputs/batches" / monitor.batch_id / "stop"
        exits = []
        app.exit = lambda *a, **k: exits.append(1)
        await pilot.press("ctrl+c")
        assert isinstance(app.screen, QuitDialog) and app.focused.id == "stay"  # Stay is the default
        assert [b.id for b in app.screen.query(Button)] == ["leave", "stop", "stay"]
        await pilot.press("ctrl+c")  # asked again over the dialog: the safe answer
        assert app.screen is monitor and not exits
        await pilot.press("ctrl+c")
        await pilot.press("escape")
        assert app.screen is monitor and not exits and not stop_file.exists()
        await pilot.press("ctrl+c")
        app.screen.query_one("#leave", Button).press()
        await pilot.pause()
        assert exits == [1] and not stop_file.exists()
        assert batch.info(stop_file.parent).state == "running"  # leaving changed nothing
        await pilot.press("ctrl+c")
        app.screen.query_one("#stop", Button).press()
        assert await until(pilot, lambda: len(exits) == 2, 10)
        assert stop_file.exists()
        stop_all(root)
    run_review(root, go)


def test_quit_without_a_batch_keeps_the_footer_question(tmp_path):
    async def go(app, pilot, screen):
        await pilot.press("ctrl+c")
        assert not isinstance(app.screen, QuitDialog) and app.confirming is app.screen
    run_review(make_root(tmp_path), go)


def test_dashboard_lists_the_active_batch_and_the_batch_list_finds_it(tmp_path, monkeypatch):
    root = make_root(tmp_path)
    with_runner(root, monkeypatch, 60)

    async def go(app, pilot, screen):
        monitor = await launch(app, pilot)
        dash = next(s for s in app.screen_stack if isinstance(s, DashboardScreen))
        app.go_to(dash, 0)
        app.rescan()
        await settle(app, pilot)
        rows = [k.value for k in dash.query_one(ActiveRuns).rows]
        assert f"batch:{monitor.batch_id}" in rows
        await pilot.press("b")
        assert isinstance(app.screen, BatchesScreen)
        assert monitor.batch_id in [o.id for o in app.screen.query_one("#bm-list").options]
        await pilot.press("enter")
        assert isinstance(app.screen, BatchMonitorScreen) and app.screen.batch_id == monitor.batch_id
        stop_all(root)
    run_review(root, go)


def test_batch_notifications_report_transitions_once():
    def info(state, trees):
        status = {"trees": [{"id": k, "name": k.upper(), "state": v, "error": None} for k, v in trees.items()],
                  "error": None, "name": "Batch"}
        return batch.Info("b1", Path("b1"), "Batch", state, status)

    notes, at = Notifications(), datetime.now()
    assert notes.observe_batches([info("running", {"t1": "running", "t2": "pending"})], at) == []  # first scan: silent
    assert notes.observe_batches([info("running", {"t1": "running", "t2": "pending"})], at) == []
    out = notes.observe_batches([info("running", {"t1": "succeeded", "t2": "running"})], at)
    assert [n.title for n in out] == ["Tree completed", "Tree started"]
    assert notes.observe_batches([info("running", {"t1": "succeeded", "t2": "running"})], at) == []
    out = notes.observe_batches([info("completed_with_failures", {"t1": "succeeded", "t2": "failed"})], at)
    assert [n.title for n in out] == ["Tree failed", "Batch completed with failures"]
    assert notes.observe_batches([info("completed_with_failures", {"t1": "succeeded", "t2": "failed"})], at) == []
    new = Notifications()
    new.observe_batches([], at)  # a batch that appears later reports its start
    assert new.observe_batches([info("running", {"t1": "running"})], at)[0].title == "Batch started"

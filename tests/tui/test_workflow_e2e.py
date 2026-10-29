"""Isolated synthetic UI workflows. No ML training, downloaded models or invented metrics."""
import asyncio
import copy
import json

import pytest
from textual.widgets import Button, Input

from test_batch import wait_for
from test_batch_ui import until, with_runner, stop_all
from test_review_batch import BASE, make_root, run_review, settle, tweaked
from xmr4el_tui import batch
from xmr4el_tui.app import XMR4ELApp
from xmr4el_tui.screens.batch_monitor import BatchMonitorScreen
from xmr4el_tui.screens.dashboard_screen import DashboardScreen
from xmr4el_tui.screens.dataset_setup import DatasetSetupScreen
from xmr4el_tui.screens.model_builder import ModelBuilderScreen
from xmr4el_tui.screens.review_batch import DetailsList, ReviewBatchScreen
from xmr4el_tui.screens.stop_dialog import QuitDialog


@pytest.mark.parametrize("mode,expected", [("ok", "completed"), ("middle_fail", "completed_with_failures")])
def test_dashboard_to_batch_and_reconnect(tmp_path, monkeypatch, mode, expected):
    root = make_root(tmp_path / "research workspace")
    (root / "configs/experiments").mkdir()
    (root / "configs/experiments/synthetic.json").write_text(json.dumps({
        "train_path": "datasets/A/train.txt", "model_config": "configs/xmr4el_base_config.json",
        "evals": [{"split": "dev", "test_path": "datasets/A/dev.pubtator"}]}))
    calls = with_runner(root, monkeypatch, seconds=2)
    if mode == "middle_fail":
        path = root / "fake_runner.py"
        path.write_text(path.read_text().replace('if mode == "fail":',
                                               'if mode == "fail" or cfg.get("predict_config", {}).get("beam_size") == 7:'))

    async def go():
        app = XMR4ELApp(root)
        async with app.run_test(size=(120, 36)) as pilot:
            await settle(app, pilot)
            assert isinstance(app.screen, DashboardScreen)
            await pilot.press("n")
            await settle(app, pilot)
            assert isinstance(app.screen, DatasetSetupScreen)
            await pilot.press("enter", "ctrl+s", "m")
            await pilot.pause()
            assert isinstance(app.screen, ModelBuilderScreen)
            await pilot.press("right", "right", "enter", "enter")  # Matcher -> editor
            assert isinstance(app.focused, Input)
            await pilot.press("ctrl+a", "ctrl+k", "2", "ctrl+s")
            assert app.draft.trees[0].config["matcher_config"]["kwargs"]["C"] == 2
            await pilot.press("escape", "n")  # Tree Selection -> new tree
            await pilot.press("right", "enter", "enter")  # preserved Matcher slot -> Search
            assert app.screen.editing.slot.id == "search"
            await pilot.press("ctrl+a", "ctrl+k", "7", "ctrl+s")
            assert app.draft.trees[1].config["predict_config"]["beam_size"] == 7
            await pilot.press("escape", "n")
            assert len(app.draft.trees) == 3
            assert await pilot.click("#mb-review")
            await settle(app, pilot)
            review_screen = app.screen
            assert isinstance(review_screen, ReviewBatchScreen)
            assert not review_screen.query_one("#rv-launch", Button).disabled
            await pilot.press("home", "alt+right", "alt+left")
            expected_ids = [t.id for t in app.draft.trees]
            await pilot.press("enter", "2")
            assert review_screen.view == "differences"
            await pilot.press("escape", "f2")
            await pilot.press("ctrl+a", "ctrl+k", *"Synthetic audit batch", "enter")
            assert app.draft.name == "Synthetic audit batch"
            assert await pilot.click("#rv-launch")  # actual mouse press: the footer asks ...
            assert not isinstance(app.screen, BatchMonitorScreen)
            await pilot.press("l")  # ... and `l` answers (like every footer confirmation) -> real detached coordinator
            assert await until(pilot, lambda: isinstance(app.screen, BatchMonitorScreen))
            bid = app.screen.batch_id
            snapshot = copy.deepcopy(calls[0].trees[0].config)
            app.draft.trees[0].config["depth"] = 999
            assert calls[0].trees[0].config == snapshot
            await pilot.press("escape")
            assert isinstance(app.screen, DashboardScreen)
            assert not any(isinstance(s, ReviewBatchScreen) for s in app.screen_stack)
            await pilot.press("ctrl+c")
            assert isinstance(app.screen, QuitDialog) and app.focused.id == "stay"
            assert await pilot.click("#leave")

        # A new app reconnects while the same coordinator is still running.
        assert batch.info(batch.directory(root, bid)).active
        other = XMR4ELApp(root)
        async with other.run_test(size=(100, 30)) as pilot:
            await settle(other, pilot)
            await pilot.press("b", "enter")
            await pilot.pause()
            monitor = other.screen
            assert isinstance(monitor, BatchMonitorScreen) and monitor.batch_id == bid
            assert "Dataset  A" in str(monitor.query_one("#bm-overview").render())
            assert await until(pilot, lambda: monitor.info.state == expected, seconds=20)
            assert [t["id"] for t in monitor.info.status["trees"]] == expected_ids
            assert len(calls) == 1
            assert monitor.info.count("failed") == (mode == "middle_fail")
            assert monitor.info.count("succeeded") == (2 if mode == "middle_fail" else 3)
            assert len({t["run_id"] for t in monitor.info.status["trees"]}) == 3
            other.rescan()
            await settle(other, pilot)
            assert len(other.snapshot.runs) == 3
            assert all(not e.metrics for r in other.snapshot.runs for e in r.evals), "no invented evaluation metrics"
            await pilot.press("escape")
            assert isinstance(other.screen, DashboardScreen)
            await pilot.press("b", "enter", "escape")
            assert isinstance(other.screen, DashboardScreen)
        assert wait_for(lambda: batch.holder(root) is None)

    try:
        asyncio.run(go())
    finally:
        stop_all(root)


@pytest.mark.parametrize("size", [(80, 24), (100, 30), (140, 40)])
def test_launch_visible_and_keyboard_reachable_with_many_trees(tmp_path, size):
    async def go(app, pilot, screen):
        button = screen.query_one("#rv-launch", Button)
        assert not button.disabled
        assert button.region.y >= 0 and button.region.bottom <= size[1] - 1
        await pilot.press("tab")
        assert app.focused is button
        await pilot.press("shift+tab")
        assert app.focused.id == "rv-queue"
    run_review(make_root(tmp_path), go, configs=(BASE,) * 14, size=size)


@pytest.mark.parametrize("path,bad,good,slot", [
    (("transformer_config", "kwargs", "batch_size"), -1, "12", "transformer"),
    (("dimension_config", "kwargs", "n_components"), 0, "12", "mention_svd"),
    (("context_dimension_config", "kwargs", "n_components"), 0, "12", "context_svd"),
    (("clustering_config", "kwargs", "n_clusters"), 0, "4", "hierarchy"),
    (("matcher_config", "kwargs", "C"), 0, "2", "matcher"),
    (("predict_config", "beam_size"), 0, "7", "search"),
])
def test_fix_issue_reaches_and_applies_each_parameter(tmp_path, path, bad, good, slot):
    cfg = tweaked(path, bad)

    async def go(app, pilot, screen):
        assert screen.query_one("#rv-launch", Button).disabled
        await pilot.press("enter")  # queue -> issues
        assert isinstance(app.focused, DetailsList)
        await pilot.press("enter")  # Fix Issue on first error
        await settle(app, pilot)
        builder = app.screen
        assert isinstance(builder, ModelBuilderScreen) and builder.editing.slot.id == slot
        assert builder.focus_path == path and isinstance(app.focused, Input)
        await pilot.press("ctrl+a", "ctrl+k", *good)
        assert app.draft.trees[0].config == cfg
        await pilot.press("ctrl+s", "escape", "escape")
        await settle(app, pilot)
        assert app.screen is screen
        value = app.draft.trees[0].config
        for key in path:
            value = value[key]
        assert value == int(good)
        assert not screen.query_one("#rv-launch", Button).disabled
    run_review(make_root(tmp_path), go, configs=(cfg,))

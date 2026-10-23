import asyncio

import pytest

pytest.importorskip("textual")

from fake_runs import make_run  # noqa: E402
from textual.widgets import DataTable, Static, TabbedContent  # noqa: E402
from xmr4el_tui.app import XMR4ELApp  # noqa: E402
from xmr4el_tui.screens import RunScreen, RunsScreen, columns_for  # noqa: E402
from xmr4el_tui.widgets import LogView  # noqa: E402


def fake_root(tmp_path):
    make_run(tmp_path, "ncbi_disease", tree_here=True, splits=("dev", "train_dev"))
    make_run(tmp_path, "ncbi_disease_test", stamp="2026-10-20_13-38-45", kind="eval", splits=("test",))
    make_run(tmp_path, "chem", stamp="2026-10-23_14-10-00", status="running", tree_id=None, metrics={"dev": None})
    make_run(tmp_path, "bad", stamp="2026-10-19_10-00-00", status="failed", tree_id="dead", metrics={"dev": None})
    return tmp_path


def text(widget):
    return str(widget.render())


def run_app(root, test, size=(80, 24)):
    async def go():
        app = XMR4ELApp(root)
        async with app.run_test(size=size) as pilot:
            await pilot.pause()
            await test(app, pilot)
    asyncio.run(go())


def test_runs_table_details_and_open(tmp_path):
    async def check(app, pilot):
        screen = app.screen
        assert isinstance(screen, RunsScreen)
        table = screen.query_one(DataTable)
        assert table.row_count == 4
        assert [c.value for c in table.columns] == ["mark", "started", "run", "kind", "tree", "split", "acc@1",
                                                    "state"]
        assert table.get_row_at(0)[-1].plain == "UNVERIFIED"  # recorded running, no process identity
        await pilot.press("down")  # ncbi_disease_test
        details = text(screen.query_one("#details", Static))
        assert "ncbi_disease_test · eval" in details and "also: ncbi_disease" in details
        assert "test  acc@1 0.9246" in details
        await pilot.press("down", "enter")
        run_screen = app.screen
        assert isinstance(run_screen, RunScreen) and run_screen.run.name == "ncbi_disease"
        overview = text(run_screen.query_one("#overview-text", Static))
        assert "Tree   8528b624 · trained here · on this machine · also: ncbi_disease_test" in overview
        assert "Rows        729 / 757" in overview and "Mention strings  not recorded" in overview
        assert "Stages not recorded" in overview
        await pilot.press("2")
        assert run_screen.query_one(TabbedContent).active == "log"
        log = run_screen.query_one(LogView)
        assert any("Run started" in ln for ln in log.lines)
        await pilot.press("3")
        inputs = text(run_screen.query_one("#inputs-text", Static))
        assert "-spec -" in inputs and "Git          b10a922" in inputs
        await pilot.press("escape")
        assert isinstance(app.screen, RunsScreen)
        assert app.screen.selected().name == "ncbi_disease"  # selection kept
    run_app(fake_root(tmp_path), check)


def test_eval_selector_and_failed_banner(tmp_path):
    async def check(app, pilot):
        await pilot.press("down", "down", "enter")
        screen = app.screen
        assert screen.split == "dev"
        screen.query_one("#split").value = "train_dev"
        await pilot.pause()
        assert "Evaluation  train_dev" in text(screen.query_one("#overview-text", Static))
        await pilot.press("escape", "down", "enter")
        overview = text(app.screen.query_one("#overview-text", Static))
        assert "ValueError: boom" in overview and "dev: no metrics" in overview
    run_app(fake_root(tmp_path), check)


def test_filter_sort_marks(tmp_path):
    async def check(app, pilot):
        screen = app.screen
        table = screen.query_one(DataTable)
        await pilot.press("slash", *"test", "enter")
        assert table.row_count == 1 and screen.filter == "test"
        await pilot.press("slash", "escape")
        assert table.row_count == 4
        await pilot.press("o", "o")  # acc@1, grouped by split
        assert screen.sort == "acc@1"
        await pilot.press("home", "space", "down", "space", "down", "space")
        assert len(screen.marked) == 2
    run_app(fake_root(tmp_path), check)


def test_narrow_and_empty(tmp_path):
    assert columns_for(60) == ["mark", "run", "tree", "split", "acc@1", "state"]
    assert "duration" in columns_for(120)

    async def narrow(app, pilot):
        assert [c.value for c in app.screen.query_one(DataTable).columns] == columns_for(60)
    run_app(fake_root(tmp_path), narrow, size=(60, 18))

    async def missing(app, pilot):
        assert "outputs/runs/ not found" in text(app.screen.query_one("#details", Static))
    run_app(tmp_path / "nowhere", missing)

    (tmp_path / "empty/outputs/runs").mkdir(parents=True)

    async def empty(app, pilot):
        assert "No runs in outputs/runs/" in text(app.screen.query_one("#details", Static))
    run_app(tmp_path / "empty", empty)

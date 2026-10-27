import asyncio

import pytest

pytest.importorskip("textual")

from fake_runs import make_run  # noqa: E402
from textual.widgets import DataTable, Static, TabbedContent  # noqa: E402
from xmr4el_tui.app import XMR4ELApp, settings_path  # noqa: E402
from xmr4el_tui import process  # noqa: E402
from xmr4el_tui.screens import CompareScreen, RunScreen, RunsScreen, StopModal, columns_for  # noqa: E402
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
        assert table.get_row_at(0)[-1].plain == "UNVER"  # recorded running, no process identity
        await pilot.press("down")  # ncbi_disease_test
        details = text(screen.query_one("#details", Static))
        assert details.startswith("ncbi_disease_test\neval · COMPLETED") and "Also      ncbi_disease\n" in details
        assert "test\nacc@1     0.9246" in details
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


def test_compare_selection_and_swap(tmp_path):
    hashed = {"preprocessing": {"context_window": None, "abbrev_expansion": None}, "population_sha256": "p"}
    make_run(tmp_path, "both", splits=("dev", "test"), metrics={"dev": hashed, "test": hashed})
    make_run(tmp_path, "test_only", stamp="2026-10-21_00-00-00", kind="eval", splits=("test",),
             metrics={"test": {**hashed, "acc@1": 0.9}})

    async def check(app, pilot):
        await pilot.press("c")  # nothing marked: refused
        assert isinstance(app.screen, RunsScreen)
        await pilot.press("space", "down", "space", "c")
        screen = app.screen
        assert isinstance(screen, CompareScreen) and screen.splits == ["test", "test"]
        assert not screen.query("#sel-0") and screen.query("#sel-1")  # selector only on the 2-eval side
        body = text(screen.query_one("#compare-text", Static))
        assert "hashes match" in body and "+0.0246" in body  # A = test_only (newest, marked first)
        screen.query_one("#sel-1").value = "dev"  # different file: no deltas
        await pilot.pause()
        body = text(screen.query_one("#compare-text", Static))
        assert "Not comparable" in body and "B - A" not in body
        await pilot.press("x")
        assert app.screen.splits == ["dev", "test"] and app.screen.query_one("#sel-0").value == "dev"
        assert app.screen.active_bindings["escape"].binding.action == "app.pop_screen"  # footer: esc runs
        await pilot.press("escape")
        assert isinstance(app.screen, RunsScreen)
    run_app(tmp_path, check)


def test_help_replaces_details_and_theme_persists(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    root = fake_root(tmp_path)

    async def help_and_theme(app, pilot):
        panel = app.screen.query_one(".side-panel")
        await pilot.press("question_mark")
        assert app.screen.query("HelpPanel") and not panel.display
        await pilot.press("escape")  # closes help, not the screen
        assert not app.screen.query("HelpPanel") and panel.display and isinstance(app.screen, RunsScreen)
        await pilot.press("ctrl+r")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert any("Rescanned" in str(n.message) for n in app._notifications)
        await pilot.press("ctrl+q")  # first press warns only
        await pilot.pause()
        assert app.is_running and any("again to quit" in str(n.message) for n in app._notifications)
        app.theme = "nord"
        await pilot.pause()
    run_app(root, help_and_theme, size=(140, 30))
    assert '"nord"' in settings_path().read_text()

    async def restored(app, pilot):
        assert app.theme == "nord"
    run_app(root, restored)


def test_stop_button_modal_and_stopping(tmp_path, monkeypatch):
    make_run(tmp_path, "chem", stamp="2026-10-23_14-10-00", status="running", tree_id=None, metrics={"dev": None},
             identity={"pid": 4182, "pgid": 4182})
    monkeypatch.setattr(process, "derived_state", lambda run, here: process.Live("RUNNING", stoppable=True))
    sent = []
    monkeypatch.setattr(process, "stop", lambda run, force=False: sent.append(force) or "SIGTERM sent to pid 4182")

    async def check(app, pilot):
        await pilot.press("enter")
        screen = app.screen
        assert isinstance(screen, RunScreen) and screen.query_one("#run-actions").display
        await pilot.click("#stop")
        assert isinstance(app.screen, StopModal) and app.focused.id == "keep", "focus starts on Keep running"
        await pilot.press("enter")
        assert app.screen is screen and sent == [], "Keep running sends nothing"
        await pilot.press("s", "tab", "enter")  # Stop run
        await pilot.pause()
        assert sent == [False] and "STOPPING" in text(screen.query_one("#top"))
        assert not screen.check_action("stop", ()) and not screen.check_action("force_stop", ())
        screen.stopping -= 31
        assert screen.check_action("force_stop", ())
        await pilot.press("k")
        assert sent == [False, True]
    run_app(tmp_path, check)

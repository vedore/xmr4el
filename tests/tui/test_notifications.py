"""Navbar (breadcrumbs, centred title, bell), session notifications (manager, toasts, center) and the contextual
footer (feedback, quit confirmation)."""
import asyncio
from dataclasses import replace
from datetime import datetime
from types import SimpleNamespace

import pytest

pytest.importorskip("textual")

from fake_runs import make_run, patch_json  # noqa: E402
from test_results import A, B, C, D, E, F, cursor_to, open_results, results_root, show_incompatible  # noqa: E402
from xmr4el_tui import app as app_module  # noqa: E402
from xmr4el_tui import runs  # noqa: E402
from xmr4el_tui.app import XMR4ELApp  # noqa: E402
from xmr4el_tui.screens.dashboard_screen import DashboardScreen  # noqa: E402
from xmr4el_tui.screens.monitor_screen import RunMonitorScreen  # noqa: E402
from xmr4el_tui.screens.notification_center import NotificationCenter  # noqa: E402
from xmr4el_tui.services.notifications import HISTORY, Notice, Notifications # noqa: E402
from xmr4el_tui.widgets.explorer import Explorer  # noqa: E402
from xmr4el_tui.widgets.top_bar import TITLE, TopBar  # noqa: E402

LIVE = "2026-10-23_14-10-00_live"


def run_app(root, test, size=(120, 36)):
    async def go():
        app = XMR4ELApp(root)
        async with app.run_test(size=size) as pilot:
            await settle(app, pilot)
            await test(app, pilot)
    asyncio.run(go())


async def settle(app, pilot):
    await app.workers.wait_for_complete()
    await pilot.pause()


def bar(app):
    return app.screen.query_one(TopBar).render().plain


def footer_note(app):
    note = app.screen.query_one("#footer-note")
    return str(note.render()) if note.display else ""


def notice(i, target=None, severity="success"):
    return Notice((f"run{i}", "COMPLETED"), severity, "Experiment completed", f"run{i} · SapBERT D2",
                  "Results are available.", datetime.now(), target)


# --- manager ----------------------------------------------------------------------------------------------------
def scanned(tmp_path, **kw):
    root = tmp_path / "ws"
    make_run(root, "one", **kw)
    return runs.scan(root)[0]


def as_state(run, state):
    """`run` as the scan shows it in `state` (RUNNING / LOST: recorded running with that live state)."""
    if state in ("RUNNING", "LOST"):
        return replace(run, recorded="running", live=SimpleNamespace(state=state))
    return replace(run, recorded=state.lower(), live=None)


def test_first_scan_records_states_without_notices(tmp_path):
    run = scanned(tmp_path)
    n = Notifications()
    assert n.observe([as_state(run, "COMPLETED"), as_state(replace(run, id="x"), "FAILED")]) == []
    assert not n.items and n.unseen == 0, "no flood of old results on startup"


@pytest.mark.parametrize("state,severity,title,target", [
    ("COMPLETED", "success", "Experiment completed", "results"),
    ("FAILED", "error", "Training failed", "monitor"),
    ("LOST", "warning", "Run interrupted", "monitor"),
    ("STOPPED", "information", "Run cancelled", "monitor"),
])
def test_transition_from_active_makes_one_notice(tmp_path, state, severity, title, target):
    run = scanned(tmp_path, model_config={"features": "sapbert_char_context", "depth": 2})
    n = Notifications()
    n.observe([as_state(run, "RUNNING")])
    new = n.observe([as_state(run, state)])
    assert [(x.severity, x.title, x.target[0]) for x in new] == [(severity, title, target)]
    assert new[0].subject == "one · SapBERT D2" and n.unseen == 1
    if target == "results":
        assert new[0].target == ("results", ("eval", run.id, "dev")), "the evaluation, explicitly"
    else:
        assert new[0].target == ("monitor", run.id)
    assert n.observe([as_state(run, state)]) == [] and len(n.items) == 1, "repeated scans add nothing"
    n.observe([as_state(run, "RUNNING")])
    assert n.observe([as_state(run, state)]) == [], "the same (run, end state) is reported once"


def test_short_run_seen_only_ended_is_notified_old_copy_is_not(tmp_path):
    run = scanned(tmp_path)  # started 2026-10-20 13:36:20 (from its id)
    n = Notifications()
    n.observe([], now=datetime(2026, 10, 20, 13, 0))
    assert [x.title for x in n.observe([as_state(run, "COMPLETED")])] == ["Experiment completed"], \
        "started and ended between two scans"
    late = Notifications()
    late.observe([], now=datetime(2026, 10, 21))
    assert late.observe([as_state(run, "COMPLETED")]) == [], "older than the session: copied in, not an event"


def test_several_evaluations_never_pick_a_baseline(tmp_path):
    run = scanned(tmp_path, splits=("dev", "test"))
    n = Notifications()
    n.observe([as_state(run, "RUNNING")])
    assert n.observe([as_state(run, "COMPLETED")])[0].target == ("monitor", run.id)


def test_history_bounded_clear_and_session_only():
    n = Notifications()
    n.add([notice(i) for i in range(HISTORY + 10)])
    assert len(n.items) == HISTORY and n.unseen == HISTORY and n.items[0].key == (f"run{HISTORY + 9}", "COMPLETED")
    assert n.add([notice(3)]) == [], "duplicate key"
    n.seen()
    assert n.unseen == 0 and len(n.items) == HISTORY, "seen keeps the history"
    n.clear()
    assert not n.items and n.unseen == 0
    assert len(n.add([notice(999)])) == 1, "a new event after Clear all appears"
    assert not Notifications().items, "a new session starts empty"


# --- app: lifecycle, toasts ---------------------------------------------------------------------------------------
def live_root(tmp_path):
    make_run(tmp_path, "done", stamp="2026-10-20_10-00-00")
    make_run(tmp_path, "live", stamp="2026-10-23_14-10-00", status="running", tree_id=None, metrics={"dev": None})
    return tmp_path


def test_scan_transition_notifies_once_and_toast_keeps_focus(tmp_path):
    root = live_root(tmp_path)

    async def check(app, pilot):
        focused = app.focused
        assert not app.notices.items and not app._notifications, "existing runs: no notification on startup"
        patch_json(root / "outputs/runs" / LIVE / "status.json", status="stopped")
        for _ in range(2):  # the second scan sees the same ended run
            app.rescan(announce=True)
            await settle(app, pilot)
        assert [n.title for n in app.notices.items] == ["Run cancelled"]
        assert [str(t.title) for t in app._notifications] == [" Run cancelled "], "one toast, title chip padded"
        assert list(app._notifications)[0].severity == "information"
        assert app.focused is focused, "a toast takes no focus"
        assert "Notifications 1" in bar(app)
    run_app(root, check)


def test_toasts_consolidate_and_skip_the_open_center(tmp_path):
    async def check(app, pilot):
        app.toast(app.notices.add([notice(1), notice(2, severity="error")]))
        await pilot.pause()
        assert len(app._notifications) == 1 and list(app._notifications)[0].severity == "error"
        assert "2 experiment events" in list(app._notifications)[0].message
        await pilot.press("ctrl+n")
        await pilot.pause()
        app.toast(app.notices.add([notice(3)]))
        await pilot.pause()
        assert len(app._notifications) == 1, "no toast over the open center"
        assert len(app.screen.ids) == 3 and app.notices.unseen == 0, "listed (and so seen) instead"
    run_app(live_root(tmp_path), check)


# --- navbar -------------------------------------------------------------------------------------------------------
def test_breadcrumbs_follow_the_screen_stack(tmp_path):
    root = results_root(tmp_path)

    async def check(app, pilot):
        w = app.size.width
        assert bar(app).startswith(" Dashboard ") and bar(app).index(TITLE) == (w - len(TITLE)) // 2
        screen, tree = await open_results(app, pilot)
        assert bar(app).startswith(" Dashboard / Results ")
        await pilot.press("enter")  # inspect
        await pilot.pause()
        assert bar(app).startswith(" Dashboard / Results / Inspect ")
        assert bar(app).index(TITLE) == (w - len(TITLE)) // 2, "the title does not move"
        await pilot.click(TopBar, offset=(16, 0))  # "Results": closes the inspection only
        await pilot.pause()
        assert screen.inspecting is None and app.screen is screen and bar(app).startswith(" Dashboard / Results ")
        await pilot.click(TopBar, offset=(3, 0))  # "Dashboard"
        await pilot.pause()
        assert isinstance(app.screen, DashboardScreen) and bar(app).startswith(" Dashboard ")
        app.screen.open_run(app.snapshot.runs[0].id)
        await settle(app, pilot)
        assert isinstance(app.screen, RunMonitorScreen) and bar(app).startswith(" Dashboard / Run Monitor ")
    run_app(root, check, size=(140, 40))


def test_narrow_navbar_keeps_title_centred_and_current_crumb(tmp_path):
    async def check(app, pilot):
        await open_results(app, pilot)
        await pilot.press("enter")
        await pilot.pause()
        line = bar(app)
        assert line.index(TITLE) == (40 - len(TITLE)) // 2 and len(line) <= 40
        assert "…" in line and line.index("Inspect") < line.index(TITLE), "cut on the left, the current end kept"
    run_app(results_root(tmp_path), check, size=(40, 20))


# --- notification center ------------------------------------------------------------------------------------------
def test_center_seen_clear_focus_and_comparison_kept(tmp_path):
    async def check(app, pilot):
        screen, tree = await open_results(app, pilot)
        await cursor_to(pilot, tree, B)
        await pilot.press("space")
        focused, members, cursor = app.focused, list(screen.members), tree.cursor_line
        app.notices.add([notice(1), notice(2)])
        app.refresh_bars()
        assert "Notifications 2" in bar(app)
        await pilot.press("ctrl+n")
        await pilot.pause()
        center = app.screen
        assert isinstance(center, NotificationCenter) and app.notices.unseen == 0
        assert "Notifications 2" not in screen.query_one(TopBar).render().plain
        await pilot.press("down")
        assert center.query_one("#nc-list").highlighted == 1, "row by row (separators are skipped)"
        await pilot.press("escape")
        await pilot.pause()
        assert app.screen is screen and app.focused is focused and screen.members == members
        assert tree.cursor_line == cursor and len(app.notices.items) == 2
        await pilot.press("ctrl+n")
        await pilot.pause()
        await pilot.press("c")
        await pilot.pause()
        assert not app.notices.items and "All caught up" in str(app.screen.query_one("#nc-empty").render())
    run_app(results_root(tmp_path), check, size=(140, 40))


def test_center_opens_destinations_or_says_they_are_gone(tmp_path):
    root = results_root(tmp_path)

    async def check(app, pilot):
        await open_results(app, pilot)
        app.notices.add([notice(1, ("results", B)), notice(2, ("monitor", "gone")), notice(3, ("monitor", A[1]))])
        await pilot.press("ctrl+n")
        await pilot.pause()
        await pilot.press("enter")  # newest: the existing run's monitor, from the Dashboard
        await settle(app, pilot)
        assert isinstance(app.screen, RunMonitorScreen) and app.screen.run.id == A[1]
        assert [type(s) for s in app.screen_stack[1:]] == [DashboardScreen, RunMonitorScreen]
        await pilot.press("ctrl+n")
        await pilot.pause()
        await pilot.press("down")  # the vanished run
        await pilot.press("enter")
        await settle(app, pilot)
        assert isinstance(app.screen, RunMonitorScreen) and "no longer found" in footer_note(app)
        await pilot.press("ctrl+n")
        await pilot.pause()
        await pilot.click("#nc-list", offset=(2, 9))  # third row: Results with B as the baseline
        await settle(app, pilot)
        assert app.screen.base == B and app.screen.members == []
    run_app(root, check, size=(140, 40))


# --- footer -------------------------------------------------------------------------------------------------------
def test_footer_feedback_replaces_toasts_and_expires(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "FEEDBACK_SECONDS", 0.3)

    async def check(app, pilot):
        screen, tree = await open_results(app, pilot)
        await show_incompatible(app, pilot)
        await cursor_to(pilot, tree, D)
        await pilot.press("space")
        assert footer_note(app).startswith(" Notice  Incompatible with the baseline")
        for key in (B, C, F, E):
            await cursor_to(pilot, tree, key)
            await pilot.press("space")
        assert "Maximum of 3 challengers" in footer_note(app), "newer feedback replaces older at once"
        await pilot.pause(0.2)
        app.say("newer")
        await pilot.pause(0.2)  # the replaced message's time is over: its timer cannot clear the newer one
        assert "newer" in footer_note(app)
        await pilot.pause(0.3)
        assert footer_note(app) == "" and not app._notifications, "back to the shortcuts; nothing toasted"
        app.say("left behind")
        await pilot.press("escape")  # back to the Dashboard
        await pilot.pause()
        assert footer_note(app) == "", "feedback does not follow navigation"
        await pilot.press("r")  # the Dashboard's Models group: not an evaluation
        assert "Select an evaluation to open Results." in footer_note(app)
    run_app(results_root(tmp_path), check, size=(140, 40))


def test_quit_confirmation_in_the_footer(tmp_path):
    exits = []

    async def check(app, pilot):
        app.exit = lambda *a, **k: exits.append(1)
        await pilot.press("ctrl+c")
        note = footer_note(app)
        assert "Quit XMR4EL? 1 active run will keep running." in note and "esc cancel" in note
        assert not app.screen.query_one("Footer").display, "the confirmation replaces the shortcuts"
        app.say("feedback")
        app.toast(app.notices.add([notice(1)]))
        await pilot.pause(0.2)
        assert "Quit XMR4EL?" in footer_note(app), "feedback and notifications do not replace it"
        await pilot.press("escape")
        assert footer_note(app) == "" and app.screen.query_one("Footer").display and not exits
        assert isinstance(app.focused, Explorer), "esc only cancelled"
        await pilot.press("ctrl+c")
        await pilot.press("ctrl+c")
        assert exits == [1]
    run_app(live_root(tmp_path), check)

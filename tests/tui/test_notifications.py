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
from xmr4el_tui import process, runs  # noqa: E402
from xmr4el_tui.app import XMR4ELApp  # noqa: E402
from xmr4el_tui.screens.dashboard_screen import DashboardScreen  # noqa: E402
from xmr4el_tui.screens.monitor_screen import RunMonitorScreen  # noqa: E402
from xmr4el_tui.screens.notification_center import NotificationCenter  # noqa: E402
from xmr4el_tui.screens.results_screen import ResultsScreen  # noqa: E402
from xmr4el_tui.services.experiment_repository import ExperimentRepository  # noqa: E402
from xmr4el_tui.services.notifications import HISTORY, Notice, Notifications, incarnation  # noqa: E402
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
    ("STOPPED", "warning", "Run stopped", "monitor"),
])
def test_transition_from_active_makes_one_notice(tmp_path, state, severity, title, target):
    run = scanned(tmp_path, model_config={"features": "sapbert_char_context", "depth": 2})
    n = Notifications()
    n.observe([as_state(run, "RUNNING")])
    new = n.observe([as_state(run, state)])
    assert [(x.severity, x.title, x.target[0]) for x in new] == [(severity, title, target)]
    assert new[0].subject == "one · SapBERT D2" and n.unseen == 1
    if target == "results":
        assert new[0].target[:2] == ("results", ("eval", run.id, "dev")), "the evaluation, explicitly"
    else:
        assert new[0].target[:2] == ("monitor", run.id)
    assert new[0].target[2] == incarnation(run), "bound to this run, not just its directory name"
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
    assert n.observe([as_state(run, "COMPLETED")])[0].target[:2] == ("monitor", run.id)


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
        assert [n.title for n in app.notices.items] == ["Run stopped"], "stopped outside the TUI: no intent claimed"
        assert [str(t.title) for t in app._notifications] == [" Run stopped "], "one toast, title chip padded"
        assert list(app._notifications)[0].severity == "warning"
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
        assert not app._notifications, "opening the center removes the shown toast (the list has it)"
        app.toast(app.notices.add([notice(3)]))
        await pilot.pause()
        assert not app._notifications, "no toast over the open center"
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


def target(app, kind, key):
    """A notice destination for the run (or evaluation key) as the app's latest scan has it."""
    run_id = key[1] if kind == "results" else key
    run = next(r for r in app.snapshot.runs if r.id == run_id)
    return kind, key, incarnation(run)


def test_center_opens_destinations_over_the_current_screen(tmp_path):
    root = results_root(tmp_path)

    async def check(app, pilot):
        screen, tree = await open_results(app, pilot)
        await cursor_to(pilot, tree, B)
        await pilot.press("space")
        members = list(screen.members)
        app.notices.add([notice(1, target(app, "results", C)), notice(2, ("monitor", "gone", ("gone", None, 1, 2))),
                         notice(3, target(app, "monitor", A[1]))])
        await pilot.press("ctrl+n")
        await pilot.pause()
        await pilot.press("enter")  # newest: the existing run's monitor, over the comparison
        await settle(app, pilot)
        assert isinstance(app.screen, RunMonitorScreen) and app.screen.run.id == A[1]
        assert [type(s) for s in app.screen_stack[1:]] == [DashboardScreen, ResultsScreen, RunMonitorScreen]
        await pilot.press("ctrl+n")
        await pilot.pause()
        await pilot.press("enter")  # the same monitor again: not stacked twice
        await settle(app, pilot)
        assert len(app.screen_stack) == 4
        await pilot.press("ctrl+n")
        await pilot.pause()
        await pilot.press("down")  # the vanished run
        await pilot.press("enter")
        await settle(app, pilot)
        assert isinstance(app.screen, RunMonitorScreen) and "no longer found" in footer_note(app)
        await pilot.press("ctrl+n")
        await pilot.pause()
        await pilot.click("#nc-list", offset=(2, 9))  # third row: Results with C as the baseline, a new session
        await settle(app, pilot)
        assert isinstance(app.screen, ResultsScreen) and app.screen.base == C and app.screen.members == []
        for _ in range(3):  # Results esc: panels -> Explorer -> back; then the monitor's esc
            await pilot.press("escape")
            await settle(app, pilot)
            if app.screen is screen:
                break
        if app.screen is not screen:
            app.pop_screen()
            await settle(app, pilot)
        assert app.screen is screen and screen.members == members and screen.base == A, "the comparison is kept"
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


# --- lifecycle reconciliation, attribution, identity (audit 2026-10-28) -------------------------------------------
def ident(run, pid=100, start="s100"):
    return replace(run, identity={"pid": pid, "start": start})


@pytest.mark.parametrize("final,title", [("COMPLETED", "Experiment completed"), ("STOPPED", "Run stopped"),
                                         ("FAILED", "Training failed")])
def test_lost_is_provisional_and_a_recorded_outcome_replaces_it(tmp_path, final, title):
    run = ident(scanned(tmp_path))
    n = Notifications()
    n.observe([as_state(run, "RUNNING")])
    assert [x.title for x in n.observe([as_state(run, "LOST")])] == ["Run interrupted"]
    assert [x.title for x in n.observe([as_state(run, final)])] == [title], "the outcome read later is notified"
    assert [x.title for x in n.items] == [title] and n.unseen == 1, "the false interruption notice is gone"
    assert n.observe([as_state(run, final)]) == [] and not n.watched, "settled: no longer watched"


def test_unreadable_status_between_running_and_the_outcome(tmp_path):
    run = ident(scanned(tmp_path))
    unknown = replace(run, recorded="unknown", identity=None, started=run.started.replace(second=59))
    n = Notifications()
    n.observe([as_state(run, "RUNNING")])
    assert n.observe([unknown]) == [], "UNKNOWN is no end"
    assert [x.title for x in n.observe([as_state(run, "COMPLETED")])] == ["Experiment completed"]


def test_stopped_names_the_signal_unless_this_tui_cancelled(tmp_path):
    run = ident(scanned(tmp_path))
    for requested, want in ((False, ("warning", "Run stopped", "Stopped by SIGTERM.")),
                            (True, ("information", "Run cancelled", "Stopped on request."))):
        n = Notifications()
        n.observe([as_state(run, "RUNNING")])
        if requested:
            n.requested.add(incarnation(run))
        x = n.observe([replace(as_state(run, "STOPPED"), error="SIGTERM")])[0]
        assert (x.severity, x.title, x.text) == want
    n = Notifications()
    n.observe([as_state(run, "RUNNING")])
    n.requested.add(incarnation(run))
    assert [x.title for x in n.observe([as_state(run, "COMPLETED")])] == ["Experiment completed"], \
        "a cancel racing completion: the recorded completion wins"
    n = Notifications()
    n.observe([as_state(run, "RUNNING")])
    assert [x.title for x in n.observe([as_state(run, "LOST")])] == ["Run interrupted"], "SIGKILL records nothing"


def test_a_new_run_reusing_a_directory_is_a_new_run(tmp_path):
    old = ident(scanned(tmp_path))
    new = ident(replace(old, started=old.started.replace(hour=15)), pid=200, start="s200")
    n = Notifications()
    n.observe([], now=old.started.replace(hour=12))
    first = n.observe([as_state(old, "COMPLETED")])[0]
    again = n.observe([as_state(new, "COMPLETED")])
    assert len(again) == 1 and again[0].key != first.key, "the replacement's completion is not deduplicated away"
    assert first.target[2] != again[0].target[2]


def test_dedup_memory_is_bounded_by_existing_runs(tmp_path):
    run = ident(scanned(tmp_path))
    n = Notifications()
    n.observe([as_state(run, "RUNNING")])
    n.observe([as_state(run, "COMPLETED")])
    assert n.sent
    n.observe([])
    assert not n.sent and not n.watched and len(n.items) == 1, "the run is gone: its keys go, its notice stays"
    n.observe([], problem="outputs/runs/ cannot be read")
    assert n.observe([], problem="outputs/runs/ cannot be read") == [], "a scan problem is reported once"
    n.observe([])
    assert not n.sent, "cleared problem forgotten"


def test_same_second_start_and_scan_time_boundary(tmp_path):
    run = scanned(tmp_path)  # started 2026-10-20 13:36:20, recorded to the second
    n = Notifications()
    n.observe([], now=datetime(2026, 10, 20, 13, 36, 20, 500000))
    assert [x.title for x in n.observe([as_state(run, "COMPLETED")])] == ["Experiment completed"], \
        "started in the scan's second: this session's"


def test_repository_rereads_a_run_whose_process_is_gone(tmp_path, monkeypatch):
    root = live_root(tmp_path)
    status = root / "outputs/runs" / LIVE / "status.json"

    def exits_meanwhile(run, here):  # the runner writes its final status between the read and the liveness check
        patch_json(status, status="completed", finished="2026-10-23T14:20:00")
        return process.Live("LOST", "process 1 is gone")

    monkeypatch.setattr("xmr4el_tui.process.derived_state", exits_meanwhile)
    run = next(r for r in ExperimentRepository(root).snapshot().runs if r.id == LIVE)
    assert runs.state_word(run) == "COMPLETED"


# --- app: destinations, overlays, layout ----------------------------------------------------------------------------
def test_old_notice_never_opens_a_replacement_run(tmp_path):
    root = live_root(tmp_path)

    async def check(app, pilot):
        old = target(app, "monitor", LIVE)
        patch_json(root / "outputs/runs" / LIVE / "status.json", started="2026-10-23T15:00:00", pid=4242)
        await rescan_app(app, pilot)
        app.open_notice(notice(1, old))
        await pilot.pause()
        assert isinstance(app.screen, DashboardScreen) and "no longer found" in footer_note(app)
    run_app(root, check)


async def rescan_app(app, pilot):
    app.rescan()
    await settle(app, pilot)


def test_center_selection_follows_its_notice_when_new_ones_arrive(tmp_path):
    async def check(app, pilot):
        app.notices.add([notice(i) for i in range(6)])
        await pilot.press("ctrl+n")
        await pilot.pause()
        box = app.screen.query_one("#nc-list")
        await pilot.press("down", "down", "down")
        chosen, row_y, scroll = app.screen.ids[box.highlighted_option.id], box.highlighted, box.scroll_y
        app.toast(app.notices.add([notice(10), notice(11)]))
        await pilot.pause()
        assert app.screen.ids[box.highlighted_option.id] is chosen, "enter would open the chosen notice"
        assert box.highlighted == row_y + 2 and box.scroll_y == scroll + 8, "same row on screen (two 4-row notices)"
        opened = []
        app.open_notice = opened.append
        await pilot.press("enter")
        assert opened == [chosen]
    run_app(live_root(tmp_path), check, size=(120, 40))


@pytest.mark.parametrize("size", [(80, 24), (100, 30), (120, 36), (140, 40)])
def test_center_fits_the_terminal(tmp_path, size):
    async def check(app, pilot):
        app.notices.add([notice(i) for i in range(HISTORY)])
        await pilot.press("ctrl+n")
        await pilot.pause()
        box, foot = app.screen.query_one("#nc-box").region, app.screen.query_one("#nc-foot").region
        assert box.bottom <= size[1] and foot.bottom <= box.bottom - 1, "the whole box and its keys are visible"
        await pilot.resize_terminal(size[0], 24)
        await pilot.pause()
        assert app.screen.query_one("#nc-box").region.bottom <= 24
    run_app(live_root(tmp_path), check, size=size)


def test_toasts_are_bounded_and_one_at_a_time(tmp_path):
    async def check(app, pilot):
        long = replace(notice(1, severity="error"), text="Traceback " * 200)
        app.toast(app.notices.add([long]))
        for i in range(2, 6):  # rapid separate deliveries (manual rescans)
            app.toast(app.notices.add([notice(i)]))
        await pilot.pause()
        assert len(app._notifications) == 1, "one visible toast"
        toast = app.screen.query("Toast")
        assert len(toast) == 1 and toast[0].region.height <= 6
        assert app.notices.items[-1].text == long.text, "the full text stays in the history"
        await pilot.press("ctrl+c")
        app.toast(app.notices.add([replace(long, key=("x", "FAILED"))]))
        await pilot.pause()
        note = app.screen.query_one("#footer-note").region
        assert not any(t.region.overlaps(note) for t in app.screen.query("Toast")), "the confirmation stays readable"
    asyncio.run(_with_toasts(live_root(tmp_path), check, (80, 24)))


async def _with_toasts(root, test, size):
    app = XMR4ELApp(root)
    async with app.run_test(size=size, notifications=True) as pilot:
        await settle(app, pilot)
        await test(app, pilot)


def test_help_keeps_the_navbar_and_a_narrow_bar_keeps_notifications(tmp_path):
    async def check(app, pilot):
        w = app.size.width
        await pilot.press("question_mark")
        await pilot.pause()
        line = bar(app)
        assert line.index(TITLE) == (w - len(TITLE)) // 2 and "Notifications" in line, "help does not move them"
        help_panel = app.screen.query_one("HelpPanel").region
        assert help_panel.y == 2 and help_panel.bottom == app.size.height - 1, "between the navbar and the footer"
    for size in ((80, 24), (100, 30), (120, 36)):
        run_app(live_root(tmp_path / str(size[0])), check, size=size)

    async def narrow(app, pilot):
        app.notices.add([notice(i) for i in range(12)])
        app.refresh_bars()
        await pilot.pause()
        assert bar(app).rstrip().endswith("Notif 12")
        await pilot.click(TopBar, offset=(app.size.width - 4, 0))
        await pilot.pause()
        assert isinstance(app.screen, NotificationCenter)
    run_app(live_root(tmp_path / "narrow"), narrow, size=(40, 20))

"""Contextual footer: shortcuts, else feedback over the whole bar (feedback-row tint), else a confirmation (confirm-row
tint); confirmation > feedback > shortcuts, and feedback timers never touch a confirmation or a newer message."""
import asyncio

import pytest

textual = pytest.importorskip("textual")

from textual.color import Color  # noqa: E402
from textual.widgets import Footer  # noqa: E402

from xmr4el_tui import app as app_module, theme  # noqa: E402
from xmr4el_tui.app import XMR4ELApp  # noqa: E402
from xmr4el_tui.widgets.context_footer import ContextFooter  # noqa: E402


def state(app):
    bar = app.screen.query_one(ContextFooter)
    note = app.screen.query_one("#footer-note")
    return bar, note, app.screen.query_one(Footer)


def run(tmp_path, test, monkeypatch, seconds=0.3):
    monkeypatch.setattr(app_module, "FEEDBACK_SECONDS", seconds)
    (tmp_path / "outputs/runs").mkdir(parents=True)

    async def go():
        app = XMR4ELApp(tmp_path)
        async with app.run_test(size=(100, 24)) as pilot:
            await app.workers.wait_for_complete()
            await pilot.pause()
            await test(app, pilot)
    asyncio.run(go())


def test_feedback_replaces_every_shortcut_then_they_return(tmp_path, monkeypatch):
    async def go(app, pilot):
        bar, note, keys = state(app)
        assert keys.display and not note.display and not bar.has_class("-feedback")
        app.say("Dataset confirmed: A", "success")
        await pilot.pause()
        assert not keys.display and note.display and bar.has_class("-feedback")
        assert note.region.width == bar.region.width and "Dataset confirmed: A" in str(note.render())
        assert bar.styles.background == note.styles.background == Color.parse(theme.color("feedback-row"))
        assert theme.color("feedback-row") != theme.color("confirm-row")
        await pilot.pause(0.5)
        assert keys.display and not note.display and not bar.has_class("-feedback")
    run(tmp_path, go, monkeypatch)


def test_newer_feedback_supersedes_and_old_timer_is_harmless(tmp_path, monkeypatch):
    async def go(app, pilot):
        app.say("first")
        old = app.note
        await pilot.pause(0.2)
        app.say("second")
        app.clear_note(old)  # a stale expiry: no effect on the newer message
        await pilot.pause(0.15)  # the first timer would have fired by now
        _, note, keys = state(app)
        assert "second" in str(note.render()) and not keys.display
        await pilot.pause(0.3)
        assert keys.display and not note.display
    run(tmp_path, go, monkeypatch)


def test_confirmation_wins_over_feedback_and_survives_its_timer(tmp_path, monkeypatch):
    async def go(app, pilot):
        app.say("Copied 10 characters")
        await pilot.press("ctrl+c")
        bar, note, keys = state(app)
        assert "Quit XMR4EL?" in str(note.render()) and bar.has_class("-confirm") and not bar.has_class("-feedback")
        assert bar.styles.background == Color.parse(theme.color("confirm-row"))
        app.say("Arrived meanwhile")
        await pilot.pause(0.5)  # every feedback timer expires
        assert "Quit XMR4EL?" in str(note.render()) and not keys.display
        await pilot.press("escape")
        assert keys.display and not note.display and not bar.classes & {"-confirm", "-feedback"}
        await pilot.press("ctrl+n")  # shortcuts and notifications still work
        assert type(app.screen).__name__ == "NotificationCenter"
    run(tmp_path, go, monkeypatch)

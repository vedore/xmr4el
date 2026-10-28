"""Fullscreen, read-only log of one run (Dashboard Description > View Logs): a LogPanel; esc returns to the
Dashboard."""
from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import Screen

from xmr4el_tui.widgets.context_footer import ContextFooter
from xmr4el_tui.widgets.log_panel import LogPanel
from xmr4el_tui.widgets.log_view import LogView
from xmr4el_tui.widgets.top_bar import TopBar


class LogScreen(Screen):
    BINDINGS = [Binding("escape", "close", "back")]
    crumbs = ("Logs",)

    def __init__(self, run, root):
        super().__init__()
        self.panel = LogPanel(run, root, id="log-panel")

    def compose(self) -> ComposeResult:
        yield TopBar(id="top")
        yield self.panel
        yield ContextFooter()

    def on_mount(self):
        self.query_one(LogView).focus()

    def on_runs_changed(self, _):
        run = self.panel.run
        self.panel.set_run(next((r for r in self.app.snapshot.runs if r.id == run.id), run))

    def action_close(self):
        self.dismiss()

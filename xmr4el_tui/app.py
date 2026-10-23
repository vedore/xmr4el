"""Textual app: rescans outputs/runs every 5 s in a thread and tells the screens (tui.md §5, §7)."""
import socket
from pathlib import Path

from textual import work
from textual.app import App
from textual.binding import Binding

from xmr4el_tui import runs
from xmr4el_tui.screens import RunsChanged, RunsScreen

RESCAN_SECONDS = 5


class XMR4ELApp(App):
    CSS_PATH = "app.tcss"
    TITLE = "XMR4EL"
    BINDINGS = [Binding("question_mark", "help", "help"), Binding("f1", "help", "help", show=False)]

    def __init__(self, root: Path):
        super().__init__()
        self.root = root
        self.host = socket.gethostname()
        self.cache = {}  # only touched by the (exclusive) scan worker after the first scan
        self.runs, self.runs_found = self.scan()

    def scan(self):
        """(runs, outputs/runs exists)."""
        try:
            return runs.scan(self.root, self.cache), True
        except FileNotFoundError:
            return [], False

    def on_mount(self):
        self.push_screen(RunsScreen())
        self.set_interval(RESCAN_SECONDS, self.rescan)

    @work(thread=True, exclusive=True, group="scan")
    def rescan(self):
        found = self.scan()
        self.call_from_thread(self.apply_scan, found)

    def apply_scan(self, found):
        self.runs, self.runs_found = found
        for screen in self.screen_stack:
            screen.post_message(RunsChanged())

    def action_help(self):
        if self.screen.query("HelpPanel"):
            self.action_hide_help_panel()
        else:
            self.action_show_help_panel()

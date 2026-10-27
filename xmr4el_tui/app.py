"""Textual app: rescans outputs/runs every 5 s in a thread and tells the screens (tui.md §5, §7)."""
import json
import os
import socket
import time
from dataclasses import replace
from pathlib import Path

from textual import work
from textual.app import App
from textual.binding import Binding

from xmr4el_tui import procinfo, process, runs
from xmr4el_tui.dashboard import DashboardScreen, RunsChanged
from xmr4el_tui.progress import current_stage

RESCAN_SECONDS = 5
QUIT_SECONDS = 3  # second ctrl+q within this window quits


def settings_path():
    """Per-user TUI settings (theme), outside the repository."""
    return Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "xmr4el" / "tui.json"


class XMR4ELApp(App):
    CSS_PATH = "app.tcss"
    TITLE = "XMR4EL"
    BINDINGS = [Binding("question_mark", "help", "help"), Binding("f1", "help", "help", show=False),
                Binding("escape", "close_help", "close help", show=False, priority=True)]

    def __init__(self, root: Path):
        super().__init__()
        self.root = root
        self.host = socket.gethostname()
        self.quit_armed = 0.0
        self.cache = {}  # only touched by the (exclusive) scan worker after the first scan
        self.launched = []  # runs this TUI started, kept only so poll() reaps them
        self.runs, self.runs_found, self.trees, self.active = self.scan()

    def scan(self):
        """(runs, outputs/runs exists, saved trees, {run id: (current stage, device)} of recorded-running runs);
        recorded-running runs carry their live state (§9). Only reads files."""
        try:
            found = runs.scan(self.root, self.cache)
        except FileNotFoundError:
            return [], False, runs.trees(self.root, []), {}
        here = procinfo.runtime()
        found = [replace(r, live=process.derived_state(r, here)) if r.recorded == "running" else r for r in found]
        active = {r.id: (current_stage(r), device(r))
                  for r in found if r.recorded == "running"}
        return found, True, runs.trees(self.root, found), active

    def on_mount(self):
        try:
            theme = json.loads(settings_path().read_text()).get("theme")
        except (OSError, ValueError, AttributeError):
            theme = None
        if theme in self.available_themes:
            self.theme = theme
        self.theme_changed_signal.subscribe(self, self.save_theme)
        self.push_screen(DashboardScreen())
        self.set_interval(RESCAN_SECONDS, self.rescan)

    @work(thread=True, exclusive=True, group="scan")
    def rescan(self, announce=False):
        found = self.scan()
        self.call_from_thread(self.apply_scan, found, announce)

    def apply_scan(self, found, announce=False):
        self.runs, self.runs_found, self.trees, self.active = found
        self.launched = [c for c in self.launched if c.poll() is None]
        if announce:
            self.notify(f"Rescanned outputs/runs/: {len(self.runs)} runs" if self.runs_found
                        else "Rescanned: outputs/runs/ not found", timeout=2)
        for screen in self.screen_stack:
            screen.post_message(RunsChanged())

    async def action_quit(self):
        """ctrl+q twice: the first press only warns."""
        now = time.monotonic()
        if now - self.quit_armed < QUIT_SECONDS:
            self.exit()
        else:
            self.quit_armed = now
            self.notify("Press ctrl+q again to quit.", title="Quit?", severity="warning", timeout=QUIT_SECONDS)

    def save_theme(self, theme):
        path = settings_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"theme": theme.name}) + "\n")
        except OSError as exc:
            self.notify(f"Theme not saved: {exc}", severity="warning")

    def action_help(self):
        if self.screen.query("HelpPanel"):
            self.action_hide_help_panel()
        else:
            self.action_show_help_panel()
        self.refresh_bindings()

    def check_action(self, action, parameters):
        """esc closes help only while it is open; otherwise the screen's esc runs and shows in the footer."""
        return action != "close_help" or bool(self.screen.query("HelpPanel"))

    def action_close_help(self):
        self.action_help()


def device(run):
    prov = run.provenance()
    return prov.get("torch_device") if isinstance(prov, dict) else None

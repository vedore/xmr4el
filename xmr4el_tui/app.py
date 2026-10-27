"""Textual app: rescans outputs/runs every 5 s in a thread and tells the screens (tui.md §5, §7)."""
import time
from dataclasses import replace
from pathlib import Path

from textual import work
from textual.app import App
from textual.binding import Binding
from textual.theme import Theme

from xmr4el_tui import procinfo, process, runs
from xmr4el_tui.messages import RunsChanged
from xmr4el_tui.screens.dashboard_screen import DashboardScreen
from xmr4el_tui.progress import current_stage

RESCAN_SECONDS = 5
QUIT_SECONDS = 3  # second ctrl+q within this window quits

# The one style (no theme picker). Screens use only theme variables in app.tcss, so another Theme can be registered
# later without touching them. Status colours stay semantic (dashboard.STATE_STYLE).
THEME = Theme(
    name="xmr4el", dark=True, primary="#2CF5C5", accent="#2CF5C5", foreground="#F0F4F8", background="#101418",
    surface="#1B2229", panel="#1B2229",
    variables={"foreground-muted": "#A5B4C3", "border": "#2CF5C5", "border-blurred": "#3A4754",
               "block-cursor-background": "#2CF5C5", "block-cursor-foreground": "#101418",
               "block-cursor-text-style": "bold", "block-cursor-blurred-background": "#2B5C55",
               "block-cursor-blurred-foreground": "#F0F4F8", "block-cursor-blurred-text-style": "none",
               "block-hover-background": "#F0F4F8 8%", "footer-key-foreground": "#2CF5C5",
               "footer-description-foreground": "#A5B4C3",
               "screen-selection-background": "#3B6E99", "screen-selection-foreground": "#F0F4F8"})


class XMR4ELApp(App):
    CSS_PATH = "app.tcss"
    TITLE = "XMR4EL"
    BINDINGS = [Binding("question_mark", "help", "keys"), Binding("f1", "help", "keys", show=False),
                Binding("n", "new_experiment", "new"),
                Binding("ctrl+c", "quit", "quit", show=False, priority=True),
                Binding("escape", "close_help", "close help", show=False, priority=True)]

    def __init__(self, root: Path):
        super().__init__()
        self.root = root
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
        self.register_theme(THEME)
        self.theme = THEME.name
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
        """ctrl+c (or ctrl+q) twice: the first press only warns."""
        now = time.monotonic()
        if now - self.quit_armed < QUIT_SECONDS:
            self.exit()
        else:
            self.quit_armed = now
            self.notify("Press ctrl+c again to quit.", title="Quit?", severity="warning", timeout=QUIT_SECONDS)

    def on_text_selected(self, _):
        """Text selected with the mouse goes to the clipboard on release (OSC 52: the terminal running the TUI,
        also through ssh and docker exec)."""
        text = self.screen.get_selected_text()
        if text:
            self.copy_to_clipboard(text)
            self.notify(f"Copied {len(text)} characters", timeout=2)

    def get_system_commands(self, screen):
        """The palette without its theme picker: one style only."""
        yield from (c for c in super().get_system_commands(screen) if c.title != "Theme")

    def action_new_experiment(self):
        self.notify("New Experiment: screen not built yet")

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

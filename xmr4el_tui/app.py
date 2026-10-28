"""Textual app: rescans outputs/runs every 5 s in a thread and tells the screens (tui.md §5, §7)."""
import threading
import time

from textual import work
from textual.app import App
from textual.binding import Binding
from textual.message import Message
from textual.theme import Theme
from textual.worker import get_current_worker

from xmr4el_tui import process
from xmr4el_tui.messages import RunsChanged
from xmr4el_tui.screens.dashboard_screen import DashboardScreen
from xmr4el_tui.services.experiment_repository import ExperimentRepository, Snapshot

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


class Scanned(Message):
    """A scan finished (posted from the scan thread: post_message is thread-safe and never blocks on shutdown)."""

    def __init__(self, snapshot, announce, reaped=()):
        super().__init__()
        self.snapshot, self.announce, self.reaped = snapshot, announce, reaped


class XMR4ELApp(App):
    CSS_PATH = "app.tcss"
    TITLE = "XMR4EL"
    BINDINGS = [Binding("question_mark", "help", "keys"), Binding("f1", "help", "keys", show=False),
                Binding("n", "new_experiment", "new"),
                Binding("ctrl+c", "quit", "quit", show=False, priority=True),
                Binding("escape", "close_help", "close help", show=False, priority=True)]

    def __init__(self, root):
        super().__init__()
        self.root = root
        self.quit_armed = 0.0
        self.repo = ExperimentRepository(root)
        self.snapshot = Snapshot()  # empty until the first scan (started on mount) lands
        self.scan_lock = threading.Lock()  # one scan at a time: it owns the repository's cache
        self.scan_wanted = self.scan_announce = False  # requests the running scan picks up
        self.launches = process.Launches(root)

    def on_mount(self):
        self.register_theme(THEME)
        self.theme = THEME.name
        self.push_screen(DashboardScreen())
        self.rescan()
        self.set_interval(RESCAN_SECONDS, self.rescan)

    @work(thread=True, group="scan")
    def rescan(self, announce=False):
        """Scan in a thread; `announce` (ctrl+r) rereads every file and says so. A request while a scan runs is
        coalesced: the running scan goes once more, so one scan touches the cache and results land in scan order.
        A request is served by any scan starting after it."""
        self.scan_announce |= announce
        self.scan_wanted = True
        worker = get_current_worker()
        while self.scan_wanted and not worker.is_cancelled and self.scan_lock.acquire(blocking=False):
            try:  # re-checked after release: a request that lost the lock race is not dropped
                while self.scan_wanted and not worker.is_cancelled:
                    self.scan_wanted, announce, self.scan_announce = False, self.scan_announce, False
                    snapshot = self.repo.snapshot(fresh=announce)
                    reaped = self.launches.reap()  # reads status / console files: here, not on the UI thread
                    if not worker.is_cancelled:  # cancelled on quit: publish nothing
                        self.post_message(Scanned(snapshot, announce, reaped))
            finally:
                self.scan_lock.release()

    def on_scanned(self, message):
        self.apply_scan(message.snapshot, message.announce, message.reaped)

    def apply_scan(self, snapshot, announce=False, reaped=()):
        if snapshot.problem and snapshot.problem != self.snapshot.problem:
            self.notify(snapshot.problem, title="Scan failed", severity="warning")
        self.snapshot = snapshot
        for run_id, what in reaped:
            self.notify(f"{run_id}: {what}", title="Launch", severity="error")
        if announce:
            self.notify(f"Rescanned outputs/runs/: {len(snapshot.runs)} runs" if snapshot.found
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


"""Textual app: rescans outputs/runs every 5 s in a thread and tells the screens (tui.md §5, §7)."""
import threading
from datetime import datetime

from textual import work
from textual.app import App
from textual.binding import Binding
from textual.message import Message
from textual.screen import ModalScreen
from textual.theme import Theme
from textual.worker import get_current_worker

from xmr4el_tui import explorer, process, theme
from xmr4el_tui.messages import RunsChanged
from xmr4el_tui.runs import state_word
from xmr4el_tui.screens.dashboard_screen import DashboardScreen
from xmr4el_tui.screens.monitor_screen import RunMonitorScreen
from xmr4el_tui.screens.notification_center import NotificationCenter
from xmr4el_tui.screens.results_screen import ResultsScreen
from xmr4el_tui.services.experiment_repository import ExperimentRepository, Snapshot
from xmr4el_tui.services.notifications import Notice, Notifications, subject
from xmr4el_tui.widgets.context_footer import ContextFooter
from xmr4el_tui.widgets.top_bar import TopBar

RESCAN_SECONDS = 5
FEEDBACK_SECONDS = 3.5  # footer feedback, then the shortcuts again
TOAST_SECONDS = 4       # below RESCAN_SECONDS: one scan's toast is gone before the next scan's
WORST_FIRST = ("error", "warning", "success", "information")

# The one style (no theme picker). Every colour comes from theme.TOKENS; screens use only theme variables in app.tcss,
# so another theme needs no CSS change.
C = theme.TOKENS
THEME = Theme(
    name="xmr4el", dark=True, primary=C["accent"], accent=C["accent"], foreground=C["text"],
    background=C["background"], surface=C["surface"], panel=C["panel"], success=C["success"],
    warning=C["warning"], error=C["error"],
    variables={"foreground-muted": C["text-muted"], "border": C["accent"], "border-blurred": C["panel-border"],
               "block-cursor-background": C["accent"], "block-cursor-foreground": C["on-accent"],
               "block-cursor-text-style": "bold", "block-cursor-blurred-background": C["cursor-blurred"],
               "block-cursor-blurred-foreground": C["text"], "block-cursor-blurred-text-style": "none",
               "block-hover-background": f"{C['text']} 8%", "footer-key-foreground": C["accent"],
               "footer-description-foreground": C["text-muted"],
               "screen-selection-background": C["text-selection"], "screen-selection-foreground": C["text"],
               **theme.css_variables()})


class Scanned(Message):
    """A scan finished (posted from the scan thread: post_message is thread-safe and never blocks on shutdown)."""

    def __init__(self, snapshot, announce, reaped=()):
        super().__init__()
        self.snapshot, self.announce, self.reaped = snapshot, announce, reaped


class XMR4ELApp(App):
    CSS_PATH = "app.tcss"
    TITLE = "XMR4EL"
    BINDINGS = [Binding("question_mark", "help", "keys"), Binding("f1", "help", "keys", show=False),
                Binding("n", "new_experiment", "new"), Binding("ctrl+n", "notifications", "notifications"),
                Binding("ctrl+c", "quit", "quit", show=False, priority=True),
                Binding("escape", "escape", "cancel", show=False, priority=True)]

    def __init__(self, root):
        super().__init__()
        self.root = root
        self.notices = Notifications()  # this session's notifications (memory only)
        self.note, self.note_timer = None, None  # footer feedback (severity, text, screen) and its expiry
        self.confirming = None  # the screen whose footer asks to confirm quitting
        self.repo = ExperimentRepository(root)
        self.snapshot = Snapshot()  # empty until the first scan (started on mount) lands
        self.scan_lock = threading.Lock()  # one scan at a time: it owns the repository's cache
        self.scan_wanted = self.scan_announce = False  # requests the running scan picks up
        self.launches = process.Launches(root)

    def get_theme_variable_defaults(self):
        """The theme tokens ($xmr-...) exist before THEME is applied (app.tcss is parsed first)."""
        return theme.css_variables()

    def on_mount(self):
        self.register_theme(THEME)
        self.theme = THEME.name
        self.screen_change_signal.subscribe(self, self.screen_changed)
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
        """New snapshot to every screen; ended runs, failed launches and a new scan problem become notifications."""
        new = []
        if snapshot.problem and snapshot.problem != self.snapshot.problem:
            new.append(Notice(("scan", snapshot.problem), "warning", "Scan failed", "outputs/runs/", snapshot.problem,
                              datetime.now(), None))
        by_id = {r.id: r for r in snapshot.runs}
        for run_id, what in reaped:  # same key as the LOST transition the scan may also see: reported once
            run = by_id.get(run_id)
            new.append(Notice((run_id, "LOST"), "error" if what.startswith("not started") else "warning",
                              "Launch failed" if what.startswith("not started") else "Run interrupted",
                              subject(run, snapshot.runs) if run else run_id, what, datetime.now(),
                              ("monitor", run_id) if run else None))
        self.snapshot = snapshot
        self.toast(self.notices.add(new) + self.notices.observe(snapshot.runs))
        if announce:
            self.say(f"Rescanned outputs/runs/: {len(snapshot.runs)} runs" if snapshot.found
                     else "Rescanned: outputs/runs/ not found")
        for screen in self.screen_stack:
            screen.post_message(RunsChanged())

    # --- notifications ------------------------------------------------------------------------------------------
    def toast(self, new):
        """One toast per scan (several events: one summary), gone before the next scan's; none over the open
        notification center, which shows them in its list instead. A toast never takes focus."""
        if not new:
            return
        if isinstance(self.screen, NotificationCenter):
            self.screen.fill()
            return
        self.refresh_bars()
        if len(new) == 1:
            n = new[0]
            title, text, severity = n.title, "\n".join(p for p in (n.subject, n.text) if p), n.severity
        else:
            title, text = "Notifications", f"{len(new)} experiment events.\nctrl+n opens notifications."
            severity = next(s for s in WORST_FIRST if any(n.severity == s for n in new))
        self.notify(text, title=f" {title} ", severity=severity, timeout=TOAST_SECONDS, markup=False)

    def action_notifications(self):
        if isinstance(self.screen, NotificationCenter):
            self.screen.dismiss()
        elif not isinstance(self.screen, ModalScreen):  # not over a dialog
            self.push_screen(NotificationCenter())

    def open_notice(self, notice):
        """A notification's destination, opened from the Dashboard (the screens above it are closed). One that no
        longer exists opens nothing and says so."""
        kind, ref = notice.target or (None, None)
        snap = self.snapshot
        if kind == "results":
            node = explorer.index(explorer.build(snap.runs, snap.trees)).get(ref, (None,))[0]
            ok = node is not None and node.kind == "eval" and bool(node.ref[1].metrics)
        else:
            ok = kind == "monitor" and any(r.id == ref for r in snap.runs)
        if not ok:
            if kind:
                self.say("That evaluation is no longer found." if kind == "results"
                         else "That run directory is no longer found.", "warning")
            return
        dashboard = next(s for s in self.screen_stack if isinstance(s, DashboardScreen))
        self.go_to(dashboard, 0)
        self.push_screen(ResultsScreen(ref) if kind == "results" else RunMonitorScreen(ref))

    # --- navbar and footer ----------------------------------------------------------------------------------------
    def go_to(self, screen, level):
        """Back to breadcrumb `level` of `screen` (navbar click): the screens above it close as with esc."""
        while self.screen is not screen:
            self.pop_screen()
        if level < len(screen.crumbs) - 1:
            screen.back_to_crumb(level)

    def screen_changed(self, screen):
        """Feedback and a pending confirmation belong to the screen they were shown on: navigating drops them (the
        signal can come after a pop completes, when feedback was already given on the new screen: that one stays)."""
        if self.note and self.note[2] is not screen:
            self.note = None
        if self.confirming is not screen:
            self.confirming = None
        self.refresh_bars()

    def refresh_bars(self):
        for screen in self.screen_stack:
            for bar in screen.query(TopBar):
                bar.refresh()
            for footer in screen.query(ContextFooter):
                footer.update_state()

    def say(self, text, severity="information", screen=None):
        """Footer feedback on `screen` (the current one) for FEEDBACK_SECONDS; a newer one replaces it at once and
        the older timer cannot clear it. Never shown over a confirmation, never kept in the notifications."""
        if self.note_timer is not None:
            self.note_timer.stop()
        note = self.note = (severity, text, screen or self.screen)
        self.note_timer = self.set_timer(FEEDBACK_SECONDS, lambda: self.clear_note(note))
        self.refresh_bars()

    def clear_note(self, note):
        if self.note is note:
            self.note = None
            self.refresh_bars()

    def quit_question(self):
        """Runs are detached (process.spawn: own session, output to files): quitting stops none of them."""
        active = sum(state_word(r) in explorer.ACTIVE for r in self.snapshot.runs)
        return "Quit XMR4EL?" + (f" {active} active run{'s' * (active > 1)} will keep running." if active else "")

    async def action_quit(self):
        """ctrl+c (or ctrl+q) asks in the footer; a second ctrl+c quits, esc cancels. No timer."""
        if self.confirming is self.screen:
            self.exit()
            return
        if isinstance(self.screen, ModalScreen):  # a dialog or the notification center: closed (cancelled) first
            self.screen.dismiss()
        self.confirming = self.screen
        self.refresh_bars()

    def on_text_selected(self, _):
        """Text selected with the mouse goes to the clipboard on release (OSC 52: the terminal running the TUI,
        also through ssh and docker exec)."""
        text = self.screen.get_selected_text()
        if text:
            self.copy_to_clipboard(text)
            self.say(f"Copied {len(text)} characters")

    def get_system_commands(self, screen):
        """The palette without its theme picker: one style only."""
        yield from (c for c in super().get_system_commands(screen) if c.title != "Theme")

    def action_new_experiment(self):
        self.say("New Experiment: screen not built yet.")

    def action_help(self):
        if self.screen.query("HelpPanel"):
            self.action_hide_help_panel()
        else:
            self.action_show_help_panel()
        self.refresh_bindings()

    def check_action(self, action, parameters):
        """esc cancels a quit confirmation, else closes help; otherwise the screen's esc runs and shows in the
        footer."""
        return action != "escape" or self.confirming is self.screen or bool(self.screen.query("HelpPanel"))

    def action_escape(self):
        if self.confirming is self.screen:  # cancelled: feedback given meanwhile is dropped too
            self.note = self.confirming = None
            self.refresh_bars()
        else:
            self.action_help()


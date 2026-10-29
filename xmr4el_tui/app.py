"""Textual app: rescans outputs/runs every 5 s in a thread and tells the screens (tui.md §5, §7)."""
import threading
from datetime import datetime

from textual import events, work
from textual.app import App
from textual.binding import Binding
from textual.message import Message
from textual.screen import ModalScreen
from textual.theme import Theme
from textual.worker import get_current_worker

from xmr4el_tui import explorer, process, theme
from xmr4el_tui.draft import ExperimentDraft
from xmr4el_tui.messages import RunsChanged
from xmr4el_tui.runs import state_word
from xmr4el_tui.screens.dashboard_screen import DashboardScreen
from xmr4el_tui.screens.dataset_setup import DatasetSetupScreen
from xmr4el_tui.screens.monitor_screen import RunMonitorScreen
from xmr4el_tui.screens.notification_center import NotificationCenter
from xmr4el_tui.screens.results_screen import ResultsScreen
from xmr4el_tui.services.experiment_repository import ExperimentRepository, Snapshot
from xmr4el_tui.services.notifications import Notice, Notifications, incarnation, subject
from xmr4el_tui.widgets.context_footer import ContextFooter
from xmr4el_tui.widgets.top_bar import TopBar

RESCAN_SECONDS = 5
FEEDBACK_SECONDS = 3.5  # footer feedback, then the shortcuts again
TOAST_SECONDS = 4       # below RESCAN_SECONDS: one scan's toast is gone before the next scan's
TOAST_LINE = 88         # characters per toast line: two wrapped rows of the 46-column toast
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
                Binding("n", "new_experiment", "new", show=False), Binding("ctrl+n", "notifications", "notifications"),
                Binding("ctrl+c", "quit", "quit", show=False, priority=True),
                Binding("escape", "escape", "cancel", show=False, priority=True)]

    def __init__(self, root):
        super().__init__()
        self.root = root
        self.notices = Notifications()  # this session's notifications (memory only)
        self.note, self.note_timer = None, None  # footer feedback (severity, text, screen) and its expiry
        self.confirming = None  # the screen whose footer asks to confirm quitting (or `asking`)
        self.asking = None      # (question, key, verb) of a screen's footer confirmation; None: the quit question
        self.draft = ExperimentDraft()  # New Experiment, kept for the session
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
        new, at = [], snapshot.at or datetime.now()
        by_id = {r.id: r for r in snapshot.runs}
        for run_id, what in reaped:  # same key as the LOST transition the scan may also see: reported once
            run = by_id.get(run_id)
            inc = incarnation(run) if run else run_id
            new.append(Notice((inc, "LOST"), "error" if what.startswith("not started") else "warning",
                              "Launch failed" if what.startswith("not started") else "Run interrupted",
                              subject(run, snapshot.runs) if run else run_id, what, at,
                              ("monitor", run_id, inc) if run else None))
        self.snapshot = snapshot
        self.toast(self.notices.add(new) + self.notices.observe(snapshot.runs, at, snapshot.problem))
        if announce:
            self.say(f"Rescanned outputs/runs/: {len(snapshot.runs)} runs" if snapshot.found
                     else "Rescanned: outputs/runs/ not found")
        for screen in self.screen_stack:
            screen.post_message(RunsChanged())

    # --- notifications ------------------------------------------------------------------------------------------
    def toast(self, new):
        """One toast at a time (several events: one summary; a newer delivery replaces the shown one), each line cut to
        TOAST_LINE characters; none over the open notification center, which lists them instead. The full text stays
        in the center. A toast never takes focus."""
        if not new:
            return
        if isinstance(self.screen, NotificationCenter):
            self.screen.fill()
            return
        self.refresh_bars()
        if len(new) == 1:
            n = new[0]
            title, lines, severity = n.title, (n.subject, n.text), n.severity
        else:
            title, lines = "Notifications", (f"{len(new)} experiment events.", "ctrl+n opens notifications.")
            severity = next(s for s in WORST_FIRST if any(n.severity == s for n in new))
        text = "\n".join(p if len(p) <= TOAST_LINE else p[:TOAST_LINE - 1] + "…" for p in lines if p)
        self.call_later(self.clear_notifications)  # queued like notify: also clears toasts still in the queue
        self.notify(text, title=f" {title} ", severity=severity, timeout=TOAST_SECONDS, markup=False)

    def action_notifications(self):
        if isinstance(self.screen, NotificationCenter):
            self.screen.dismiss()
        elif not isinstance(self.screen, ModalScreen):  # not over a dialog; the toast would cover the list
            self.call_later(self.clear_notifications)
            self.push_screen(NotificationCenter())

    def open_notice(self, notice):
        """A notification's destination, over the current screen (esc returns to it as it was; an open comparison is
        never replaced). Opened only if the latest scan still has the same run behind it (not a new run reusing its
        directory name); else nothing opens and the footer says so."""
        kind, ref, inc = notice.target or (None, None, None)
        snap = self.snapshot
        if kind == "results":
            node = explorer.index(explorer.build(snap.runs, snap.trees)).get(ref, (None,))[0]
            ok = node is not None and node.kind == "eval" and bool(node.ref[1].metrics) \
                and incarnation(node.ref[0]) == inc
        else:
            ok = kind == "monitor" and any(r.id == ref and incarnation(r) == inc for r in snap.runs)
        if not ok:
            if kind:
                self.say("That evaluation is no longer found." if kind == "results"
                         else "That run directory is no longer found.", "warning")
            return
        if kind == "monitor" and isinstance(self.screen, RunMonitorScreen) and self.screen.run.id == ref:
            return  # already open
        self.push_screen(ResultsScreen(ref) if kind == "results" else RunMonitorScreen(ref))

    # --- navbar and footer ----------------------------------------------------------------------------------------
    def go_to(self, screen, level):
        """Back to breadcrumb `level` of `screen` (navbar click): the screens above it close as with esc. A screen
        holding unapplied changes (`leave_guard`) asks first; the navigation resumes if they are discarded."""
        while self.screen is not screen:
            guard = getattr(self.screen, "leave_guard", None)
            if guard is not None and guard(lambda: self.go_to(screen, level)):
                return
            self.pop_screen()
        if level < len(screen.crumbs) - 1:
            screen.back_to_crumb(level)

    def screen_changed(self, screen):
        """Feedback and a pending confirmation belong to the screen they were shown on: navigating drops them (the
        signal can come after a pop completes, when feedback was already given on the new screen: that one stays)."""
        if self.note and self.note[2] is not screen:
            self.note = None
        if self.confirming is not screen:
            self.confirming = self.asking = None
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
        if self.confirming is self.screen and self.asking is None:
            self.exit()
            return
        if isinstance(self.screen, ModalScreen):  # a dialog or the notification center: closed (cancelled) first
            self.screen.dismiss()
        self.confirming, self.asking = self.screen, None
        self.refresh_bars()

    async def on_event(self, event):
        """A footer confirmation is modal: until it is answered only its key, esc and ctrl+c / ctrl+q reach the
        screen; every other key, paste and mouse press, release, click or scroll is dropped (mouse moves pass)."""
        if self.confirming is not None and self.screen_stack and self.confirming is self.screen_stack[-1] \
                and isinstance(event, events.InputEvent) \
                and not event.is_forwarded \
                and not self.confirm_allows(event):
            return
        await super().on_event(event)

    def confirm_allows(self, event):
        if isinstance(event, events.Key):
            return event.key in ("escape", "ctrl+c", "ctrl+q") or bool(self.asking and event.key == self.asking[1])
        return isinstance(event, events.MouseMove)

    def ask(self, question, key, verb):
        """The quit convention for a screen's own action: the footer asks `question`; `key` again does it (the screen
        checks `asked`), esc cancels."""
        self.confirming, self.asking = self.screen, (question, key, verb)
        self.refresh_bars()

    def asked(self, key):
        """True (and the question closed) when the current screen's footer was asking for `key`."""
        if self.confirming is self.screen and self.asking and self.asking[1] == key:
            self.confirming = self.asking = None
            self.refresh_bars()
            return True
        return False

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
        """New Experiment (Dataset Setup) over the current screen; already open: back to it. Not over a dialog."""
        open_ = next((s for s in self.screen_stack if isinstance(s, DatasetSetupScreen)), None)
        if open_ is not None:
            self.go_to(open_, 0)
        elif not isinstance(self.screen, ModalScreen):
            self.push_screen(DatasetSetupScreen())

    def action_help(self):
        if self.screen.query("HelpPanel"):
            self.action_hide_help_panel()
        else:
            self.action_show_help_panel()
            self.call_after_refresh(self.fit_help)
        self.refresh_bindings()

    def on_resize(self):
        self.fit_help()

    def fit_help(self):
        """The docked help panel spans the rows between the navbar (2) and the footer (1): a dock's margin is not taken
        from its height, so it is set here."""
        for screen in self.screen_stack:
            for panel in screen.query("HelpPanel"):
                panel.styles.height = max(1, self.size.height - 3)

    def check_action(self, action, parameters):
        """esc cancels a quit confirmation, else closes help; otherwise the screen's esc runs and shows in the
        footer."""
        return action != "escape" or self.confirming is self.screen or bool(self.screen.query("HelpPanel"))

    def action_escape(self):
        if self.confirming is self.screen:  # cancelled: feedback given meanwhile is dropped too
            self.note = self.confirming = self.asking = None
            self.refresh_bars()
            getattr(self.screen, "confirm_cancelled", lambda: None)()
        else:
            self.action_help()


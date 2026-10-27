"""Fullscreen, read-only log of one run (Dashboard Description > View Logs). Polls the log every POLL_SECONDS in a
worker while the run is active, reads it once when finished; esc returns to the Dashboard."""
from rich.text import Text
from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.message import Message
from textual.screen import Screen
from textual.widgets import Footer, Input, Static

from xmr4el_tui import explorer, fmt
from xmr4el_tui.logs import LogTail, log_path
from xmr4el_tui.runs import state_word
from xmr4el_tui.screens.dashboard_details import state_text
from xmr4el_tui.widgets.log_view import LogView
from xmr4el_tui.widgets.top_bar import TopBar

POLL_SECONDS = 1


class LogRead(Message):
    """One LogTail.read result (posted from the reading worker)."""

    def __init__(self, lines, reset, error, final):
        super().__init__()
        self.lines, self.reset, self.error, self.final = lines, reset, error, final


class LogScreen(Screen):
    BINDINGS = [Binding("escape", "close", "back"), Binding("slash", "search", "search"),
                Binding("n", "match(1)", "next"), Binding("N", "match(-1)", "previous"),
                Binding("l", "level", "level")]

    def __init__(self, run, root):
        super().__init__()
        self.run, self.path = run, log_path(root, run)
        self.tail, self.reading, self.timer = LogTail(self.path), False, None

    def compose(self) -> ComposeResult:
        yield TopBar("Dashboard", id="top")
        yield Static(id="log-header")
        yield LogView(id="log-view")
        yield Static(id="log-missing")
        yield Input(placeholder="search (enter: find, esc: cancel)", id="log-search")
        yield Footer(show_command_palette=False)

    def on_mount(self):
        self.query_one("#log-search").display = False
        self.query_one("#log-missing").display = False
        self.query_one(LogView).border_title = f"Log · {self.path}"
        self.query_one(LogView).focus()
        if self.active():
            self.timer = self.set_interval(POLL_SECONDS, self.tick)
        self.tick()
        self.header()

    def active(self):
        return state_word(self.run) in explorer.ACTIVE

    # --- reading ---------------------------------------------------------------------------------------------
    def tick(self):
        if not self.reading:  # one read at a time: the LogTail is not shared between threads
            self.reading = True
            self.load(not self.active())

    @work(thread=True, group="log")
    def load(self, final):
        lines, reset = self.tail.read(final)
        self.post_message(LogRead(lines, reset, self.tail.error, final))

    def on_log_read(self, message):
        self.reading = False
        view, missing = self.query_one(LogView), self.query_one("#log-missing")
        if message.lines or message.reset:
            view.append(message.lines, message.reset)
        empty = message.error is not None and not view.all
        view.display, missing.display = not empty, empty
        if empty:
            missing.update(Text(f"No log to show: {self.path} ({message.error}). Nothing is generated here; the "
                                "runner writes run.log from the start of a run.", style="yellow"))
        if message.final and self.timer:
            self.timer.stop()
            self.timer = None
        self.header(message.error)

    def on_runs_changed(self, _):
        self.run = next((r for r in self.app.snapshot.runs if r.id == self.run.id), self.run)
        self.header(self.tail.error)

    # --- header and keys -------------------------------------------------------------------------------------
    def header(self, error=None):
        view = self.query_one(LogView)
        t = Text.assemble((f" {self.run.name}", "bold"), f"  {self.run.id}  ", state_text(state_word(self.run)))
        if self.tail.skipped:
            t.append(f"  (first {fmt.count(self.tail.skipped)} bytes not loaded)", style="dim")
        if error and view.all:
            t.append(f"  log {error}", style="yellow")
        t.append(f"  level {view.min_level or 'all'}" + (" · following" if self.timer else ""), style="dim")
        if view.query:
            at = f"{view.current + 1}/{len(view.matches)}" if view.current is not None else f"0/{len(view.matches)}"
            t.append(f"  “{view.query}” {at}", style="bold" if view.matches else "yellow")
        self.query_one("#log-header", Static).update(t)

    def action_close(self):
        search = self.query_one("#log-search", Input)
        if search.display:
            search.display = False
            self.query_one(LogView).focus()
        else:
            self.dismiss()

    def action_search(self):
        search = self.query_one("#log-search", Input)
        search.display = True
        search.value = self.query_one(LogView).query
        search.focus()

    def on_input_submitted(self, event):
        view = self.query_one(LogView)
        view.search(event.value)
        event.input.display = False
        view.focus()
        self.header(self.tail.error)

    def action_match(self, direction):
        self.query_one(LogView).step(direction)
        self.header(self.tail.error)

    def action_level(self):
        self.query_one(LogView).cycle_level()
        self.header(self.tail.error)

    def check_action(self, action, parameters):
        """Typing in the search box is text, not n / N / l / slash."""
        return not (action in ("match", "level", "search") and isinstance(self.focused, Input))

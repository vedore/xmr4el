"""A run's log, read-only: header line, LogView, search box. One LogTail per panel, polled every POLL_SECONDS in a
worker while the run is active and read once when it is finished. Used by the fullscreen log viewer (Dashboard View
Logs) and the Run monitor's Live logs panel."""
from rich.text import Text
from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.message import Message
from textual.widgets import Input, Static

from xmr4el_tui import explorer, fmt
from xmr4el_tui.fmt import STATE_STYLE
from xmr4el_tui.logs import LogTail, log_path
from xmr4el_tui.runs import state_word
from xmr4el_tui.widgets.log_view import LogView

POLL_SECONDS = 1


class LogRead(Message):
    """One LogTail.read result (posted from the reading worker)."""

    def __init__(self, lines, reset, error, final):
        super().__init__()
        self.lines, self.reset, self.error, self.final = lines, reset, error, final


class LogPanel(Vertical):
    """`/` search (enter finds, esc closes the box), `n` / `N` next / previous match, `l` level filter. `named`: the
    header starts with the run's name, id and state (the fullscreen viewer; the monitor shows them elsewhere)."""
    ALLOW_MAXIMIZE = True  # the Run monitor's fullscreen log
    BINDINGS = [Binding("slash", "search", "search"), Binding("n", "match(1)", "next"),
                Binding("N", "match(-1)", "previous"), Binding("l", "level", "level"),
                Binding("escape", "close_search", "close search", show=False)]

    def __init__(self, run, root, named=True, **kwargs):
        super().__init__(**kwargs)
        self.run, self.named, self.root = run, named, root
        self.path = self.tail = None  # chosen by the first read, in the worker (it stats files)
        self.reading, self.timer, self.again = False, None, False

    def compose(self) -> ComposeResult:
        yield Static(id="log-header")
        yield LogView(id="log-view")
        yield Static(id="log-missing")
        yield Input(placeholder="search (enter: find, esc: cancel)", id="log-search")

    def on_mount(self):
        self.query_one("#log-search").display = False
        self.query_one("#log-missing").display = False
        if self.active():
            self.timer = self.set_interval(POLL_SECONDS, self.tick)
        self.tick()
        self.header()

    def active(self):
        return state_word(self.run) in explorer.ACTIVE

    def set_run(self, run):
        """The run as the latest scan sees it; once finished, the next tick is the final read."""
        self.run = run
        self.header()

    # --- reading ---------------------------------------------------------------------------------------------
    def tick(self):
        """One read at a time (the LogTail is not shared between threads); a tick during a read is not dropped.
        A read started after the run ended is final: the runner logged everything before its final status."""
        if self.reading:
            self.again = True
        else:
            self.reading = True
            self.load(not self.active())

    @work(thread=True, group="log")
    def load(self, final):
        if self.tail is None:
            self.path = log_path(self.root, self.run)
            self.tail = LogTail(self.path)
        lines, reset = self.tail.read(final)
        self.post_message(LogRead(lines, reset, self.tail.error, final))

    def on_log_read(self, message):
        message.stop()
        self.reading = False
        if self.named:  # the Run monitor titles its panel itself
            self.border_title = f"Log · {self.path}"
        view, missing = self.query_one(LogView), self.query_one("#log-missing", Static)
        if message.lines or message.reset:
            view.append(message.lines, message.reset)
        empty = message.error is not None and not view.all
        view.display, missing.display = not empty, empty
        if empty:
            missing.update(Text(f"No log to show: {self.path} ({message.error}). Nothing is generated here; the "
                                "runner writes run.log from the start of a run.", style="yellow"))
        if self.again:
            self.again = False
            self.tick()
        elif message.final and self.timer:
            self.timer.stop()
            self.timer = None
        self.header()

    # --- header and keys -------------------------------------------------------------------------------------
    def header(self):
        view, error = self.query_one(LogView), self.tail and self.tail.error
        t = Text()
        if self.named:
            word = state_word(self.run)
            t.append_text(Text.assemble((f" {self.run.name}", "bold"), f"  {self.run.id}  ",
                                        (word, STATE_STYLE[word]), "  "))
        if self.tail and self.tail.skipped:
            t.append(f"(first {fmt.count(self.tail.skipped)} bytes not loaded)  ", style="dim")
        if error and view.all:
            t.append(f"log {error}  ", style="yellow")
        t.append(f"level {view.min_level or 'all'}" + (" · following" if self.timer else ""), style="dim")
        if view.query:
            at = f"{view.current + 1}/{len(view.matches)}" if view.current is not None else f"0/{len(view.matches)}"
            t.append(f"  “{view.query}” {at}", style="bold" if view.matches else "yellow")
        self.query_one("#log-header", Static).update(t)

    def action_close_search(self):
        self.query_one("#log-search", Input).display = False
        self.query_one(LogView).focus()

    def action_search(self):
        search = self.query_one("#log-search", Input)
        search.display = True
        search.value = self.query_one(LogView).query
        search.focus()

    def on_input_submitted(self, event):
        event.stop()
        view = self.query_one(LogView)
        view.search(event.value)
        event.input.display = False
        view.focus()
        self.header()

    def action_match(self, direction):
        self.query_one(LogView).step(direction)
        self.header()

    def action_level(self):
        self.query_one(LogView).cycle_level()
        self.header()

    def check_action(self, action, parameters):
        """Typing in the search box is text, not n / N / l / slash; esc here only closes an open search box (else
        the screen's esc runs)."""
        if action == "close_search":
            return self.query_one("#log-search").display
        return not (action in ("match", "level", "search") and isinstance(self.screen.focused, Input))

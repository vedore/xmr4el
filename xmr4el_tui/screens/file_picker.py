"""Local file picker (tui.md §5.13): a file on the machine running the TUI (over ssh / docker exec: the server's).

One path field over the entries of its directory. Typing filters them (the text after the last /); a typed or pasted
path to a directory opens it, to a file chooses it. up / down move in the list, enter opens the directory or chooses
the file under the cursor, `..` goes up. esc cancels: the caller's mapping is unchanged. Returns the path as recorded
(relative to the project root when inside it), or None.
"""
import os
from pathlib import Path

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Input
from textual.widgets.option_list import Option

from xmr4el_tui import fmt, theme
from xmr4el_tui.datasets import resolve, shown
from xmr4el_tui.widgets.context_footer import ContextFooter
from xmr4el_tui.widgets.description import Rows
from xmr4el_tui.widgets.top_bar import TopBar

LIKELY = (".pubtator", ".txt")  # shown normally; other files are dimmed, still selectable (a PubTator file can be .txt)
MAX_ENTRIES = 2000  # ponytail: a huge directory lists its first 2000 matches; filtering narrows it


class FilePicker(Screen):
    BINDINGS = [Binding("escape", "cancel", "cancel"), Binding("up", "move(-1)", show=False),
                Binding("down", "move(1)", show=False)]

    def __init__(self, root, title, current=None):
        """`title`: what is chosen ("Training file"); `current`: the mapping's path, whose directory opens first
        (else datasets/, else the project root)."""
        super().__init__()
        self.root, self.title, self.entries = Path(root), title, []
        self.current = resolve(root, current) if current else None  # highlighted when its directory is listed
        start = resolve(root, current).parent if current else Path(root, "datasets")
        self.start = start if start.is_dir() else Path(root)

    @property
    def crumbs(self):
        return (self.title,)

    def compose(self) -> ComposeResult:
        yield TopBar(id="top")
        with Vertical(id="fp"):
            yield Input(placeholder="Path, or text to filter", id="fp-path")
            yield Rows(id="fp-list")
        yield ContextFooter()

    def on_mount(self):
        self.query_one("#fp").border_title = f"Choose {self.title[0].lower()}{self.title[1:]}"
        self.open_dir(self.start)
        self.query_one(Input).focus()

    def text_of(self, directory):
        rel = shown(self.root, directory)
        return "" if rel == "." else rel.rstrip("/") + "/"

    def open_dir(self, directory):
        """Show `directory` (the path field ends with /: no filter)."""
        field = self.query_one(Input)
        field.value = self.text_of(directory)
        field.cursor_position = len(field.value)

    def on_input_changed(self, event):
        text = event.value
        where, query = (text, "") if text.endswith("/") or not text else os.path.split(text)
        self.list_dir(Path(os.path.normpath(resolve(self.root, where))), query)

    def list_dir(self, directory, query):
        self.cwd, box = directory, self.query_one(Rows)
        try:
            found = sorted(os.scandir(directory), key=lambda e: (not e.is_dir(), e.name.casefold()))
        except OSError as err:
            self.entries = []
            box.set_options([Option(Text(f"Cannot open {shown(self.root, directory)}: {err.strerror}",
                                         theme.style("error")), disabled=True)])
            return
        q = query.casefold()
        found = [e for e in found if q in e.name.casefold() and (q.startswith(".") or not e.name.startswith("."))]
        up = [] if query or directory.parent == directory else [None]
        self.entries = up + found[:MAX_ENTRIES]
        box.set_options([Option(self.row(e)) for e in self.entries] or
                        [Option(Text("No matching entries", theme.style("text-muted")), disabled=True)])
        names = [e and Path(e.path) for e in self.entries]
        box.highlighted = (names.index(self.current) if self.current in names else 0) if self.entries else None
        box.scroll_home(animate=False)

    def row(self, entry):
        if entry is None:
            return Text("../", theme.style("text-muted"))
        try:
            is_dir = entry.is_dir()
            size = None if is_dir else entry.stat().st_size
        except OSError:
            is_dir, size = False, None
        if is_dir:
            return Text(entry.name + "/")
        likely = entry.name.lower().endswith(LIKELY)
        return Text.assemble((entry.name, "" if likely else theme.style("text-disabled")), "  ",
                             (fmt.size(size), theme.style("text-muted")))

    def action_move(self, step):
        box = self.query_one(Rows)
        if self.entries:
            box.highlighted = max(0, min(len(self.entries) - 1, (box.highlighted or 0) + step))

    def on_input_submitted(self, event):
        """A typed path to a file chooses it, to another directory opens it; else the highlighted entry."""
        path = Path(os.path.normpath(resolve(self.root, event.value or ".")))
        if path.is_file():
            self.dismiss(shown(self.root, path))
        elif path.is_dir() and path != self.cwd:
            self.open_dir(path)
        elif self.entries and self.query_one(Rows).highlighted is not None:
            self.activate(self.entries[self.query_one(Rows).highlighted])
        else:
            self.app.say(f"Not found: {event.value}", "warning")

    def on_option_list_option_selected(self, event):
        self.activate(self.entries[event.option_index])
        self.query_one(Input).focus()

    def activate(self, entry):
        target = self.cwd.parent if entry is None else Path(entry.path)
        if target.is_dir():
            self.open_dir(target)
        elif target.is_file():
            self.dismiss(shown(self.root, target))
        else:
            self.app.say(f"No longer found: {shown(self.root, target)}", "warning")

    def action_cancel(self):
        self.dismiss(None)

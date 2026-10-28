"""Read-only Description panel: one selectable option per line, headings skipped."""
from textual.binding import Binding
from textual.widgets import OptionList
from textual.widgets.option_list import Option

from xmr4el_tui.widgets.row_rendering import FullRow, Selectable

HEAD = "bold"  # a line wholly in this style is a Description heading: shown, not selectable


def selectable(line):
    """A Description line the cursor stops on: not blank, not a heading."""
    head = str(line.style) == HEAD or any(str(sp.style) == HEAD and sp.start == 0 and sp.end >= len(line.plain)
                                          for sp in line.spans)
    return bool(line.plain.strip()) and not head


class Rows(Selectable, FullRow, OptionList):
    """OptionList with the full-row cursor and mouse text selection (Description, Dataset Setup, file picker)."""
    CURSOR = "option-list--option-highlighted"

    def cursor_at(self, y):
        try:
            return self.highlighted is not None and self._lines[self.scroll_offset.y + y][0] == self.highlighted
        except IndexError:
            return False


class Description(Rows):
    """Read-only lines of the selected item, one option each: up / down (or a click) select a row, headings and
    blank lines are skipped, nothing is editable. left / right leave for Results / Run monitor. The selected row is
    one colour (a green COMPLETED would vanish on the accent)."""
    BINDINGS = [Binding("left", "screen.results", "results"), Binding("right", "screen.monitor", "monitor"),
                Binding("escape", "screen.focus_explorer", "explorer")]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.subject, self.shown, self.keys = None, [], []

    def show(self, subject, text):
        """Show `text` for `subject` (an explorer key). Unchanged content is not redrawn; a refresh of the same
        subject keeps the row with the same field label selected."""
        lines = text.copy()
        lines.rstrip()
        lines = lines.split("\n", allow_blank=True)
        shown = [(line.plain, tuple(line.spans), str(line.style)) for line in lines]
        if subject == self.subject and shown == self.shown:
            return
        old = self.keys[self.highlighted] if subject == self.subject and self.highlighted is not None else None
        self.clear_options()
        self.forget_lines()
        self.add_options([Option(line, disabled=not selectable(line)) for line in lines])
        self.subject, self.shown, self.keys = subject, shown, [line.plain[:14].strip() for line in lines]
        rows = [i for i, line in enumerate(lines) if selectable(line)]
        self.highlighted = next((i for i in rows if self.keys[i] == old), rows[0] if rows else None)

    def text(self):
        return "\n".join(plain for plain, _, _ in self.shown)

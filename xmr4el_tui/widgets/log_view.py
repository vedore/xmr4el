"""Read-only log lines with a level filter and highlighted search matches (the Dashboard's log viewer; the Run
monitor reuses it). Only the rows on screen are rendered, so a large log costs one row per screen line; at most
MAX_LINES lines are kept (the oldest are dropped)."""
import re

from rich.segment import Segment
from rich.style import Style
from rich.text import Text
from textual.geometry import Size
from textual.scroll_view import ScrollView
from textual.strip import Strip

from xmr4el_tui.logs import LEVELS

MAX_LINES = 100_000
FILTERS = (None, "INFO", "WARNING", "ERROR")  # minimum level shown; None = every line
LEVEL_STYLE = {"DEBUG": Style(dim=True), "WARNING": Style(color="yellow"), "ERROR": Style(color="red"),
               "CRITICAL": Style(color="red", bold=True)}
MATCH, CURRENT = Style(bgcolor="#3B6E99"), Style(bgcolor="#2CF5C5", color="#101418", bold=True)


class LogView(ScrollView, can_focus=True):
    """up / down / page up / page down / home / end scroll (ScrollView keys). New lines follow the end while the
    view is at the end."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.all, self.shown, self.width_max = [], [], 0  # shown: indices into `all` passing the filter
        self.min_level, self.query, self.matches, self.current = None, "", [], None  # matches: rows of `shown`

    # --- content ---------------------------------------------------------------------------------------------
    def append(self, lines, reset=False):
        follow = self.scroll_offset.y >= self.max_scroll_y
        if reset:
            self.all, self.width_max = [], 0
        self.all.extend(lines)
        if len(self.all) > MAX_LINES or reset:
            del self.all[:max(0, len(self.all) - MAX_LINES)]
            self.refilter()
        else:
            self.extend_shown(range(len(self.all) - len(lines), len(self.all)))
        self.width_max = max([self.width_max, *(len(line.text) for line in lines)])
        self.virtual_size = Size(self.width_max, len(self.shown))
        self.refresh()
        if follow:
            self.call_after_refresh(self.scroll_end, animate=False)

    def passes(self, line):
        return self.min_level is None or (line.level in LEVELS and
                                          LEVELS.index(line.level) >= LEVELS.index(self.min_level))

    def extend_shown(self, indices):
        start = len(self.shown)
        self.shown.extend(i for i in indices if self.passes(self.all[i]))
        if self.query:
            q = self.query.casefold()
            self.matches.extend(r for r in range(start, len(self.shown)) if q in self.all[self.shown[r]].text.casefold())

    def refilter(self):
        self.shown, self.matches, self.current = [], [], None
        self.extend_shown(range(len(self.all)))
        self.virtual_size = Size(self.width_max, len(self.shown))
        self.refresh()

    def cycle_level(self):
        self.min_level = FILTERS[(FILTERS.index(self.min_level) + 1) % len(FILTERS)]
        self.refilter()
        self.call_after_refresh(self.scroll_end, animate=False)

    # --- search ----------------------------------------------------------------------------------------------
    def search(self, query):
        """Highlight `query` (case-insensitive); the cursor goes to the first match from the top of the view."""
        self.query = query
        self.refilter()
        if self.matches:
            top = self.scroll_offset.y
            self.current = next((i for i, r in enumerate(self.matches) if r >= top), 0)
            self.show_current()

    def step(self, direction):
        if self.matches:
            self.current = 0 if self.current is None else (self.current + direction) % len(self.matches)
            self.show_current()

    def show_current(self):
        self.call_after_refresh(self.scroll_to, y=max(0, self.matches[self.current] - self.size.height // 2),
                                animate=False)
        self.refresh()

    # --- rendering -------------------------------------------------------------------------------------------
    def render_line(self, y):
        x, row, width = self.scroll_offset.x, self.scroll_offset.y + y, self.size.width
        base = self.rich_style
        if row >= len(self.shown):
            return Strip.blank(width, base)
        line = self.all[self.shown[row]]
        text = Text(line.text, style=Style(color=base.color) + LEVEL_STYLE.get(line.level, Style()), no_wrap=True,
                    end="")
        if self.query:
            here = self.current is not None and self.matches[self.current] == row
            text.highlight_regex(re.compile(re.escape(self.query), re.IGNORECASE), CURRENT if here else MATCH)
        segments = list(text.render(self.app.console)) or [Segment("", base)]
        return Strip(segments).crop_extend(x, x + width, None).apply_style(Style(bgcolor=base.bgcolor))

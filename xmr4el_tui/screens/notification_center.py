"""Notification center: the session's notifications in a small box under the bell, over the current screen.

A ModalScreen so the screen below keeps its focus, cursor, scroll and comparison as they were; esc (or ctrl+n)
closes it and Textual gives focus back to the widget that had it. enter / click on a row opens its destination.
"""
from datetime import datetime

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import OptionList, Static
from textual.widgets.option_list import Option

from xmr4el_tui import fmt, theme


def age(at, now=None):
    seconds = ((now or datetime.now()) - at).total_seconds()
    if seconds < 60:
        return "Now"
    if seconds < 3600:
        return f"{int(seconds // 60)}m ago"
    return f"{int(seconds // 3600)}h ago" if seconds < 86400 else fmt.stamp(at)


def row(notice, width, now=None):
    """Title chip and age on one line, then subject and description (cut to `width`)."""
    title, when = f" {notice.title} ", age(notice.at, now)
    gap = max(1, width - len(title) - len(when))
    text = Text.assemble((title, theme.chip(notice.severity)), " " * gap, (when, theme.style("text-muted")), "\n",
                         notice.subject or fmt.DASH, "\n", (notice.text, theme.style("text-muted")))
    text.no_wrap, text.overflow = True, "ellipsis"
    return text


class NotificationCenter(ModalScreen):
    BINDINGS = [Binding("escape,ctrl+n", "dismiss", "close"), Binding("c", "clear", "clear all")]

    def compose(self) -> ComposeResult:
        with Vertical(id="nc-box"):
            with Horizontal(id="nc-head"):
                yield Static("Notifications", id="nc-title")
                yield Static("Clear all", id="nc-clear")
            yield OptionList(id="nc-list")
            yield Static(id="nc-empty")
            yield Static(id="nc-foot")

    def on_mount(self):
        self.fill()
        self.query_one(OptionList).focus()

    def fill(self):
        """(Re)draw from the app's history; shown = seen. The highlighted row stays on the same notice."""
        notices, box = self.app.notices, self.query_one(OptionList)
        self.app.notices.seen()
        self.app.refresh_bars()
        keep = box.highlighted_option.id if box.highlighted_option is not None else None
        width = (box.scrollable_content_region.width or 48) - 2
        self.ids = {str(i): n for i, n in enumerate(notices.items)}
        options = []
        for i, n in self.ids.items():
            options += [None] if options else []
            options.append(Option(row(n, width), id=i))
        box.set_options(options)
        if self.ids:
            box.highlighted = box.get_option_index(keep) if keep in self.ids else 0
        box.display = bool(self.ids)
        self.query_one("#nc-clear").display = bool(self.ids)
        self.query_one("#nc-empty", Static).update("All caught up\nNo notifications in this session.")
        self.query_one("#nc-empty").display = not self.ids
        keys = "↑↓ move · enter open · c clear all · esc close" if self.ids else "esc close"
        self.query_one("#nc-foot", Static).update(Text.assemble(("Current session only", theme.style("text-muted")),
                                                                "\n", (keys, theme.style("text-disabled"))))

    def on_option_list_option_selected(self, event):
        notice = self.ids.get(event.option.id)
        self.dismiss()
        if notice is not None:
            self.app.open_notice(notice)

    def on_click(self, event):
        if event.widget is not None and event.widget.id == "nc-clear":
            self.action_clear()
        elif event.widget is self:  # outside the box
            self.dismiss()

    def action_clear(self):
        """Clear all: the session's history only (no file, log, run or result is touched)."""
        self.app.notices.clear()
        self.fill()

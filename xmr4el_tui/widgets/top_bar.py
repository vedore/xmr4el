"""Application header (navbar): breadcrumbs, XMR4EL, notifications."""
from rich.text import Text
from textual.screen import ModalScreen
from textual.widgets import Static

TITLE = "XMR4EL"
BELL = "Notifications"  # text, no emoji
BELL_SHORT = "Notif"    # when the full label and count do not fit right of the title
SEP = " / "


def crumbs(app, screen):
    """[(label, screen, level)] from the bottom of the app's screen stack up to `screen`: each screen's `crumbs`
    (Results adds Inspect while inspecting). Modal screens (dialogs, notification center) add none. The stack is the
    one navigation history: no second one is kept."""
    stack = app.screen_stack
    upto = stack[:stack.index(screen) + 1] if screen in stack else stack
    return [(label, s, level) for s in upto if not isinstance(s, ModalScreen)
            for level, label in enumerate(getattr(s, "crumbs", ()))]


class TopBar(Static):
    """Breadcrumbs left, XMR4EL centred on the full terminal width, `Notifications` (unseen count) right. Shared by
    every screen. A click on an earlier crumb goes back to it; a click on Notifications opens the notification center."""
    COMPONENT_CLASSES = {"topbar--title", "topbar--crumb", "topbar--current", "topbar--bell", "topbar--unseen"}

    def __init__(self, **kwargs):
        super().__init__("", **kwargs)
        self.spans = []  # [(x0, x1, what)] of the last render: a crumb tuple or "bell"

    def render(self):
        width, style = self.size.width, self.get_component_rich_style
        # centred on the app, not the bar (the help panel narrows the bar)
        start = max(0, min(width - len(TITLE), (self.app.size.width - len(TITLE)) // 2 - self.region.x))
        unseen = self.app.notices.unseen
        for label in (BELL, BELL_SHORT):
            bell = Text.assemble((label, style("topbar--bell")),
                                 *((" ", (str(unseen), style("topbar--unseen"))) if unseen else ()), " ")
            bell_x = width - bell.cell_len
            self.spans = [(bell_x, width, "bell")] if bell_x >= start + len(TITLE) + 1 else []
            if self.spans:
                break
        trail = crumbs(self.app, self.screen)
        nav, x = Text(" "), 1
        for i, crumb in enumerate(trail):
            last = i == len(trail) - 1
            if i:
                nav.append(SEP, style("topbar--crumb"))
                x += len(SEP)
            nav.append(crumb[0], style("topbar--current" if last else "topbar--crumb"))
            if not last:
                self.spans.append((x, x + len(crumb[0]), crumb))
            x += len(crumb[0])
        room = start - 1
        if nav.cell_len > room:  # keep the current end of the trail: "… / Results"
            nav = Text.assemble(" …", nav[nav.cell_len - room + 2:]) if room > 3 else Text()
            self.spans = [s for s in self.spans if s[2] == "bell"]
        line = Text.assemble(nav, " " * (start - nav.cell_len), (TITLE, style("topbar--title")))
        if self.spans and self.spans[0][2] == "bell":
            line.append(" " * (bell_x - line.cell_len))
            line.append_text(bell)
        return line

    def on_click(self, event):
        hit = next((what for x0, x1, what in self.spans if x0 <= event.x < x1), None)
        if hit == "bell":
            self.app.action_notifications()
        elif hit is not None:
            self.app.go_to(hit[1], hit[2])

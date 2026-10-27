"""Application header."""
from rich.text import Text
from textual.widgets import Static

TITLE = "XMR4EL"
AREAS = ("Dashboard", "New Experiment", "Run Monitor", "Results")


class TopBar(Static):
    """XMR4EL centred on the full width; the areas on the right, `area` highlighted, dropped when they would reach
    the title. Shared by every screen."""
    COMPONENT_CLASSES = {"topbar--title", "topbar--area", "topbar--current"}

    def __init__(self, area, **kwargs):
        super().__init__("", **kwargs)
        self.area = area

    def render(self):
        width, style = self.size.width, self.get_component_rich_style
        # centred on the app, not the bar (the help panel narrows the bar)
        start = max(0, min(width - len(TITLE), (self.app.size.width - len(TITLE)) // 2 - self.region.x))
        nav = Text(" ").join(Text(f" {a} ", style=style("topbar--current" if a == self.area else "topbar--area"))
                             for a in AREAS)
        gap = width - start - len(TITLE) - nav.cell_len - 1
        return Text.assemble(" " * start, (TITLE, style("topbar--title")), *((" " * gap, nav) if gap >= 2 else ()))
